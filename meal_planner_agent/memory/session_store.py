"""Persistent session state, SQLite vector memory store, and Vertex AI Search / Memory Bank integration."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from meal_planner_agent.config import AgentSettings, get_settings
from meal_planner_agent.observability.pii_redactor import PIIRedactor, get_pii_redactor

_VECTOR_DIMS = 64
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def compute_text_embedding(text: str, dims: int = _VECTOR_DIMS) -> list[float]:
    """Compute a deterministic L2-normalized dense vector for semantic retrieval.

    Used for lightweight persistent SQLite vector similarity search when offline
    or running locally without a remote Vertex AI embedding endpoint.
    """
    vector = [0.0] * dims
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return vector

    # Combine unigrams and character trigrams for robust culinary matching
    features: list[str] = list(tokens)
    for token in tokens:
        if len(token) >= 4:
            features.extend(token[i : i + 3] for i in range(len(token) - 2))

    for feat in features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [round(v / norm, 6) for v in vector]
    return vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


class PersistentMealMemoryStore:
    """Persistent async SQLite + Vector store (with optional Vertex AI Search bridge).

    Stores PII-scrubbed household dietary profiles and episodic meal-planning memories
    with vector embeddings across turns and application restarts.
    """

    def __init__(
        self,
        db_path: str | None = None,
        settings: AgentSettings | None = None,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._db_path = db_path or self._settings.memory_db_path
        self._redactor = redactor or get_pii_redactor()
        self._initialized = False

    async def initialize(self) -> None:
        """Create SQLite persistence tables if they do not already exist."""
        if self._initialized:
            return
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS household_profiles (
                    user_id TEXT PRIMARY KEY,
                    allergens_json TEXT NOT NULL,
                    dietary_restrictions_json TEXT NOT NULL,
                    preferred_cuisines_json TEXT NOT NULL,
                    weekly_budget_usd REAL,
                    daily_calorie_target INTEGER,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    author TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_app_user
                ON episodic_memories (app_name, user_id)
                """
            )
            await db.commit()
        self._initialized = True

    async def upsert_household_profile(
        self,
        user_id: str,
        *,
        allergens: list[str] | None = None,
        dietary_restrictions: list[str] | None = None,
        preferred_cuisines: list[str] | None = None,
        weekly_budget_usd: float | None = None,
        daily_calorie_target: int | None = None,
    ) -> dict[str, Any]:
        """Persist or merge household dietary preferences cleanly (PII-scrubbed)."""
        await self.initialize()
        existing = await self.get_household_profile(user_id)
        merged_allergens = sorted(
            set(existing.get("allergens", []))
            | {self._redactor.redact_text(a).lower().strip() for a in (allergens or []) if a.strip()}
        )
        merged_diets = sorted(
            set(existing.get("dietary_restrictions", []))
            | {
                self._redactor.redact_text(d).lower().strip()
                for d in (dietary_restrictions or [])
                if d.strip()
            }
        )
        merged_cuisines = sorted(
            set(existing.get("preferred_cuisines", []))
            | {
                self._redactor.redact_text(c).lower().strip()
                for c in (preferred_cuisines or [])
                if c.strip()
            }
        )
        budget = weekly_budget_usd if weekly_budget_usd is not None else existing.get("weekly_budget_usd")
        calories = (
            daily_calorie_target
            if daily_calorie_target is not None
            else existing.get("daily_calorie_target")
        )
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO household_profiles (
                    user_id, allergens_json, dietary_restrictions_json,
                    preferred_cuisines_json, weekly_budget_usd, daily_calorie_target, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    allergens_json = excluded.allergens_json,
                    dietary_restrictions_json = excluded.dietary_restrictions_json,
                    preferred_cuisines_json = excluded.preferred_cuisines_json,
                    weekly_budget_usd = excluded.weekly_budget_usd,
                    daily_calorie_target = excluded.daily_calorie_target,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    json.dumps(merged_allergens),
                    json.dumps(merged_diets),
                    json.dumps(merged_cuisines),
                    budget,
                    calories,
                    now,
                ),
            )
            await db.commit()

        return {
            "user_id": user_id,
            "allergens": merged_allergens,
            "dietary_restrictions": merged_diets,
            "preferred_cuisines": merged_cuisines,
            "weekly_budget_usd": budget,
            "daily_calorie_target": calories,
            "updated_at": now,
        }

    async def get_household_profile(self, user_id: str) -> dict[str, Any]:
        """Load persisted dietary profile for a household user."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT allergens_json, dietary_restrictions_json, preferred_cuisines_json,
                       weekly_budget_usd, daily_calorie_target, updated_at
                FROM household_profiles WHERE user_id = ?
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
        if not row:
            return {
                "user_id": user_id,
                "allergens": [],
                "dietary_restrictions": [],
                "preferred_cuisines": [],
                "weekly_budget_usd": None,
                "daily_calorie_target": None,
                "updated_at": None,
            }
        return {
            "user_id": user_id,
            "allergens": json.loads(row[0]),
            "dietary_restrictions": json.loads(row[1]),
            "preferred_cuisines": json.loads(row[2]),
            "weekly_budget_usd": row[3],
            "daily_calorie_target": row[4],
            "updated_at": row[5],
        }

    async def store_memory_chunk(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        author: str,
        raw_text: str,
    ) -> int:
        """Scrub PII, embed text, and persist a memory chunk into SQLite vector storage."""
        await self.initialize()
        scrubbed_text = self._redactor.redact_text(raw_text).strip()
        if not scrubbed_text:
            return -1
        embedding = compute_text_embedding(scrubbed_text)
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO episodic_memories (
                    app_name, user_id, session_id, author, content_text, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_name,
                    user_id,
                    session_id,
                    author,
                    scrubbed_text,
                    json.dumps(embedding),
                    now,
                ),
            )
            await db.commit()
            return int(cursor.lastrowid or -1)

    async def semantic_search(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Retrieve top-K semantically similar memories via cosine similarity (or Vertex AI Search)."""
        await self.initialize()

        # Optional Vertex AI Search query when configured on GCP
        vertex_hits = self._search_vertex_ai_datastore(query=query, top_k=top_k)

        query_vec = compute_text_embedding(self._redactor.redact_text(query))
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT id, session_id, author, content_text, embedding_json, created_at
                FROM episodic_memories
                WHERE app_name = ? AND user_id = ?
                ORDER BY id DESC
                LIMIT 250
                """,
                (app_name, user_id),
            )
            rows = await cursor.fetchall()

        scored: list[dict[str, Any]] = []
        for row in rows:
            vec = json.loads(row[4])
            score = cosine_similarity(query_vec, vec)
            if score >= min_score:
                scored.append(
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "author": row[2],
                        "content_text": row[3],
                        "score": round(score, 4),
                        "created_at": row[5],
                        "source": "sqlite_vector_store",
                    }
                )

        scored.sort(key=lambda item: item["score"], reverse=True)
        combined = scored[:top_k] + vertex_hits
        return combined[:top_k]

    def _search_vertex_ai_datastore(self, *, query: str, top_k: int) -> list[dict[str, Any]]:
        if not (
            self._settings.google_cloud_project
            and self._settings.vertex_ai_search_datastore_id
        ):
            return []
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine  # type: ignore[import-untyped]

            client = discoveryengine.SearchServiceClient()
            serving_config = (
                f"projects/{self._settings.google_cloud_project}"
                f"/locations/global/collections/default_collection"
                f"/dataStores/{self._settings.vertex_ai_search_datastore_id}"
                f"/servingConfigs/default_config"
            )
            request = discoveryengine.SearchRequest(
                serving_config=serving_config,
                query=self._redactor.redact_text(query),
                page_size=top_k,
            )
            response = client.search(request)
            hits: list[dict[str, Any]] = []
            for result in response.results:
                doc_data = dict(result.document.derived_struct_data)
                snippet = str(doc_data.get("title") or doc_data.get("link") or result.id)
                hits.append(
                    {
                        "id": result.id,
                        "session_id": "vertex_ai_search",
                        "author": "knowledge_base",
                        "content_text": self._redactor.redact_text(snippet),
                        "score": 0.9,
                        "created_at": datetime.now(UTC).isoformat(),
                        "source": "vertex_ai_search",
                    }
                )
            return hits
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                "Vertex AI Search unavailable, using local SQLite vector store: %s", exc
            )
            return []


class PersistentMealMemoryService(BaseMemoryService):
    """ADK-compatible BaseMemoryService backed by PersistentMealMemoryStore."""

    def __init__(self, store: PersistentMealMemoryStore | None = None) -> None:
        self._store = store or PersistentMealMemoryStore()

    @property
    def store(self) -> PersistentMealMemoryStore:
        return self._store

    async def add_session_to_memory(self, session: Session) -> None:
        """Persist PII-scrubbed session events into the SQLite vector memory store."""
        for event in getattr(session, "events", []) or []:
            content = getattr(event, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            text = " ".join(
                part.text for part in content.parts if getattr(part, "text", None)
            ).strip()
            if text:
                await self._store.store_memory_chunk(
                    app_name=session.app_name,
                    user_id=session.user_id,
                    session_id=session.id,
                    author=getattr(event, "author", "user") or "user",
                    raw_text=text,
                )

    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Search persistent semantic memory and return ADK MemoryEntry items."""
        hits = await self._store.semantic_search(
            app_name=app_name,
            user_id=user_id,
            query=query,
            top_k=5,
        )
        entries = [
            MemoryEntry(
                content=types.Content(
                    role="user", parts=[types.Part.from_text(text=hit["content_text"])]
                ),
                author=hit["author"],
                timestamp=hit["created_at"],
            )
            for hit in hits
        ]
        return SearchMemoryResponse(memories=entries)


def build_session_service(settings: AgentSettings | None = None) -> BaseSessionService:
    """Create a persistent ADK SqliteSessionService backed by aiosqlite."""
    cfg = settings or get_settings()
    cfg.ensure_data_dirs()
    return SqliteSessionService(db_path=cfg.session_db_path)


def build_memory_service(settings: AgentSettings | None = None) -> BaseMemoryService:
    """Build VertexAiMemoryBankService on GCP when configured, or local SQLite vector memory."""
    cfg = settings or get_settings()
    if cfg.google_cloud_project and cfg.vertex_ai_memory_bank_engine_id:
        return VertexAiMemoryBankService(
            project=cfg.google_cloud_project,
            location=cfg.google_cloud_location,
            agent_engine_id=cfg.vertex_ai_memory_bank_engine_id,
        )
    cfg.ensure_data_dirs()
    return PersistentMealMemoryService(PersistentMealMemoryStore(db_path=cfg.memory_db_path, settings=cfg))
