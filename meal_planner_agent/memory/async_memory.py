"""Non-blocking asynchronous memory consolidation worker and ADK plugin."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin

from meal_planner_agent.memory.session_store import PersistentMealMemoryStore
from meal_planner_agent.observability.pii_redactor import PIIRedactor, get_pii_redactor
from meal_planner_agent.observability.structured_logging import (
    get_structured_logger,
    log_structured_event,
)

_ALLERGEN_PATTERN = re.compile(
    r"\b(?:allergic to|allergy to|no|avoid)\s+(peanuts?|tree nuts?|shellfish|shrimp|dairy|milk|eggs?|soy|wheat|gluten|fish)\b",
    re.IGNORECASE,
)
_DIET_PATTERN = re.compile(
    r"\b(vegan|vegetarian|pescatarian|gluten[- ]free|dairy[- ]free)\b",
    re.IGNORECASE,
)
_BUDGET_PATTERN = re.compile(r"\$(\d{2,4}(?:\.\d{1,2})?)")
_CALORIE_PATTERN = re.compile(r"\b(\d{4})\s*(?:kcal|calories|cals)\b", re.IGNORECASE)


def extract_household_preferences(text: str) -> dict[str, Any]:
    """Deterministically extract dietary constraints, allergies, budget, and calorie goals."""
    allergens = sorted({m.group(1).lower().rstrip("s") for m in _ALLERGEN_PATTERN.finditer(text)})
    diets = sorted(
        {m.group(1).lower().replace("-", "_").replace(" ", "_") for m in _DIET_PATTERN.finditer(text)}
    )
    budget_match = _BUDGET_PATTERN.search(text)
    calorie_match = _CALORIE_PATTERN.search(text)
    return {
        "allergens": allergens,
        "dietary_restrictions": diets,
        "weekly_budget_usd": float(budget_match.group(1)) if budget_match else None,
        "daily_calorie_target": int(calorie_match.group(1)) if calorie_match else None,
    }


class AsyncMemoryConsolidationWorker:
    """Executes expensive memory extraction & vector indexing in background async tasks.

    Prevents UI/response blocking by dispatching preference extraction, PII scrubbing,
    and SQLite/vector persistence via ``asyncio.create_task``. Tracks pending background
    tasks so callers can cleanly await completion during shutdown or tests.
    """

    def __init__(
        self,
        store: PersistentMealMemoryStore | None = None,
        redactor: PIIRedactor | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._store = store or PersistentMealMemoryStore()
        self._redactor = redactor or get_pii_redactor()
        self._logger = logger or get_structured_logger("meal_planner_agent.memory")
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self.completed_jobs: int = 0

    def enqueue_turn_consolidation(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        author: str,
        raw_text: str,
    ) -> asyncio.Task[Any]:
        """Schedule non-blocking background memory consolidation for a conversation turn."""
        task = asyncio.create_task(
            self._consolidate_turn(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
                author=author,
                raw_text=raw_text,
            ),
            name=f"memory_consolidate:{session_id}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self._pending_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log_structured_event(
                self._logger,
                "async_memory_error",
                f"Background memory consolidation failed: {exc}",
                level=logging.ERROR,
            )
        else:
            self.completed_jobs += 1

    async def _consolidate_turn(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        author: str,
        raw_text: str,
    ) -> dict[str, Any]:
        scrubbed = self._redactor.redact_text(raw_text)
        extracted = extract_household_preferences(scrubbed)

        chunk_id = await self._store.store_memory_chunk(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            author=author,
            raw_text=scrubbed,
        )

        profile = None
        if any(
            [
                extracted["allergens"],
                extracted["dietary_restrictions"],
                extracted["weekly_budget_usd"] is not None,
                extracted["daily_calorie_target"] is not None,
            ]
        ):
            profile = await self._store.upsert_household_profile(
                user_id,
                allergens=extracted["allergens"],
                dietary_restrictions=extracted["dietary_restrictions"],
                weekly_budget_usd=extracted["weekly_budget_usd"],
                daily_calorie_target=extracted["daily_calorie_target"],
            )

        log_structured_event(
            self._logger,
            "async_memory_consolidated",
            "Completed background memory consolidation and vector indexing.",
            user_id=user_id,
            session_id=session_id,
            chunk_id=chunk_id,
            updated_profile=bool(profile),
        )
        return {"chunk_id": chunk_id, "profile": profile}

    @property
    def pending_count(self) -> int:
        return len(self._pending_tasks)

    async def flush(self) -> None:
        """Await all currently scheduled background memory consolidation tasks."""
        if not self._pending_tasks:
            return
        await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)


class AsyncMemoryConsolidationPlugin(BasePlugin):
    """ADK Plugin that dispatches non-blocking memory consolidation after each run."""

    def __init__(
        self,
        worker: AsyncMemoryConsolidationWorker | None = None,
        name: str = "async_memory_consolidation_plugin",
    ) -> None:
        super().__init__(name=name)
        self._worker = worker or AsyncMemoryConsolidationWorker()

    @property
    def worker(self) -> AsyncMemoryConsolidationWorker:
        return self._worker

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        """Kick off background consolidation for recent turn content without blocking UI."""
        session = invocation_context.session
        events = getattr(session, "events", None) or []
        if not events:
            return

        # Consolidate the latest user/agent text events asynchronously
        recent_snippets: list[str] = []
        for event in events[-2:]:
            content = getattr(event, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            text = " ".join(
                part.text for part in content.parts if getattr(part, "text", None)
            ).strip()
            if text:
                recent_snippets.append(f"{getattr(event, 'author', 'user')}: {text}")

        if recent_snippets:
            self._worker.enqueue_turn_consolidation(
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                author="turn_summary",
                raw_text="\n".join(recent_snippets),
            )
