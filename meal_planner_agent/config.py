"""Typed application settings and strategic model routing configuration."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ModelTier(str, Enum):
    """Logical model capability tiers for cost/latency-optimized routing."""

    FAST = "fast"
    REASONING = "reasoning"


class TaskComplexity(str, Enum):
    """Task categories mapped to appropriate LLM tiers."""

    PANTRY_LOOKUP = "pantry_lookup"
    PROFILE_RETRIEVAL = "profile_retrieval"
    GROCERY_LIST_OPTIMIZATION = "grocery_list_optimization"
    HISTORY_SUMMARIZATION = "history_summarization"
    MEAL_PLAN_SYNTHESIS = "meal_plan_synthesis"
    ALLERGEN_SAFETY_AUDIT = "allergen_safety_audit"
    ORCHESTRATION = "orchestration"


class AgentSettings(BaseModel):
    """Centralized, validated runtime configuration for the Meal Planning Concierge."""

    app_name: str = Field(default="meal_planner_agent")
    google_cloud_project: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    google_cloud_location: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    )
    use_vertex_ai: bool = Field(
        default_factory=lambda: os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    )

    # Secret Manager secret IDs (never hardcoded secret values)
    gemini_api_key_secret_id: str = Field(
        default_factory=lambda: os.getenv(
            "GEMINI_API_KEY_SECRET_ID", "meal-planner-gemini-api-key"
        )
    )
    grocery_partner_secret_id: str = Field(
        default_factory=lambda: os.getenv(
            "GROCERY_PARTNER_API_SECRET_ID", "meal-planner-grocery-partner-key"
        )
    )

    # Strategic Model Routing defaults
    fast_model_name: str = Field(
        default_factory=lambda: os.getenv("FAST_MODEL_NAME", "gemini-3.8-flash")
    )
    reasoning_model_name: str = Field(
        default_factory=lambda: os.getenv("REASONING_MODEL_NAME", "gemini-3.1-pro-preview")
    )

    # Persistent Storage & Vertex AI Search / Memory Bank
    session_db_path: str = Field(
        default_factory=lambda: os.getenv("SESSION_DB_PATH", "data/meal_planner_sessions.db")
    )
    memory_db_path: str = Field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", "data/meal_planner_memory.db")
    )
    vertex_ai_search_datastore_id: str = Field(
        default_factory=lambda: os.getenv("VERTEX_AI_SEARCH_DATASTORE_ID", "")
    )
    vertex_ai_memory_bank_engine_id: str = Field(
        default_factory=lambda: os.getenv("VERTEX_AI_MEMORY_BANK_ENGINE_ID", "")
    )

    # Context Compaction & Caching thresholds
    compaction_interval: int = Field(
        default_factory=lambda: int(os.getenv("COMPACTION_INTERVAL", "6")), ge=2
    )
    compaction_overlap_size: int = Field(
        default_factory=lambda: int(os.getenv("COMPACTION_OVERLAP_SIZE", "2")), ge=0
    )
    compaction_token_threshold: int = Field(
        default_factory=lambda: int(os.getenv("COMPACTION_TOKEN_THRESHOLD", "8000")), ge=1000
    )
    compaction_event_retention_size: int = Field(
        default_factory=lambda: int(os.getenv("COMPACTION_EVENT_RETENTION_SIZE", "4")), ge=1
    )
    context_cache_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CONTEXT_CACHE_TTL_SECONDS", "1800")), ge=60
    )

    # Safety Guardrails & HITL Thresholds
    hitl_order_approval_threshold_usd: float = Field(
        default_factory=lambda: float(os.getenv("HITL_ORDER_APPROVAL_THRESHOLD_USD", "0.01")),
        ge=0.0,
    )
    max_weekly_grocery_budget_usd: float = Field(
        default_factory=lambda: float(os.getenv("MAX_WEEKLY_GROCERY_BUDGET_USD", "500.0")),
        gt=0.0,
    )
    enable_cloud_dlp_redaction: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_CLOUD_DLP_REDACTION", "false").lower() == "true"
    )

    # Observability & Distributed Tracing
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()  # type: ignore[arg-type]
    )
    otel_service_name: str = Field(
        default_factory=lambda: os.getenv("OTEL_SERVICE_NAME", "meal-planning-concierge-agent")
    )
    otel_exporter_otlp_endpoint: str = Field(
        default_factory=lambda: os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    )
    enable_cloud_trace: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_CLOUD_TRACE", "false").lower() == "true"
    )

    def ensure_data_dirs(self) -> None:
        """Create local directories for SQLite session and vector memory stores."""
        for raw_path in (self.session_db_path, self.memory_db_path):
            if raw_path and raw_path != ":memory:":
                Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


class StrategicModelRouter:
    """Routes agent subtasks between fast (Flash) and deep reasoning (Pro) models."""

    _ROUTING_TABLE: ClassVar[dict[TaskComplexity, ModelTier]] = {
        TaskComplexity.PANTRY_LOOKUP: ModelTier.FAST,
        TaskComplexity.PROFILE_RETRIEVAL: ModelTier.FAST,
        TaskComplexity.GROCERY_LIST_OPTIMIZATION: ModelTier.FAST,
        TaskComplexity.HISTORY_SUMMARIZATION: ModelTier.FAST,
        TaskComplexity.MEAL_PLAN_SYNTHESIS: ModelTier.REASONING,
        TaskComplexity.ALLERGEN_SAFETY_AUDIT: ModelTier.REASONING,
        TaskComplexity.ORCHESTRATION: ModelTier.REASONING,
    }

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self._settings = settings or get_settings()

    def select_model(self, task: TaskComplexity) -> str:
        """Select the optimal Gemini model identifier for a given task complexity."""
        tier = self._ROUTING_TABLE[task]
        if tier is ModelTier.FAST:
            return self._settings.fast_model_name
        return self._settings.reasoning_model_name

    def select_model_for_prompt(self, user_prompt: str) -> tuple[str, ModelTier, str]:
        """Dynamically classify an ad-hoc request to route between Flash and Pro."""
        lower = user_prompt.lower()
        complex_signals = (
            "allergy",
            "allergies",
            "anaphyl",
            "medical",
            "diabetic",
            "macro",
            "weekly plan",
            "7-day",
            "optimize",
            "compare",
            "budget",
            "family of",
        )
        if any(signal in lower for signal in complex_signals) or len(user_prompt) > 220:
            return (
                self._settings.reasoning_model_name,
                ModelTier.REASONING,
                "Multi-constraint nutritional/allergen planning routed to reasoning tier.",
            )
        return (
            self._settings.fast_model_name,
            ModelTier.FAST,
            "Quick lookup or lightweight household request routed to fast tier.",
        )


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    """Return cached runtime settings."""
    return AgentSettings()
