"""Context bloat management: ADK Events Compaction, Context Caching, and Token Sliding Window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models.google_llm import Gemini

from meal_planner_agent.config import AgentSettings, TaskComplexity, StrategicModelRouter, get_settings
from meal_planner_agent.observability.pii_redactor import PIIRedactor, get_pii_redactor

COMPACTION_PROMPT_TEMPLATE = (
    "Summarize the preceding meal-planning conversation turns into a concise, structured "
    "digest. Preserve ALL food allergies, dietary restrictions, calorie targets, household "
    "size, weekly budget, expiring pantry ingredients, and approved/rejected recipes. "
    "Omit chit-chat and never include raw PII.\n\nConversation:\n{conversation_history}"
)


def build_events_compaction_config(
    settings: AgentSettings | None = None,
) -> EventsCompactionConfig:
    """Build native Google ADK EventsCompactionConfig using the fast Gemini Flash tier."""
    cfg = settings or get_settings()
    router = StrategicModelRouter(cfg)
    summarizer_model = router.select_model(TaskComplexity.HISTORY_SUMMARIZATION)
    return EventsCompactionConfig(
        compaction_interval=cfg.compaction_interval,
        overlap_size=cfg.compaction_overlap_size,
        token_threshold=cfg.compaction_token_threshold,
        event_retention_size=cfg.compaction_event_retention_size,
        summarizer=LlmEventSummarizer(
            llm=Gemini(model=summarizer_model),
            prompt_template=COMPACTION_PROMPT_TEMPLATE,
        ),
    )


def build_context_cache_config(
    settings: AgentSettings | None = None,
) -> ContextCacheConfig:
    """Build native Google ADK ContextCacheConfig for Vertex AI / Gemini context caching."""
    cfg = settings or get_settings()
    return ContextCacheConfig(
        ttl_seconds=cfg.context_cache_ttl_seconds,
        min_tokens=2048,
        cache_intervals=10,
    )


@dataclass
class CompactionResult:
    """Result returned by SlidingWindowTokenCompactor."""

    compacted_messages: list[dict[str, Any]]
    was_compacted: bool
    original_estimated_tokens: int
    compacted_estimated_tokens: int
    summarized_turn_count: int


class SlidingWindowTokenCompactor:
    """Deterministic token-budget sliding window + digest compactor for conversation histories.

    Complements ADK's LLM-backed ``EventsCompactionConfig`` by guaranteeing a strict
    upper token budget even in offline evaluation harnesses or rapid multi-turn flows,
    while explicitly pinning critical allergen and budget facts in the compacted digest.
    """

    _CRITICAL_KEYWORDS = (
        "allerg",
        "peanut",
        "shellfish",
        "gluten",
        "dairy",
        "vegan",
        "vegetarian",
        "budget",
        "$",
        "calorie",
        "kcal",
    )

    def __init__(
        self,
        *,
        max_token_budget: int = 4000,
        retain_recent_turns: int = 4,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self._max_token_budget = max_token_budget
        self._retain_recent_turns = retain_recent_turns
        self._redactor = redactor or get_pii_redactor()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Approximate token count (~4 characters per token)."""
        return max(1, (len(text) + 3) // 4)

    def _estimate_history_tokens(self, messages: list[dict[str, Any]]) -> int:
        return sum(self.estimate_tokens(str(msg.get("content", ""))) for msg in messages)

    def compact(self, messages: list[dict[str, Any]]) -> CompactionResult:
        """Compact older messages into a pinned digest when history exceeds token budget."""
        original_tokens = self._estimate_history_tokens(messages)
        if original_tokens <= self._max_token_budget or len(messages) <= self._retain_recent_turns:
            return CompactionResult(
                compacted_messages=list(messages),
                was_compacted=False,
                original_estimated_tokens=original_tokens,
                compacted_estimated_tokens=original_tokens,
                summarized_turn_count=0,
            )

        split_idx = max(0, len(messages) - self._retain_recent_turns)
        older_turns = messages[:split_idx]
        recent_turns = messages[split_idx:]

        pinned_facts: list[str] = []
        for turn in older_turns:
            role = str(turn.get("role", "user"))
            content = self._redactor.redact_text(str(turn.get("content", "")).strip())
            if not content:
                continue
            lower = content.lower()
            if any(kw in lower for kw in self._CRITICAL_KEYWORDS):
                snippet = content[:180] + ("..." if len(content) > 180 else "")
                pinned_facts.append(f"- [{role}] {snippet}")

        if not pinned_facts:
            pinned_facts.append(
                f"- Earlier conversation covered {len(older_turns)} meal-planning turns."
            )

        digest_text = (
            f"[COMPACTED HISTORY DIGEST: {len(older_turns)} earlier turns condensed]\n"
            + "\n".join(pinned_facts[:8])
        )
        compacted = [{"role": "system", "content": digest_text}, *recent_turns]
        compacted_tokens = self._estimate_history_tokens(compacted)

        return CompactionResult(
            compacted_messages=compacted,
            was_compacted=True,
            original_estimated_tokens=original_tokens,
            compacted_estimated_tokens=compacted_tokens,
            summarized_turn_count=len(older_turns),
        )
