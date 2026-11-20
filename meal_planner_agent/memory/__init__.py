"""Context compaction, persistent session/vector memory, and async consolidation exports."""

from meal_planner_agent.memory.async_memory import (
    AsyncMemoryConsolidationPlugin,
    AsyncMemoryConsolidationWorker,
    extract_household_preferences,
)
from meal_planner_agent.memory.compaction import (
    CompactionResult,
    SlidingWindowTokenCompactor,
    build_context_cache_config,
    build_events_compaction_config,
)
from meal_planner_agent.memory.session_store import (
    PersistentMealMemoryService,
    PersistentMealMemoryStore,
    build_memory_service,
    build_session_service,
    compute_text_embedding,
    cosine_similarity,
)

__all__ = [
    "AsyncMemoryConsolidationPlugin",
    "AsyncMemoryConsolidationWorker",
    "CompactionResult",
    "PersistentMealMemoryService",
    "PersistentMealMemoryStore",
    "SlidingWindowTokenCompactor",
    "build_context_cache_config",
    "build_events_compaction_config",
    "build_memory_service",
    "build_session_service",
    "compute_text_embedding",
    "cosine_similarity",
    "extract_household_preferences",
]
