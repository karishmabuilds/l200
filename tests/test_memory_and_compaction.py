"""Tests for Context & Memory: persistent SQLite state, vector search, compaction, and async tasks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from meal_planner_agent.config import AgentSettings
from meal_planner_agent.memory import (
    AsyncMemoryConsolidationWorker,
    PersistentMealMemoryService,
    PersistentMealMemoryStore,
    SlidingWindowTokenCompactor,
    build_context_cache_config,
    build_events_compaction_config,
    build_session_service,
)


class TestMemoryAndCompaction(unittest.IsolatedAsyncioTestCase):
    """Validate persistence, semantic retrieval, context compaction, and async consolidation."""

    async def test_persistent_sqlite_session_service_and_vector_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_db = str(Path(tmpdir) / "sessions.db")
            memory_db = str(Path(tmpdir) / "memory.db")
            settings = AgentSettings(session_db_path=session_db, memory_db_path=memory_db)

            # 1. Persistent ADK SqliteSessionService
            session_service = build_session_service(settings)
            created = await session_service.create_session(
                app_name="meal_planner_agent",
                user_id="household_42",
                session_id="sess_001",
                state={"household_allergens": ["peanut"]},
            )
            fetched = await session_service.get_session(
                app_name="meal_planner_agent",
                user_id="household_42",
                session_id=created.id,
            )
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.state.get("household_allergens"), ["peanut"])

            # 2. Persistent SQLite Vector Memory Store + PII scrubbing
            store = PersistentMealMemoryStore(db_path=memory_db, settings=settings)
            await store.store_memory_chunk(
                app_name="meal_planner_agent",
                user_id="household_42",
                session_id="sess_001",
                author="user",
                raw_text="We love spicy red lentil curry and avoid peanuts. Reach me at chef@example.com.",
            )
            service = PersistentMealMemoryService(store)
            response = await service.search_memory(
                app_name="meal_planner_agent",
                user_id="household_42",
                query="red lentil curry peanut",
            )
            self.assertGreaterEqual(len(response.memories), 1)
            stored_text = response.memories[0].content.parts[0].text
            self.assertIn("[REDACTED_EMAIL]", stored_text)
            self.assertNotIn("chef@example.com", stored_text)

    async def test_async_background_memory_consolidation_does_not_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db = str(Path(tmpdir) / "async_memory.db")
            store = PersistentMealMemoryStore(db_path=memory_db)
            worker = AsyncMemoryConsolidationWorker(store=store)

            task = worker.enqueue_turn_consolidation(
                app_name="meal_planner_agent",
                user_id="household_async",
                session_id="sess_async",
                author="user",
                raw_text=(
                    "I am allergic to shellfish and gluten-free. Keep our weekly budget "
                    "under $120 and target 2100 calories per day. Phone: 415-555-0199."
                ),
            )
            self.assertFalse(task.done())  # Scheduled asynchronously without blocking caller
            await worker.flush()
            self.assertEqual(worker.completed_jobs, 1)

            profile = await store.get_household_profile("household_async")
            self.assertIn("shellfish", profile["allergens"])
            self.assertIn("gluten_free", profile["dietary_restrictions"])
            self.assertEqual(profile["weekly_budget_usd"], 120.0)
            self.assertEqual(profile["daily_calorie_target"], 2100)

    def test_adk_compaction_context_cache_and_sliding_window(self) -> None:
        settings = AgentSettings()
        compaction_cfg = build_events_compaction_config(settings)
        cache_cfg = build_context_cache_config(settings)
        self.assertEqual(compaction_cfg.token_threshold, settings.compaction_token_threshold)
        self.assertEqual(cache_cfg.ttl_seconds, settings.context_cache_ttl_seconds)

        compactor = SlidingWindowTokenCompactor(max_token_budget=120, retain_recent_turns=2)
        history = [
            {"role": "user", "content": "Severe peanut allergy and $90 weekly budget! " + ("x" * 200)},
            {"role": "assistant", "content": "Noted your peanut allergy and budget. " + ("y" * 200)},
            {"role": "user", "content": "Also vegetarian meals please. " + ("z" * 200)},
            {"role": "assistant", "content": "Drafted 5-day vegetarian menu."},
            {"role": "user", "content": "Can we swap Tuesday dinner?"},
        ]
        result = compactor.compact(history)
        self.assertTrue(result.was_compacted)
        self.assertEqual(result.summarized_turn_count, 3)
        self.assertLess(result.compacted_estimated_tokens, result.original_estimated_tokens)
        self.assertIn("peanut", result.compacted_messages[0]["content"].lower())


if __name__ == "__main__":
    unittest.main()
