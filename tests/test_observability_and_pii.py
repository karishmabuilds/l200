"""Tests for Observability & Tracing: JSON logs, Intent vs Outcome, OTel spans, and PII redaction."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace

from meal_planner_agent.observability import (
    ObservabilityAndTracingPlugin,
    PIIRedactor,
    get_in_memory_span_exporter,
    get_structured_logger,
)


class TestObservabilityAndPII(unittest.IsolatedAsyncioTestCase):
    """Verify structured JSON logging, intent/outcome logs, linked OTel spans, and PII scrubbing."""

    def test_pii_redactor_scrubs_text_and_nested_payloads(self) -> None:
        redactor = PIIRedactor()
        synthetic_key = "AIza" + "SyDummyTestKey" + ("0" * 21)
        raw = (
            "Deliver groceries to 1600 Amphitheatre Parkway for user@example.com, "
            "call 650-555-1234, SSN 123-45-6789, card 4111-1111-1111-1111, "
            f"key {synthetic_key}."
        )
        scrubbed = redactor.redact_text(raw)
        self.assertNotIn("1600 Amphitheatre Parkway", scrubbed)
        self.assertNotIn("user@example.com", scrubbed)
        self.assertNotIn("650-555-1234", scrubbed)
        self.assertNotIn("123-45-6789", scrubbed)
        self.assertNotIn("4111-1111-1111-1111", scrubbed)
        self.assertNotIn(synthetic_key, scrubbed)
        self.assertIn("[REDACTED_STREET_ADDRESS]", scrubbed)
        self.assertIn("[REDACTED_EMAIL]", scrubbed)
        self.assertIn("[REDACTED_PHONE]", scrubbed)
        self.assertIn("[REDACTED_SSN]", scrubbed)
        self.assertIn("[REDACTED_API_KEY]", scrubbed)

        nested = redactor.redact_payload(
            {"email": "hidden@corp.com", "notes": "Call 408-555-9999"}
        )
        self.assertEqual(nested["email"], "[REDACTED_SENSITIVE_FIELD]")
        self.assertIn("[REDACTED_PHONE]", nested["notes"])

    async def test_observability_plugin_captures_intent_vs_outcome_and_linked_otel_spans(self) -> None:
        exporter = get_in_memory_span_exporter()
        exporter.clear()

        log_buffer = io.StringIO()
        logger = get_structured_logger("test.observability.plugin", stream=log_buffer)
        plugin = ObservabilityAndTracingPlugin(logger=logger)

        inv_ctx = SimpleNamespace(
            invocation_id="inv_trace_1",
            user_id="household_99",
            session=SimpleNamespace(id="sess_trace_1"),
        )
        agent_obj = SimpleNamespace(name="chef_nutrition_planner")
        cb_ctx = SimpleNamespace(
            invocation_id="inv_trace_1",
            agent_name="chef_nutrition_planner",
            session=SimpleNamespace(id="sess_trace_1"),
        )
        tool_obj = SimpleNamespace(name="audit_meal_plan_allergen_safety")
        tool_ctx = SimpleNamespace(
            agent_name="meal_plan_safety_auditor",
            function_call_id="call_trace_1",
        )

        # Simulate nested run -> agent -> tool lifecycle
        await plugin.before_run_callback(invocation_context=inv_ctx)
        await plugin.before_agent_callback(agent=agent_obj, callback_context=cb_ctx)
        await plugin.before_tool_callback(
            tool=tool_obj,
            tool_args={
                "household_allergens": ["peanut"],
                "contact_note": "Notify alice@example.com",
            },
            tool_context=tool_ctx,
        )
        await plugin.after_tool_callback(
            tool=tool_obj,
            tool_args={"household_allergens": ["peanut"]},
            tool_context=tool_ctx,
            result={"status": "ok", "summary": "All recipes safe."},
        )
        await plugin.after_agent_callback(agent=agent_obj, callback_context=cb_ctx)
        await plugin.after_run_callback(invocation_context=inv_ctx)

        # 1. Verify OpenTelemetry spans exist and share the SAME distributed trace_id
        spans = exporter.get_finished_spans()
        span_names = {s.name for s in spans}
        self.assertIn("concierge.run", span_names)
        self.assertIn("agent.chef_nutrition_planner", span_names)
        self.assertIn("tool.audit_meal_plan_allergen_safety", span_names)
        trace_ids = {s.context.trace_id for s in spans}
        self.assertEqual(len(trace_ids), 1, "Child spans must link to the root run trace_id")

        # 2. Verify valid structured JSON logs with Intent vs. Outcome & PII scrubbing
        log_lines = [line for line in log_buffer.getvalue().strip().splitlines() if line]
        parsed = [json.loads(line) for line in log_lines]
        event_types = [entry["event_type"] for entry in parsed]
        self.assertIn("run_intent", event_types)
        self.assertIn("tool_intent", event_types)
        self.assertIn("tool_outcome", event_types)
        self.assertIn("run_outcome", event_types)

        tool_intent_log = next(e for e in parsed if e["event_type"] == "tool_intent")
        tool_outcome_log = next(e for e in parsed if e["event_type"] == "tool_outcome")
        self.assertEqual(tool_intent_log["intended_action"], "invoke_tool:audit_meal_plan_allergen_safety")
        self.assertEqual(tool_outcome_log["actual_outcome"], "ok")
        self.assertTrue(tool_outcome_log["intent_outcome_aligned"])
        self.assertIn("trace_id", tool_intent_log)
        self.assertNotIn("alice@example.com", json.dumps(tool_intent_log))


if __name__ == "__main__":
    unittest.main()
