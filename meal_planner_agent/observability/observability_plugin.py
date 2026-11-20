"""ADK Observability Plugin capturing Distributed Traces and Intent vs. Outcome logs."""

from __future__ import annotations

import logging
import time
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from meal_planner_agent.observability.pii_redactor import PIIRedactor, get_pii_redactor
from meal_planner_agent.observability.structured_logging import (
    get_structured_logger,
    log_structured_event,
)
from meal_planner_agent.observability.tracing import get_tracer


class ObservabilityAndTracingPlugin(BasePlugin):
    """ADK Plugin delivering OpenTelemetry spans + Intent vs. Outcome JSON telemetry.

    For every run, sub-agent invocation, LLM request, and tool execution, this plugin:
    1. Opens and links OpenTelemetry spans from root query to final answer.
    2. Explicitly records the agent's **INTENT** before execution (target agent/tool,
       goal, and PII-scrubbed arguments).
    3. Explicitly records the actual **OUTCOME** after execution (status, latency_ms,
       error flags, and whether outcome matched intent).
    """

    def __init__(
        self,
        name: str = "observability_and_tracing_plugin",
        logger: logging.Logger | None = None,
        redactor: PIIRedactor | None = None,
    ) -> None:
        super().__init__(name=name)
        self._logger = logger or get_structured_logger("meal_planner_agent.telemetry")
        self._redactor = redactor or get_pii_redactor()
        self._tracer = get_tracer("meal_planner_agent.plugin")
        self._active_spans: dict[str, tuple[Span, Any, float, dict[str, Any]]] = {}

    def _start_tracked_span(
        self,
        key: str,
        span_name: str,
        attributes: dict[str, Any],
        intent_record: dict[str, Any],
    ) -> None:
        span = self._tracer.start_span(span_name)
        for attr_key, attr_val in attributes.items():
            if attr_val is not None:
                span.set_attribute(attr_key, str(attr_val))
        token = otel_context.attach(trace.set_span_in_context(span))
        self._active_spans[key] = (span, token, time.perf_counter(), intent_record)

    def _finish_tracked_span(
        self,
        key: str,
        *,
        outcome_status: str,
        error_message: str | None = None,
    ) -> tuple[float, dict[str, Any]]:
        entry = self._active_spans.pop(key, None)
        if entry is None:
            return 0.0, {}
        span, token, start_time, intent_record = entry
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        span.set_attribute("outcome.status", outcome_status)
        span.set_attribute("latency_ms", latency_ms)
        if error_message:
            span.set_status(Status(StatusCode.ERROR, self._redactor.redact_text(error_message)))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
        try:
            otel_context.detach(token)
        except Exception:
            pass
        return latency_ms, intent_record

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        key = f"run:{invocation_context.invocation_id}"
        intent = {
            "session_id": invocation_context.session.id,
            "user_id": invocation_context.user_id,
            "intended_action": "execute_concierge_turn",
        }
        self._start_tracked_span(
            key,
            "concierge.run",
            {
                "adk.invocation_id": invocation_context.invocation_id,
                "adk.session_id": invocation_context.session.id,
            },
            intent,
        )
        log_structured_event(
            self._logger,
            "run_intent",
            "Starting Meal Planning Concierge invocation.",
            **intent,
        )
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        key = f"run:{invocation_context.invocation_id}"
        latency_ms, intent = self._finish_tracked_span(key, outcome_status="completed")
        log_structured_event(
            self._logger,
            "run_outcome",
            "Completed Meal Planning Concierge invocation.",
            session_id=invocation_context.session.id,
            intended_action=intent.get("intended_action", "execute_concierge_turn"),
            actual_outcome="completed",
            intent_outcome_aligned=True,
            latency_ms=latency_ms,
        )

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        key = f"agent:{callback_context.invocation_id}:{agent.name}"
        intent = {
            "agent_name": agent.name,
            "session_id": callback_context.session.id,
            "intended_action": f"delegate_to_{agent.name}",
        }
        self._start_tracked_span(
            key,
            f"agent.{agent.name}",
            {
                "adk.agent.name": agent.name,
                "adk.session_id": callback_context.session.id,
            },
            intent,
        )
        log_structured_event(
            self._logger,
            "agent_intent",
            f"Agent '{agent.name}' intends to execute step.",
            **intent,
        )
        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> types.Content | None:
        key = f"agent:{callback_context.invocation_id}:{agent.name}"
        latency_ms, intent = self._finish_tracked_span(key, outcome_status="completed")
        log_structured_event(
            self._logger,
            "agent_outcome",
            f"Agent '{agent.name}' finished step.",
            agent_name=agent.name,
            session_id=callback_context.session.id,
            intended_action=intent.get("intended_action", f"delegate_to_{agent.name}"),
            actual_outcome="completed",
            intent_outcome_aligned=True,
            latency_ms=latency_ms,
        )
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        key = f"llm:{callback_context.invocation_id}:{callback_context.agent_name}"
        model_name = getattr(llm_request, "model", "unknown")
        intent = {
            "agent_name": callback_context.agent_name,
            "model": model_name,
            "intended_action": f"llm_inference:{model_name}",
        }
        self._start_tracked_span(
            key,
            f"llm.generate.{callback_context.agent_name}",
            {"gen_ai.request.model": model_name, "adk.agent.name": callback_context.agent_name},
            intent,
        )
        log_structured_event(
            self._logger,
            "llm_intent",
            f"Invoking model '{model_name}' for agent '{callback_context.agent_name}'.",
            **intent,
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        key = f"llm:{callback_context.invocation_id}:{callback_context.agent_name}"
        has_error = bool(getattr(llm_response, "error_code", None))
        status_label = "error" if has_error else "ok"
        latency_ms, intent = self._finish_tracked_span(
            key,
            outcome_status=status_label,
            error_message=getattr(llm_response, "error_message", None),
        )
        log_structured_event(
            self._logger,
            "llm_outcome",
            f"Model call completed for agent '{callback_context.agent_name}'.",
            agent_name=callback_context.agent_name,
            intended_action=intent.get("intended_action"),
            actual_outcome=status_label,
            intent_outcome_aligned=not has_error,
            latency_ms=latency_ms,
        )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        call_id = getattr(tool_context, "function_call_id", None) or tool.name
        key = f"tool:{call_id}"
        scrubbed_args = self._redactor.redact_payload(tool_args)
        intent = {
            "agent_name": tool_context.agent_name,
            "tool_name": tool.name,
            "intended_action": f"invoke_tool:{tool.name}",
            "intended_args": scrubbed_args,
        }
        self._start_tracked_span(
            key,
            f"tool.{tool.name}",
            {
                "adk.tool.name": tool.name,
                "adk.agent.name": tool_context.agent_name,
            },
            intent,
        )
        log_structured_event(
            self._logger,
            "tool_intent",
            f"Agent intends to execute tool '{tool.name}'.",
            **intent,
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        call_id = getattr(tool_context, "function_call_id", None) or tool.name
        key = f"tool:{call_id}"
        scrubbed_result = self._redactor.redact_payload(result)
        actual_status = (
            scrubbed_result.get("status", "ok") if isinstance(scrubbed_result, dict) else "ok"
        )
        is_error = actual_status == "error"
        error_msg = scrubbed_result.get("message") if is_error and isinstance(scrubbed_result, dict) else None
        latency_ms, intent = self._finish_tracked_span(
            key, outcome_status=str(actual_status), error_message=error_msg
        )
        log_structured_event(
            self._logger,
            "tool_outcome",
            f"Tool '{tool.name}' finished with status '{actual_status}'.",
            agent_name=tool_context.agent_name,
            tool_name=tool.name,
            intended_action=intent.get("intended_action", f"invoke_tool:{tool.name}"),
            intended_args=intent.get("intended_args", {}),
            actual_outcome=actual_status,
            outcome_summary=scrubbed_result.get("summary")
            or scrubbed_result.get("message")
            if isinstance(scrubbed_result, dict)
            else None,
            intent_outcome_aligned=actual_status in {"ok", "confirmed", "awaiting_human_confirmation"},
            latency_ms=latency_ms,
        )
        return None
