"""ADK Guardrails, Policy Enforcement, and Self-Evaluation Plugin."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from meal_planner_agent.config import AgentSettings, get_settings
from meal_planner_agent.observability.pii_redactor import PIIRedactor, get_pii_redactor
from meal_planner_agent.observability.structured_logging import (
    get_structured_logger,
    log_structured_event,
)
from meal_planner_agent.schemas import ToolErrorResponse

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(?:the\s+)?(?:allergen|safety|confirmation|hitl|budget)", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?(?:system\s+prompt|api\s+key|secret)", re.IGNORECASE),
    re.compile(r"skip\s+(?:human\s+)?confirmation\s+and\s+(?:order|charge|buy)", re.IGNORECASE),
    re.compile(r"pretend\s+allergies\s+do\s+not\s+exist", re.IGNORECASE),
]

_MEDICAL_PRESCRIPTION_PATTERN = re.compile(
    r"\b(?:stop\s+taking\s+your\s+medication|cures?\s+diabetes|prescribe\s+insulin)\b",
    re.IGNORECASE,
)


@dataclass
class SelfEvaluationReport:
    """Result of the automated self-evaluation check executed on model outputs."""

    passed: bool
    safety_score: float
    findings: list[str]


class ResponseSelfEvaluator:
    """Performs lightweight deterministic self-evaluation on every candidate response.

    Verifies three safety invariants before text reaches the user:
    1. No household allergens leaked into recommended meal text.
    2. No raw PII leaked into generated text.
    3. No unsafe medical prescription claims.
    """

    def __init__(self, redactor: PIIRedactor | None = None) -> None:
        self._redactor = redactor or get_pii_redactor()

    def evaluate(
        self,
        response_text: str,
        *,
        household_allergens: list[str] | None = None,
    ) -> SelfEvaluationReport:
        findings: list[str] = []
        lower = response_text.lower()

        # 1. Allergen cross-check (ignore explicit safety disclaimers like "peanut-free" / "no peanut")
        for allergen in household_allergens or []:
            term = allergen.strip().lower()
            if not term:
                continue
            safe_phrases = (
                f"{term}-free",
                f"{term} free",
                f"no {term}",
                f"without {term}",
                f"avoid {term}",
                f"excluding {term}",
                f"exclude {term}",
                "excluded allergens",
                f"allergic to {term}",
            )
            cleaned = lower
            for safe in safe_phrases:
                cleaned = cleaned.replace(safe, "")
            if re.search(rf"\b{re.escape(term)}s?\b", cleaned):
                findings.append(f"Potential allergen leak detected in output: '{term}'.")

        # 2. PII leak check
        if self._redactor.redact_text(response_text) != response_text:
            findings.append("Unredacted PII pattern detected in model output.")

        # 3. Unsafe medical claim check
        if _MEDICAL_PRESCRIPTION_PATTERN.search(response_text):
            findings.append("Unsafe clinical/medical prescription language detected.")

        score = max(0.0, round(1.0 - 0.5 * len(findings), 2))
        return SelfEvaluationReport(
            passed=len(findings) == 0,
            safety_score=score,
            findings=findings,
        )


class GuardrailsAndPolicyPlugin(BasePlugin):
    """ADK Plugin implementing input guardrails, tool policy checks, and output self-eval."""

    def __init__(
        self,
        name: str = "guardrails_and_policy_plugin",
        settings: AgentSettings | None = None,
        self_evaluator: ResponseSelfEvaluator | None = None,
        redactor: PIIRedactor | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name=name)
        self._settings = settings or get_settings()
        self._redactor = redactor or get_pii_redactor()
        self._evaluator = self_evaluator or ResponseSelfEvaluator(self._redactor)
        self._logger = logger or get_structured_logger("meal_planner_agent.guardrails")

    @staticmethod
    def _extract_request_text(llm_request: LlmRequest) -> str:
        chunks: list[str] = []
        for content in getattr(llm_request, "contents", None) or []:
            if getattr(content, "role", "") != "user":
                continue
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None):
                    chunks.append(part.text)
        return "\n".join(chunks)

    @staticmethod
    def _extract_response_text(llm_response: LlmResponse) -> str:
        content = getattr(llm_response, "content", None)
        if not content or not getattr(content, "parts", None):
            return ""
        return "\n".join(part.text for part in content.parts if getattr(part, "text", None))

    def detect_prompt_injection(self, text: str) -> str | None:
        """Return matching policy rule description if text attempts prompt injection."""
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                return f"Blocked prompt-injection pattern: {pattern.pattern}"
        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """Block prompt injection and safety-bypass attacks before reaching the LLM."""
        user_text = self._extract_request_text(llm_request)
        violation = self.detect_prompt_injection(user_text)
        if violation:
            log_structured_event(
                self._logger,
                "guardrail_input_blocked",
                "Blocked unsafe user prompt before LLM invocation.",
                level=logging.WARNING,
                agent_name=callback_context.agent_name,
                policy_violation=violation,
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "I cannot bypass allergen safety checks, reveal credentials, "
                                "or skip human confirmation for grocery orders. Please share "
                                "your legitimate meal-planning preferences and I will gladly help."
                            )
                        )
                    ],
                )
            )
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Enforce spend-cap and allergen policy constraints prior to tool execution."""
        if tool.name == "submit_grocery_delivery_order":
            est_total = float(tool_args.get("estimated_total_usd", 0.0) or 0.0)
            if est_total > self._settings.max_weekly_grocery_budget_usd:
                log_structured_event(
                    self._logger,
                    "guardrail_tool_blocked",
                    "Blocked grocery order exceeding maximum spend safety policy.",
                    level=logging.WARNING,
                    tool_name=tool.name,
                    estimated_total_usd=est_total,
                    max_allowed_usd=self._settings.max_weekly_grocery_budget_usd,
                )
                return ToolErrorResponse(
                    error_code="POLICY_SPEND_CAP_EXCEEDED",
                    message=(
                        f"Policy guardrail blocked order of ${est_total:.2f} because it exceeds "
                        f"the hard cap of ${self._settings.max_weekly_grocery_budget_usd:.2f}."
                    ),
                    recovery_instructions=(
                        "Trim expensive items from the grocery list and re-estimate the basket "
                        "cost using estimate_grocery_basket_cost."
                    ),
                    suggested_next_tool="estimate_grocery_basket_cost",
                ).model_dump()
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Run automated self-evaluation on candidate responses to prevent allergen/PII leaks."""
        text = self._extract_response_text(llm_response)
        if not text:
            return None

        state = getattr(callback_context, "state", None)
        allergens: list[str] = []
        if state is not None and hasattr(state, "get"):
            raw = state.get("household_allergens", [])
            if isinstance(raw, list):
                allergens = [str(a) for a in raw]

        report = self._evaluator.evaluate(text, household_allergens=allergens)
        log_structured_event(
            self._logger,
            "guardrail_self_eval",
            "Completed post-generation self-evaluation check.",
            agent_name=callback_context.agent_name,
            self_eval_passed=report.passed,
            safety_score=report.safety_score,
            findings=report.findings,
        )

        if not report.passed:
            # Scrub PII immediately; if allergen or medical safety failed, append explicit safety hold
            sanitized = self._redactor.redact_text(text)
            if any("allergen" in f.lower() or "medical" in f.lower() for f in report.findings):
                sanitized = (
                    "[SAFETY GUARDRAIL HOLD] The draft response triggered an automated safety "
                    f"review ({'; '.join(report.findings)}). Please verify your household "
                    "allergens and rerun `audit_meal_plan_allergen_safety` before proceeding."
                )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=sanitized)],
                )
            )
        return None
