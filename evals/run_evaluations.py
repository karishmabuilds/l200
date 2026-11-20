"""Automated Golden Dataset Evaluation Harness for Regression Testing.

Supports two execution modes:
1. Default deterministic regression suite (CI-friendly, zero external API calls):
   - Validates `golden_dataset.evalset.json` against Google ADK's `EvalSet` schema.
   - Replays every golden trajectory's tool invocations, schema contracts, allergen
     safety invariants, HITL approval halts, prompt-injection guardrails, and PII
     scrubbing checks.
2. Live ADK AgentEvaluator mode (`--live`):
   - Runs `google.adk.evaluation.agent_evaluator.AgentEvaluator.evaluate(...)` against
     the live Gemini/Vertex AI backend using `evals/golden_dataset.evalset.json` and
     `evals/test_config.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repository root is importable when invoked as `python evals/run_evaluations.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google.adk.evaluation.eval_set import EvalSet

from meal_planner_agent.guardrails import GuardrailsAndPolicyPlugin, ResponseSelfEvaluator
from meal_planner_agent.observability import get_pii_redactor
from meal_planner_agent.tools import (
    audit_meal_plan_allergen_safety,
    build_deduplicated_grocery_list,
    calculate_recipe_macronutrients,
    estimate_grocery_basket_cost,
    generate_weekly_meal_schedule,
    inspect_household_pantry_inventory,
    submit_grocery_delivery_order,
)

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_EVALSET_PATH = EVALS_DIR / "golden_dataset.evalset.json"
TEST_CONFIG_PATH = EVALS_DIR / "test_config.json"

TOOL_REGISTRY: dict[str, Any] = {
    "inspect_household_pantry_inventory": inspect_household_pantry_inventory,
    "generate_weekly_meal_schedule": generate_weekly_meal_schedule,
    "audit_meal_plan_allergen_safety": audit_meal_plan_allergen_safety,
    "calculate_recipe_macronutrients": calculate_recipe_macronutrients,
    "build_deduplicated_grocery_list": build_deduplicated_grocery_list,
    "estimate_grocery_basket_cost": estimate_grocery_basket_cost,
    "submit_grocery_delivery_order": submit_grocery_delivery_order,
}


@dataclass
class CaseEvalResult:
    """Outcome summary for a single golden evaluation case."""

    eval_id: str
    passed: bool
    checks_run: int
    failures: list[str]


def run_offline_golden_evaluations(
    evalset_path: Path = GOLDEN_EVALSET_PATH,
) -> list[CaseEvalResult]:
    """Validate and replay every case in the golden ADK EvalSet deterministically."""
    raw_data = json.loads(evalset_path.read_text(encoding="utf-8"))
    eval_set = EvalSet.model_validate(raw_data)

    policy_plugin = GuardrailsAndPolicyPlugin()
    self_evaluator = ResponseSelfEvaluator()
    redactor = get_pii_redactor()
    results: list[CaseEvalResult] = []

    for case in eval_set.eval_cases:
        failures: list[str] = []
        checks_run = 0
        state = (
            case.session_input.state
            if case.session_input and case.session_input.state
            else {}
        )
        allergens = list(state.get("household_allergens", []))

        for turn in case.conversation or []:
            user_text = " ".join(
                p.text
                for p in (turn.user_content.parts if turn.user_content and turn.user_content.parts else [])
                if p.text
            )
            expected_reply = " ".join(
                p.text
                for p in (turn.final_response.parts if turn.final_response and turn.final_response.parts else [])
                if p.text
            )

            # Check 1: Prompt injection detection matches expectation
            checks_run += 1
            injection_hit = policy_plugin.detect_prompt_injection(user_text)
            if "injection" in case.eval_id and not injection_hit:
                failures.append("Expected prompt injection to be blocked, but guardrail allowed it.")
            if "injection" not in case.eval_id and injection_hit:
                failures.append(f"False-positive prompt injection block: {injection_hit}")

            # Check 2: Golden reference response passes self-eval & PII cleanliness
            checks_run += 1
            report = self_evaluator.evaluate(expected_reply, household_allergens=allergens)
            if not report.passed:
                failures.append(f"Golden response failed self-eval: {report.findings}")
            if redactor.redact_text(expected_reply) != expected_reply:
                failures.append("Golden response contains unredacted PII.")

            # Check 3: Replay every expected tool invocation and validate contract
            tool_uses = (
                turn.intermediate_data.tool_uses
                if turn.intermediate_data and turn.intermediate_data.tool_uses
                else []
            )
            for tool_call in tool_uses:
                checks_run += 1
                tool_fn = TOOL_REGISTRY.get(tool_call.name or "")
                if tool_fn is None:
                    failures.append(f"Unknown tool in golden trajectory: {tool_call.name}")
                    continue
                output = tool_fn(**(tool_call.args or {}))
                if not isinstance(output, dict) or "status" not in output:
                    failures.append(f"Tool {tool_call.name} did not return structured status dict.")
                    continue
                if tool_call.name == "submit_grocery_delivery_order":
                    if (
                        not (tool_call.args or {}).get("human_confirmed", False)
                        and output.get("status") != "awaiting_human_confirmation"
                    ):
                        failures.append(
                            "HITL violation: unconfirmed grocery order was not halted."
                        )
                elif tool_call.name == "audit_meal_plan_allergen_safety":
                    if output.get("status") != "ok" or not output.get("is_safe"):
                        failures.append(f"Allergen audit unexpectedly failed: {output}")
                else:
                    if output.get("status") != "ok":
                        failures.append(f"Tool {tool_call.name} returned error: {output}")

        results.append(
            CaseEvalResult(
                eval_id=case.eval_id,
                passed=len(failures) == 0,
                checks_run=checks_run,
                failures=failures,
            )
        )
    return results


async def run_live_adk_evaluations() -> None:
    """Run live end-to-end ADK AgentEvaluator against Gemini/Vertex AI."""
    from google.adk.evaluation.agent_evaluator import AgentEvaluator

    await AgentEvaluator.evaluate(
        agent_module="meal_planner_agent",
        eval_dataset_file_path_or_dir=str(GOLDEN_EVALSET_PATH),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Meal Planning Concierge evaluation suite.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live LLM evaluations via ADK AgentEvaluator (requires Gemini/Vertex creds).",
    )
    args = parser.parse_args()

    results = run_offline_golden_evaluations()
    summary = {
        "eval_set": str(GOLDEN_EVALSET_PATH.name),
        "total_cases": len(results),
        "passed_cases": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / max(1, len(results)), 4),
        "results": [asdict(r) for r in results],
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")

    if any(not r.passed for r in results):
        return 1

    if args.live:
        asyncio.run(run_live_adk_evaluations())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
