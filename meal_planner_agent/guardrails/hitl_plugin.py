"""Human-in-the-Loop (HITL) Approval Plugin halting high-stakes tool executions."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from meal_planner_agent.config import AgentSettings, get_settings
from meal_planner_agent.observability.structured_logging import (
    get_structured_logger,
    log_structured_event,
)
from meal_planner_agent.schemas import GroceryOrderResponse


class HumanInTheLoopApprovalPlugin(BasePlugin):
    """ADK Plugin enforcing explicit human confirmation gates before high-stakes actions.

    Intercepts high-stakes state-mutating or financial tools (such as placing a paid
    grocery delivery order) in ``before_tool_callback``. Unless explicit approval is
    present on ``tool_context.tool_confirmation`` (ADK resumable confirmation) or
    explicitly passed via ``human_confirmed=True``, execution is halted immediately
    and an interactive confirmation request is emitted.
    """

    HIGH_STAKES_TOOLS: frozenset[str] = frozenset({"submit_grocery_delivery_order"})

    def __init__(
        self,
        name: str = "human_in_the_loop_approval_plugin",
        settings: AgentSettings | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name=name)
        self._settings = settings or get_settings()
        self._logger = logger or get_structured_logger("meal_planner_agent.hitl")

    def requires_approval(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Determine whether the tool call crosses a high-stakes HITL threshold."""
        if tool_name not in self.HIGH_STAKES_TOOLS:
            return False
        total = float(tool_args.get("estimated_total_usd", 0.0) or 0.0)
        return total >= self._settings.hitl_order_approval_threshold_usd

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Halt high-stakes tool execution until explicit human confirmation is recorded."""
        if not self.requires_approval(tool.name, tool_args):
            return None

        confirmation = getattr(tool_context, "tool_confirmation", None)
        adk_approved = bool(confirmation and getattr(confirmation, "confirmed", False))
        explicit_flag = bool(tool_args.get("human_confirmed", False))

        if adk_approved or explicit_flag:
            log_structured_event(
                self._logger,
                "hitl_approval_verified",
                f"Human confirmation verified for high-stakes tool '{tool.name}'.",
                tool_name=tool.name,
                agent_name=tool_context.agent_name,
                approval_source="adk_tool_confirmation" if adk_approved else "explicit_flag",
            )
            return None

        estimated_total = float(tool_args.get("estimated_total_usd", 0.0) or 0.0)
        delivery_window = str(tool_args.get("delivery_window", "unspecified window"))
        items = tool_args.get("items") or []
        household_id = str(tool_args.get("household_id", "default_household"))

        if hasattr(tool_context, "request_confirmation"):
            tool_context.request_confirmation(
                hint=(
                    f"Confirm grocery order of {len(items)} items totaling "
                    f"${estimated_total:.2f} for delivery window '{delivery_window}'?"
                ),
                payload={
                    "tool_name": tool.name,
                    "household_id": household_id,
                    "estimated_total_usd": estimated_total,
                    "delivery_window": delivery_window,
                    "item_count": len(items),
                },
            )

        log_structured_event(
            self._logger,
            "hitl_approval_required",
            f"Halted high-stakes tool '{tool.name}' pending explicit human approval.",
            level=logging.WARNING,
            tool_name=tool.name,
            agent_name=tool_context.agent_name,
            estimated_total_usd=estimated_total,
            delivery_window=delivery_window,
            item_count=len(items),
        )

        return GroceryOrderResponse(
            status="awaiting_human_confirmation",
            order_id=None,
            household_id=household_id,
            delivery_window=delivery_window,
            charged_total_usd=0.0,
            item_count=len(items),
            message=(
                f"HUMAN CONFIRMATION REQUIRED: Execution of '{tool.name}' has been halted. "
                f"Please confirm whether you authorize charging ${estimated_total:.2f} for "
                f"{len(items)} grocery items to be delivered during '{delivery_window}'."
            ),
        ).model_dump()
