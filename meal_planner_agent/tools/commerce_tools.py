"""Grocery basket pricing and Human-in-the-Loop (HITL) delivery order tools."""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from meal_planner_agent.config import get_settings
from meal_planner_agent.schemas import (
    GroceryBasketCostRequest,
    GroceryBasketCostResponse,
    GroceryOrderRequest,
    GroceryOrderResponse,
    IngredientQuantity,
    ToolErrorResponse,
)
from meal_planner_agent.tools.catalog_data import INGREDIENT_DATABASE


def estimate_grocery_basket_cost(
    items: list[dict[str, Any]],
    weekly_budget_usd: float,
    delivery_zip_code: str = "94043",
) -> dict[str, Any]:
    """Estimate grocery basket subtotal, estimated tax, delivery fee, and budget fit.

    Calculates itemized ingredient pricing, applies regional delivery and tax estimates,
    compares the total against ``weekly_budget_usd``, and generates concrete cost-saving
    suggestions whenever the basket exceeds the user's budget.

    Args:
        items: Non-empty list of ``{"name": str, "grams": float}`` items to price.
        weekly_budget_usd: User's maximum target grocery spend in USD (0 < budget <= 2000).
        delivery_zip_code: 5-digit US ZIP code used to estimate delivery fee & tax
            (default ``"94043"``).

    Returns:
        A validated ``GroceryBasketCostResponse`` JSON dict with cost breakdown and
        budget status, or ``ToolErrorResponse`` with recovery instructions.
    """
    try:
        validated = GroceryBasketCostRequest(
            items=[IngredientQuantity(**ing) for ing in items],
            weekly_budget_usd=weekly_budget_usd,
            delivery_zip_code=delivery_zip_code,
        )
    except (ValidationError, TypeError) as exc:
        return ToolErrorResponse(
            error_code="INVALID_BASKET_COST_INPUT",
            message=f"Grocery basket cost request failed validation: {exc}",
            recovery_instructions=(
                "Ensure 'items' is a non-empty list of {'name': str, 'grams': float > 0}, "
                "'weekly_budget_usd' is between 0.01 and 2000, and 'delivery_zip_code' is 5 digits."
            ),
            suggested_next_tool="build_deduplicated_grocery_list",
        ).model_dump()

    subtotal = 0.0
    expensive_items: list[tuple[str, float]] = []
    for item in validated.items:
        entry = INGREDIENT_DATABASE.get(item.name.strip().lower())
        price_per_100g = entry["price_per_100g"] if entry else 1.25
        item_cost = (item.grams / 100.0) * price_per_100g
        subtotal += item_cost
        expensive_items.append((item.name.lower(), item_cost))

    subtotal = round(subtotal, 2)
    # Most fresh groceries are tax-exempt or lightly taxed (~2.5% blended estimate)
    estimated_tax = round(subtotal * 0.025, 2)
    delivery_fee = 0.0 if subtotal >= 75.0 else 6.99
    total = round(subtotal + estimated_tax + delivery_fee, 2)
    variance = round(validated.weekly_budget_usd - total, 2)
    within_budget = variance >= 0

    tips: list[str] = []
    if not within_budget:
        expensive_items.sort(key=lambda pair: pair[1], reverse=True)
        top_names = ", ".join(name for name, _ in expensive_items[:2])
        tips.append(
            f"Basket exceeds budget by ${abs(variance):.2f}. Consider swapping higher-cost "
            f"items ({top_names}) for chickpeas, red lentils, or brown rice."
        )
    if delivery_fee > 0:
        tips.append(
            f"Orders of $75+ qualify for $0 delivery fee (currently ${delivery_fee:.2f})."
        )

    return GroceryBasketCostResponse(
        subtotal_usd=subtotal,
        estimated_tax_usd=estimated_tax,
        delivery_fee_usd=delivery_fee,
        total_usd=total,
        within_budget=within_budget,
        budget_variance_usd=variance,
        cost_saving_tips=tips,
    ).model_dump()


def requires_grocery_order_confirmation(
    household_id: str,
    delivery_window: str,
    items: list[dict[str, Any]],
    estimated_total_usd: float,
    weekly_budget_usd: float,
    human_confirmed: bool = False,
) -> bool:
    """ADK FunctionTool confirmation predicate for high-stakes financial actions.

    Returns ``True`` whenever the order meets or exceeds the configured HITL approval
    threshold, instructing Google ADK to halt execution and require explicit human
    confirmation before executing ``submit_grocery_delivery_order``.
    """
    settings = get_settings()
    return estimated_total_usd >= settings.hitl_order_approval_threshold_usd


def submit_grocery_delivery_order(
    household_id: str,
    delivery_window: str,
    items: list[dict[str, Any]],
    estimated_total_usd: float,
    weekly_budget_usd: float,
    human_confirmed: bool = False,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Submit a grocery delivery order with mandatory Human-in-the-Loop confirmation.

    High-stakes financial action: placing a delivery order commits real household spend.
    This tool enforces a dual Human-in-the-Loop (HITL) safeguard:
    1. When invoked through ADK ``ToolContext``, it checks ``tool_context.tool_confirmation``
       and calls ``tool_context.request_confirmation(...)`` to pause the agent until the
       user approves the exact total and delivery window.
    2. When invoked directly or without an approved ``tool_confirmation``, it requires
       ``human_confirmed=True``; otherwise it refuses to place the order and returns
       ``status="awaiting_human_confirmation"``.

    Args:
        household_id: Unique household identifier for delivery routing.
        delivery_window: Requested delivery time slot (e.g., ``"Sunday 2:00 PM - 4:00 PM"``).
        items: Non-empty list of ``{"name": str, "grams": float}`` grocery items.
        estimated_total_usd: Expected order total in USD (> 0 and <= 2000).
        weekly_budget_usd: User's weekly grocery budget ceiling in USD.
        human_confirmed: Explicit boolean flag confirming the user approved checkout.
        tool_context: Optional Google ADK ``ToolContext`` injected automatically at runtime.

    Returns:
        A validated ``GroceryOrderResponse`` dict (either ``awaiting_human_confirmation``
        or ``confirmed`` with a unique ``order_id``), or ``ToolErrorResponse`` if
        validation or budget policy checks fail.
    """
    try:
        validated = GroceryOrderRequest(
            household_id=household_id,
            delivery_window=delivery_window,
            items=[IngredientQuantity(**ing) for ing in items],
            estimated_total_usd=estimated_total_usd,
            weekly_budget_usd=weekly_budget_usd,
            human_confirmed=human_confirmed,
        )
    except (ValidationError, TypeError) as exc:
        return ToolErrorResponse(
            error_code="INVALID_ORDER_PAYLOAD",
            message=f"Grocery delivery order failed validation: {exc}",
            recovery_instructions=(
                "Ensure delivery_window is specified, items is non-empty, and "
                "estimated_total_usd / weekly_budget_usd are positive numbers."
            ),
            suggested_next_tool="estimate_grocery_basket_cost",
        ).model_dump()

    settings = get_settings()
    if validated.estimated_total_usd > settings.max_weekly_grocery_budget_usd:
        return ToolErrorResponse(
            error_code="HARD_SPEND_CAP_EXCEEDED",
            message=(
                f"Order total ${validated.estimated_total_usd:.2f} exceeds the system safety "
                f"cap of ${settings.max_weekly_grocery_budget_usd:.2f}."
            ),
            recovery_instructions=(
                "Reduce basket quantities or remove luxury items using "
                "build_deduplicated_grocery_list and re-estimate with estimate_grocery_basket_cost."
            ),
            suggested_next_tool="estimate_grocery_basket_cost",
        ).model_dump()

    # ADK native Human-in-the-Loop confirmation flow
    adk_confirmed = False
    if tool_context is not None:
        confirmation = getattr(tool_context, "tool_confirmation", None)
        if confirmation is None:
            tool_context.request_confirmation(
                hint=(
                    f"Approve grocery delivery order of {len(validated.items)} items "
                    f"totaling ${validated.estimated_total_usd:.2f} "
                    f"(budget: ${validated.weekly_budget_usd:.2f}) "
                    f"for delivery window '{validated.delivery_window}'?"
                ),
                payload={
                    "household_id": validated.household_id,
                    "estimated_total_usd": validated.estimated_total_usd,
                    "weekly_budget_usd": validated.weekly_budget_usd,
                    "delivery_window": validated.delivery_window,
                    "item_count": len(validated.items),
                },
            )
            return GroceryOrderResponse(
                status="awaiting_human_confirmation",
                order_id=None,
                household_id=validated.household_id,
                delivery_window=validated.delivery_window,
                charged_total_usd=0.0,
                item_count=len(validated.items),
                message=(
                    "Order paused for Human-in-the-Loop approval. Please confirm you want "
                    f"to charge ${validated.estimated_total_usd:.2f} for {len(validated.items)} "
                    f"items scheduled for {validated.delivery_window}."
                ),
            ).model_dump()

        if not getattr(confirmation, "confirmed", False):
            return ToolErrorResponse(
                error_code="HUMAN_REJECTED_ORDER",
                message="The user declined to approve this grocery delivery order.",
                recovery_instructions=(
                    "Acknowledge the cancellation politely and ask whether the user would "
                    "like to modify the meal plan, lower the basket cost, or pick another window."
                ),
                suggested_next_tool="estimate_grocery_basket_cost",
            ).model_dump()
        adk_confirmed = True

    if not (adk_confirmed or validated.human_confirmed):
        return GroceryOrderResponse(
            status="awaiting_human_confirmation",
            order_id=None,
            household_id=validated.household_id,
            delivery_window=validated.delivery_window,
            charged_total_usd=0.0,
            item_count=len(validated.items),
            message=(
                "Explicit human confirmation is required before charging "
                f"${validated.estimated_total_usd:.2f}. Present the basket total and "
                "delivery window to the user and ask for confirmation."
            ),
        ).model_dump()

    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    return GroceryOrderResponse(
        status="confirmed",
        order_id=order_id,
        household_id=validated.household_id,
        delivery_window=validated.delivery_window,
        charged_total_usd=validated.estimated_total_usd,
        item_count=len(validated.items),
        message=(
            f"Grocery delivery order {order_id} confirmed for {validated.delivery_window} "
            f"(${validated.estimated_total_usd:.2f} for {len(validated.items)} items)."
        ),
    ).model_dump()
