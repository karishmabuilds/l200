"""Pantry inventory lookup and deduplicated grocery list tools."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from meal_planner_agent.schemas import (
    DeduplicatedGroceryRequest,
    DeduplicatedGroceryResponse,
    GroceryLineItem,
    IngredientQuantity,
    PantryInventoryRequest,
    PantryInventoryResponse,
    PantryItem,
    ToolErrorResponse,
)
from meal_planner_agent.tools.catalog_data import (
    DEFAULT_HOUSEHOLD_PANTRIES,
    INGREDIENT_DATABASE,
)


def _get_pantry_for_household(household_id: str) -> list[PantryItem]:
    raw_items = DEFAULT_HOUSEHOLD_PANTRIES.get(
        household_id, DEFAULT_HOUSEHOLD_PANTRIES["default_household"]
    )
    return [PantryItem(**item) for item in raw_items]


def inspect_household_pantry_inventory(
    household_id: str = "default_household",
    expiring_within_days: int | None = None,
) -> dict[str, Any]:
    """Inspect current stocked ingredients in a household pantry, fridge, and freezer.

    Retrieves available ingredient quantities in grams and highlights perishable items
    expiring within 5 days so meal planning minimizes food waste.

    Args:
        household_id: Unique identifier of the household pantry (default
            ``"default_household"``).
        expiring_within_days: Optional integer filter (1-365) returning only items
            expiring within the specified number of days.

    Returns:
        A validated ``PantryInventoryResponse`` JSON dictionary with ``items`` and
        ``expiring_soon_highlights``, or a ``ToolErrorResponse`` with recovery
        instructions on invalid inputs.
    """
    try:
        validated = PantryInventoryRequest(
            household_id=household_id,
            expiring_within_days=expiring_within_days,
        )
    except ValidationError as exc:
        return ToolErrorResponse(
            error_code="INVALID_PANTRY_QUERY",
            message=f"Pantry inventory query failed validation: {exc}",
            recovery_instructions=(
                "Provide a non-empty 'household_id' (e.g., 'default_household') and an "
                "optional positive integer 'expiring_within_days' between 1 and 365."
            ),
            suggested_next_tool="inspect_household_pantry_inventory",
        ).model_dump()

    items = _get_pantry_for_household(validated.household_id)
    if validated.expiring_within_days is not None:
        items = [i for i in items if i.expires_in_days <= validated.expiring_within_days]

    expiring_soon = [
        f"{item.ingredient_name} ({int(item.available_grams)}g in {item.storage_zone}, "
        f"expires in {item.expires_in_days}d)"
        for item in sorted(items, key=lambda x: x.expires_in_days)
        if item.expires_in_days <= 5
    ]

    return PantryInventoryResponse(
        household_id=validated.household_id,
        items=items,
        expiring_soon_highlights=expiring_soon,
    ).model_dump()


def build_deduplicated_grocery_list(
    required_ingredients: list[dict[str, Any]],
    household_id: str = "default_household",
    subtract_pantry_inventory: bool = True,
) -> dict[str, Any]:
    """Consolidate duplicate recipe ingredients and subtract stocked pantry items.

    Merges duplicate ingredient entries across a multi-day meal schedule, deducts
    already-stocked pantry grams when ``subtract_pantry_inventory`` is ``True``,
    assigns supermarket aisles, and computes estimated line-item costs.

    Args:
        required_ingredients: Non-empty list of ``{"name": str, "grams": float}``
            entries collected from planned meals.
        household_id: Household identifier whose pantry quantities should be offset.
        subtract_pantry_inventory: When ``True`` (default), deducts existing pantry
            quantities so users do not over-purchase staples.

    Returns:
        A validated ``DeduplicatedGroceryResponse`` JSON dictionary grouped by aisle,
        or a ``ToolErrorResponse`` with guidance if inputs fail schema validation.
    """
    try:
        validated = DeduplicatedGroceryRequest(
            household_id=household_id,
            required_ingredients=[IngredientQuantity(**ing) for ing in required_ingredients],
            subtract_pantry_inventory=subtract_pantry_inventory,
        )
    except (ValidationError, TypeError) as exc:
        return ToolErrorResponse(
            error_code="INVALID_GROCERY_LIST_INPUT",
            message=f"Deduplicated grocery list input failed validation: {exc}",
            recovery_instructions=(
                "Pass 'required_ingredients' as a non-empty list of {'name': str, 'grams': float > 0} "
                "objects extracted from generate_weekly_meal_schedule."
            ),
            suggested_next_tool="generate_weekly_meal_schedule",
        ).model_dump()

    aggregated: dict[str, float] = defaultdict(float)
    for item in validated.required_ingredients:
        aggregated[item.name.strip().lower()] += item.grams

    pantry_map: dict[str, float] = {}
    if validated.subtract_pantry_inventory:
        for p_item in _get_pantry_for_household(validated.household_id):
            pantry_map[p_item.ingredient_name.lower()] = p_item.available_grams

    line_items: list[GroceryLineItem] = []
    pantry_covered: list[str] = []
    subtotal = 0.0

    for ing_name, total_grams in sorted(aggregated.items()):
        available = pantry_map.get(ing_name, 0.0)
        offset = min(total_grams, available)
        net_grams = round(total_grams - offset, 1)
        if net_grams <= 0:
            pantry_covered.append(ing_name)
            continue

        entry = INGREDIENT_DATABASE.get(ing_name)
        aisle = entry["aisle"] if entry else "Specialty & Misc"
        unit_price = entry["price_per_100g"] if entry else 1.25
        item_cost = round((net_grams / 100.0) * unit_price, 2)
        subtotal += item_cost
        line_items.append(
            GroceryLineItem(
                ingredient_name=ing_name,
                net_grams_to_buy=net_grams,
                pantry_offset_grams=round(offset, 1),
                aisle=aisle,
                estimated_cost_usd=item_cost,
            )
        )

    line_items.sort(key=lambda item: (item.aisle, item.ingredient_name))

    return DeduplicatedGroceryResponse(
        household_id=validated.household_id,
        line_items=line_items,
        pantry_covered_ingredients=sorted(pantry_covered),
        estimated_subtotal_usd=round(subtotal, 2),
    ).model_dump()
