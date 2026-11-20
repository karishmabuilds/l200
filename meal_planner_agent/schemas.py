"""Explicit Pydantic JSON schemas for strict tool I/O validation and structured outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictSchema(BaseModel):
    """Base schema enforcing strict field contracts to constrain LLM arguments."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ToolErrorResponse(StrictSchema):
    """Structured error payload providing guided recovery instructions to the LLM."""

    status: Literal["error"] = "error"
    error_code: str = Field(..., description="Machine-readable error classification code.")
    message: str = Field(..., description="Human-readable summary of what went wrong.")
    recovery_instructions: str = Field(
        ...,
        description="Actionable recovery steps telling the LLM how to fix and retry.",
    )
    suggested_next_tool: str | None = Field(
        default=None,
        description="Optional recommended tool to invoke to recover from this error.",
    )


class IngredientQuantity(StrictSchema):
    """A single measured ingredient for nutrition calculation or shopping."""

    name: str = Field(..., min_length=1, max_length=80, description="Canonical ingredient name.")
    grams: float = Field(..., gt=0, le=50000, description="Weight of the ingredient in grams.")


class MacroBreakdown(StrictSchema):
    """Calculated macronutrient profile for a recipe or meal."""

    calories_kcal: float = Field(..., ge=0)
    protein_g: float = Field(..., ge=0)
    carbs_g: float = Field(..., ge=0)
    fat_g: float = Field(..., ge=0)
    fiber_g: float = Field(..., ge=0)


class RecipeMacronutrientRequest(StrictSchema):
    """Validated input schema for calculate_recipe_macronutrients."""

    recipe_name: str = Field(..., min_length=2, max_length=120)
    servings: int = Field(..., ge=1, le=20)
    ingredients: list[IngredientQuantity] = Field(..., min_length=1, max_length=40)


class RecipeMacronutrientResponse(StrictSchema):
    """Structured output schema for calculate_recipe_macronutrients."""

    status: Literal["ok"] = "ok"
    recipe_name: str
    servings: int
    total_macros: MacroBreakdown
    per_serving_macros: MacroBreakdown
    unrecognized_ingredients: list[str] = Field(default_factory=list)


class AllergenAuditRequest(StrictSchema):
    """Validated input schema for audit_meal_plan_allergen_safety."""

    household_allergens: list[str] = Field(
        ...,
        description="List of allergens that must be strictly excluded (e.g., peanut, shellfish).",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary preferences such as vegan, vegetarian, gluten_free, dairy_free.",
    )
    planned_recipes: list[str] = Field(
        ...,
        min_length=1,
        max_length=28,
        description="List of recipe IDs or titles proposed in the meal schedule.",
    )


class AllergenViolation(StrictSchema):
    """Details of a single allergen or dietary conflict discovered during audit."""

    recipe_name: str
    offending_ingredient: str
    violated_constraint: str
    severity: Literal["critical_allergen", "dietary_mismatch"]
    safe_substitution_hint: str


class AllergenAuditResponse(StrictSchema):
    """Structured output schema for audit_meal_plan_allergen_safety."""

    status: Literal["ok"] = "ok"
    is_safe: bool
    violations: list[AllergenViolation] = Field(default_factory=list)
    audited_recipe_count: int = Field(..., ge=0)
    summary: str


class WeeklyMealScheduleRequest(StrictSchema):
    """Validated input schema for generate_weekly_meal_schedule."""

    days: int = Field(default=5, ge=1, le=7, description="Number of days to schedule (1-7).")
    daily_calorie_target: int = Field(
        default=2000, ge=1200, le=4500, description="Target daily energy intake per adult in kcal."
    )
    household_size: int = Field(default=2, ge=1, le=12, description="Number of diners.")
    dietary_restrictions: list[str] = Field(default_factory=list)
    household_allergens: list[str] = Field(default_factory=list)
    preferred_cuisines: list[str] = Field(default_factory=list)
    max_prep_minutes: int = Field(default=45, ge=10, le=180)


class ScheduledMeal(StrictSchema):
    """A single scheduled meal slot."""

    meal_type: Literal["breakfast", "lunch", "dinner"]
    recipe_id: str
    recipe_title: str
    cuisine: str
    prep_minutes: int
    estimated_cost_usd: float
    macros_per_serving: MacroBreakdown
    ingredients: list[IngredientQuantity]


class DayMealPlan(StrictSchema):
    """Complete meals for a single day."""

    day_number: int = Field(..., ge=1, le=7)
    day_label: str
    meals: list[ScheduledMeal]
    daily_calories_per_serving: float
    daily_protein_g_per_serving: float


class WeeklyMealScheduleResponse(StrictSchema):
    """Structured output schema for generate_weekly_meal_schedule."""

    status: Literal["ok"] = "ok"
    days_scheduled: int
    household_size: int
    daily_calorie_target: int
    schedule: list[DayMealPlan]
    estimated_total_cost_usd: float
    safety_notes: list[str] = Field(default_factory=list)


class PantryItem(StrictSchema):
    """An ingredient currently stocked in the user's household pantry."""

    ingredient_name: str
    available_grams: float = Field(..., ge=0)
    expires_in_days: int = Field(..., ge=0)
    storage_zone: Literal["pantry", "fridge", "freezer"]


class PantryInventoryRequest(StrictSchema):
    """Validated input schema for inspect_household_pantry_inventory."""

    household_id: str = Field(..., min_length=1, max_length=64)
    expiring_within_days: int | None = Field(
        default=None, ge=1, le=365, description="Filter to items expiring within N days."
    )


class PantryInventoryResponse(StrictSchema):
    """Structured output schema for inspect_household_pantry_inventory."""

    status: Literal["ok"] = "ok"
    household_id: str
    items: list[PantryItem]
    expiring_soon_highlights: list[str] = Field(default_factory=list)


class DeduplicatedGroceryRequest(StrictSchema):
    """Validated input schema for build_deduplicated_grocery_list."""

    household_id: str = Field(..., min_length=1, max_length=64)
    required_ingredients: list[IngredientQuantity] = Field(..., min_length=1, max_length=200)
    subtract_pantry_inventory: bool = Field(default=True)


class GroceryLineItem(StrictSchema):
    """Single deduplicated shopping list entry with aisle and estimated cost."""

    ingredient_name: str
    net_grams_to_buy: float = Field(..., gt=0)
    pantry_offset_grams: float = Field(..., ge=0)
    aisle: str
    estimated_cost_usd: float = Field(..., ge=0)


class DeduplicatedGroceryResponse(StrictSchema):
    """Structured output schema for build_deduplicated_grocery_list."""

    status: Literal["ok"] = "ok"
    household_id: str
    line_items: list[GroceryLineItem]
    pantry_covered_ingredients: list[str] = Field(default_factory=list)
    estimated_subtotal_usd: float = Field(..., ge=0)


class GroceryBasketCostRequest(StrictSchema):
    """Validated input schema for estimate_grocery_basket_cost."""

    items: list[IngredientQuantity] = Field(..., min_length=1, max_length=200)
    weekly_budget_usd: float = Field(..., gt=0, le=2000)
    delivery_zip_code: str = Field(..., pattern=r"^\d{5}$")


class GroceryBasketCostResponse(StrictSchema):
    """Structured output schema for estimate_grocery_basket_cost."""

    status: Literal["ok"] = "ok"
    subtotal_usd: float
    estimated_tax_usd: float
    delivery_fee_usd: float
    total_usd: float
    within_budget: bool
    budget_variance_usd: float
    cost_saving_tips: list[str] = Field(default_factory=list)


class GroceryOrderRequest(StrictSchema):
    """Validated input schema for high-stakes submit_grocery_delivery_order."""

    household_id: str = Field(..., min_length=1, max_length=64)
    delivery_window: str = Field(
        ..., min_length=4, max_length=64, description="Requested delivery window, e.g. Sun 2-4pm."
    )
    items: list[IngredientQuantity] = Field(..., min_length=1, max_length=200)
    estimated_total_usd: float = Field(..., gt=0, le=2000)
    weekly_budget_usd: float = Field(..., gt=0, le=2000)
    human_confirmed: bool = Field(
        default=False,
        description="Must only be true after explicit human confirmation via HITL workflow.",
    )

    @field_validator("delivery_window")
    @classmethod
    def validate_delivery_window(cls, value: str) -> str:
        if len(value.strip()) < 4:
            raise ValueError("delivery_window must specify a day and time slot.")
        return value.strip()


class GroceryOrderResponse(StrictSchema):
    """Structured output schema for submit_grocery_delivery_order."""

    status: Literal["confirmed", "awaiting_human_confirmation"]
    order_id: str | None = None
    household_id: str
    delivery_window: str
    charged_total_usd: float
    item_count: int
    message: str


class ConciergeFinalSummary(StrictSchema):
    """Structured final output contract for evaluation & UI presentation."""

    plan_summary: str
    allergen_audit_passed: bool
    estimated_grocery_total_usd: float
    requires_human_approval_for_checkout: bool
    next_recommended_action: str
