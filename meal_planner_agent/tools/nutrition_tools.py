"""Nutrition calculation, allergen safety auditing, and weekly meal scheduling tools."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from meal_planner_agent.schemas import (
    AllergenAuditRequest,
    AllergenAuditResponse,
    AllergenViolation,
    DayMealPlan,
    IngredientQuantity,
    MacroBreakdown,
    RecipeMacronutrientRequest,
    RecipeMacronutrientResponse,
    ScheduledMeal,
    ToolErrorResponse,
    WeeklyMealScheduleRequest,
    WeeklyMealScheduleResponse,
)
from meal_planner_agent.tools.catalog_data import (
    ALLERGEN_SUBSTITUTIONS,
    INGREDIENT_DATABASE,
    RECIPE_CATALOG,
)

DAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

ALLERGEN_ALIASES: dict[str, str] = {
    "peanuts": "peanut",
    "peanut": "peanut",
    "nuts": "tree_nut",
    "tree nuts": "tree_nut",
    "tree_nut": "tree_nut",
    "almonds": "tree_nut",
    "milk": "dairy",
    "dairy": "dairy",
    "lactose": "dairy",
    "cheese": "dairy",
    "eggs": "egg",
    "egg": "egg",
    "soy": "soy",
    "tofu": "soy",
    "shellfish": "shellfish",
    "shrimp": "shellfish",
    "fish": "fish",
    "salmon": "fish",
    "wheat": "wheat",
    "gluten": "gluten",
}


def _normalize_allergen(raw: str) -> str:
    cleaned = raw.strip().lower().replace("-", "_")
    return ALLERGEN_ALIASES.get(cleaned, cleaned)


def _compute_macros_and_cost(ingredients: list[dict[str, Any]]) -> tuple[MacroBreakdown, float, list[str]]:
    kcal = protein = carbs = fat = fiber = cost = 0.0
    unrecognized: list[str] = []
    for item in ingredients:
        name = str(item["name"]).strip().lower()
        grams = float(item["grams"])
        entry = INGREDIENT_DATABASE.get(name)
        if not entry:
            unrecognized.append(name)
            continue
        factor = grams / 100.0
        kcal += entry["kcal"] * factor
        protein += entry["protein"] * factor
        carbs += entry["carbs"] * factor
        fat += entry["fat"] * factor
        fiber += entry["fiber"] * factor
        cost += entry["price_per_100g"] * factor
    return (
        MacroBreakdown(
            calories_kcal=round(kcal, 1),
            protein_g=round(protein, 1),
            carbs_g=round(carbs, 1),
            fat_g=round(fat, 1),
            fiber_g=round(fiber, 1),
        ),
        round(cost, 2),
        unrecognized,
    )


def calculate_recipe_macronutrients(
    recipe_name: str,
    servings: int,
    ingredients: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate exact total and per-serving macronutrients for a recipe.

    Validates all ingredient quantities against a strict Pydantic schema, looks up
    USDA-style per-100g nutrition profiles, and returns calories, protein, carbs,
    fat, and fiber for both the whole recipe and a single serving.

    Args:
        recipe_name: Human-readable title of the recipe (2-120 characters).
        servings: Number of portions the recipe yields (integer between 1 and 20).
        ingredients: Non-empty list of objects with ``name`` (str) and ``grams``
            (float > 0), e.g., ``[{"name": "quinoa", "grams": 150}]``.

    Returns:
        A validated JSON dictionary conforming to ``RecipeMacronutrientResponse``
        when successful, or ``ToolErrorResponse`` with actionable recovery
        instructions if validation fails or all ingredients are unrecognized.
    """
    try:
        validated = RecipeMacronutrientRequest(
            recipe_name=recipe_name,
            servings=servings,
            ingredients=[IngredientQuantity(**ing) for ing in ingredients],
        )
    except (ValidationError, TypeError) as exc:
        return ToolErrorResponse(
            error_code="INVALID_RECIPE_INPUT",
            message=f"Recipe macronutrient input failed schema validation: {exc}",
            recovery_instructions=(
                "Ensure 'recipe_name' is at least 2 characters, 'servings' is an integer "
                "between 1 and 20, and each ingredient has {'name': str, 'grams': float > 0}."
            ),
            suggested_next_tool="calculate_recipe_macronutrients",
        ).model_dump()

    total_macros, _, unrecognized = _compute_macros_and_cost(
        [ing.model_dump() for ing in validated.ingredients]
    )

    if len(unrecognized) == len(validated.ingredients):
        supported = ", ".join(sorted(INGREDIENT_DATABASE.keys())[:10])
        return ToolErrorResponse(
            error_code="UNRECOGNIZED_INGREDIENTS",
            message=f"None of the supplied ingredients were found: {unrecognized}.",
            recovery_instructions=(
                f"Map ingredients to canonical nutrition database names (examples: {supported}) "
                "and retry calculate_recipe_macronutrients."
            ),
            suggested_next_tool="calculate_recipe_macronutrients",
        ).model_dump()

    per_serving = MacroBreakdown(
        calories_kcal=round(total_macros.calories_kcal / validated.servings, 1),
        protein_g=round(total_macros.protein_g / validated.servings, 1),
        carbs_g=round(total_macros.carbs_g / validated.servings, 1),
        fat_g=round(total_macros.fat_g / validated.servings, 1),
        fiber_g=round(total_macros.fiber_g / validated.servings, 1),
    )

    return RecipeMacronutrientResponse(
        recipe_name=validated.recipe_name,
        servings=validated.servings,
        total_macros=total_macros,
        per_serving_macros=per_serving,
        unrecognized_ingredients=unrecognized,
    ).model_dump()


def audit_meal_plan_allergen_safety(
    household_allergens: list[str],
    planned_recipes: list[str],
    dietary_restrictions: list[str] | None = None,
) -> dict[str, Any]:
    """Audit proposed recipes against household allergens and dietary rules.

    Performs a strict ingredient-level safety check for every planned recipe against
    life-threatening allergens (peanut, tree_nut, dairy, egg, soy, shellfish, fish,
    wheat, gluten) and dietary restrictions (vegan, vegetarian, gluten_free,
    dairy_free). Returns safe substitution hints for any detected conflict.

    Args:
        household_allergens: List of allergens to strictly exclude (may be empty).
        planned_recipes: Non-empty list of recipe IDs or titles to inspect.
        dietary_restrictions: Optional list of dietary labels such as ``"vegan"``,
            ``"vegetarian"``, ``"gluten_free"``, or ``"dairy_free"``.

    Returns:
        A validated ``AllergenAuditResponse`` JSON dict containing ``is_safe``,
        ``violations``, and substitution guidance, or ``ToolErrorResponse`` on
        invalid inputs.
    """
    try:
        validated = AllergenAuditRequest(
            household_allergens=household_allergens,
            dietary_restrictions=dietary_restrictions or [],
            planned_recipes=planned_recipes,
        )
    except ValidationError as exc:
        return ToolErrorResponse(
            error_code="INVALID_ALLERGEN_AUDIT_INPUT",
            message=f"Allergen audit request failed validation: {exc}",
            recovery_instructions=(
                "Pass a non-empty list of recipe titles or IDs in 'planned_recipes' "
                "and lists of strings for 'household_allergens' and 'dietary_restrictions'."
            ),
            suggested_next_tool="audit_meal_plan_allergen_safety",
        ).model_dump()

    normalized_allergens = {_normalize_allergen(a) for a in validated.household_allergens if a.strip()}
    normalized_diets = {d.strip().lower().replace("-", "_").replace(" ", "_") for d in validated.dietary_restrictions if d.strip()}

    recipe_lookup = {}
    for recipe in RECIPE_CATALOG:
        recipe_lookup[recipe["recipe_id"].lower()] = recipe
        recipe_lookup[recipe["title"].lower()] = recipe

    violations: list[AllergenViolation] = []

    for candidate in validated.planned_recipes:
        key = candidate.strip().lower()
        recipe = recipe_lookup.get(key)

        if recipe is None:
            # Heuristic fallback for free-text custom recipes mentioned by the user
            for allergen in normalized_allergens:
                if allergen in key or any(
                    alias in key and mapped == allergen for alias, mapped in ALLERGEN_ALIASES.items()
                ):
                    violations.append(
                        AllergenViolation(
                            recipe_name=candidate,
                            offending_ingredient=allergen,
                            violated_constraint=f"allergen:{allergen}",
                            severity="critical_allergen",
                            safe_substitution_hint=ALLERGEN_SUBSTITUTIONS.get(
                                allergen, f"Remove all {allergen}-containing ingredients."
                            ),
                        )
                    )
            continue

        # Ingredient-level allergen verification
        for ing in recipe["ingredients"]:
            ing_name = ing["name"].lower()
            entry = INGREDIENT_DATABASE.get(ing_name, {})
            ing_allergens = set(entry.get("allergens", []))
            matched_allergens = normalized_allergens.intersection(ing_allergens)
            for allergen in sorted(matched_allergens):
                violations.append(
                    AllergenViolation(
                        recipe_name=recipe["title"],
                        offending_ingredient=ing_name,
                        violated_constraint=f"allergen:{allergen}",
                        severity="critical_allergen",
                        safe_substitution_hint=ALLERGEN_SUBSTITUTIONS.get(
                            allergen, f"Substitute {ing_name} with an allergen-free alternative."
                        ),
                    )
                )

            # Dietary restriction verification
            ing_tags = set(entry.get("tags", []))
            for diet in sorted(normalized_diets):
                if diet in {"vegan", "vegetarian", "dairy_free", "gluten_free"} and diet not in ing_tags:
                    violations.append(
                        AllergenViolation(
                            recipe_name=recipe["title"],
                            offending_ingredient=ing_name,
                            violated_constraint=f"diet:{diet}",
                            severity="dietary_mismatch",
                            safe_substitution_hint=(
                                f"Replace '{ing_name}' with a {diet}-compliant protein or base."
                            ),
                        )
                    )

    is_safe = len(violations) == 0
    summary = (
        f"All {len(validated.planned_recipes)} recipes passed strict allergen and dietary checks."
        if is_safe
        else f"BLOCKED: Found {len(violations)} allergen/dietary violation(s) across "
        f"{len(validated.planned_recipes)} recipes. Apply substitutions before presenting."
    )

    return AllergenAuditResponse(
        is_safe=is_safe,
        violations=violations,
        audited_recipe_count=len(validated.planned_recipes),
        summary=summary,
    ).model_dump()


def generate_weekly_meal_schedule(
    days: int = 5,
    daily_calorie_target: int = 2000,
    household_size: int = 2,
    dietary_restrictions: list[str] | None = None,
    household_allergens: list[str] | None = None,
    preferred_cuisines: list[str] | None = None,
    max_prep_minutes: int = 45,
) -> dict[str, Any]:
    """Generate a personalized, allergen-filtered multi-day meal schedule.

    Selects breakfast, lunch, and dinner recipes from the culinary catalog that
    strictly exclude all household allergens, honor dietary restrictions and prep
    time caps, and rotate preferred cuisines across the requested number of days.

    Args:
        days: Number of days to schedule (1 through 7, default 5).
        daily_calorie_target: Target daily energy intake per diner in kcal (1200-4500).
        household_size: Number of household diners (1-12, default 2).
        dietary_restrictions: Optional dietary requirements (e.g., ``["vegetarian"]``).
        household_allergens: Optional allergens to strictly filter out (e.g., ``["peanut"]``).
        preferred_cuisines: Optional preferred cuisines to prioritize (e.g., ``["mediterranean"]``).
        max_prep_minutes: Maximum allowed preparation time per meal in minutes (10-180).

    Returns:
        A validated ``WeeklyMealScheduleResponse`` JSON dictionary with daily meals,
        macronutrients, ingredients, and total estimated cost, or a
        ``ToolErrorResponse`` explaining how to relax impossible constraints.
    """
    try:
        validated = WeeklyMealScheduleRequest(
            days=days,
            daily_calorie_target=daily_calorie_target,
            household_size=household_size,
            dietary_restrictions=dietary_restrictions or [],
            household_allergens=household_allergens or [],
            preferred_cuisines=preferred_cuisines or [],
            max_prep_minutes=max_prep_minutes,
        )
    except ValidationError as exc:
        return ToolErrorResponse(
            error_code="INVALID_SCHEDULE_PARAMETERS",
            message=f"Meal schedule parameters failed validation: {exc}",
            recovery_instructions=(
                "Verify 'days' is 1-7, 'daily_calorie_target' is 1200-4500, "
                "'household_size' is 1-12, and 'max_prep_minutes' is 10-180."
            ),
            suggested_next_tool="generate_weekly_meal_schedule",
        ).model_dump()

    normalized_allergens = {_normalize_allergen(a) for a in validated.household_allergens if a.strip()}
    normalized_diets = {
        d.strip().lower().replace("-", "_").replace(" ", "_")
        for d in validated.dietary_restrictions
        if d.strip()
    }
    preferred = {c.strip().lower() for c in validated.preferred_cuisines if c.strip()}

    def _recipe_is_eligible(recipe: dict[str, Any]) -> bool:
        if recipe["prep_minutes"] > validated.max_prep_minutes:
            return False
        recipe_tags = set(recipe["dietary_tags"])
        for diet in normalized_diets:
            if diet in {"vegan", "vegetarian", "dairy_free", "gluten_free", "pescatarian"}:
                if diet not in recipe_tags:
                    return False
        for ing in recipe["ingredients"]:
            entry = INGREDIENT_DATABASE.get(ing["name"].lower(), {})
            if normalized_allergens.intersection(entry.get("allergens", [])):
                return False
        return True

    eligible_by_slot: dict[str, list[dict[str, Any]]] = {
        "breakfast": [],
        "lunch": [],
        "dinner": [],
    }
    for recipe in RECIPE_CATALOG:
        if _recipe_is_eligible(recipe):
            eligible_by_slot[recipe["meal_type"]].append(recipe)

    missing_slots = [slot for slot, recipes in eligible_by_slot.items() if not recipes]
    if missing_slots:
        return ToolErrorResponse(
            error_code="NO_ELIGIBLE_RECIPES_FOR_CONSTRAINTS",
            message=(
                f"No catalog recipes satisfied all constraints for meal slots: {missing_slots}."
            ),
            recovery_instructions=(
                "Try increasing 'max_prep_minutes' (e.g., to 35+ minutes) or confirm "
                "whether all dietary restrictions apply simultaneously. Never relax "
                "household_allergens; ask the user for clarification instead."
            ),
            suggested_next_tool="generate_weekly_meal_schedule",
        ).model_dump()

    # Sort preferred cuisines first within each meal slot for deterministic variety
    for slot in eligible_by_slot:
        eligible_by_slot[slot].sort(
            key=lambda r: (0 if r["cuisine"].lower() in preferred else 1, r["recipe_id"])
        )

    schedule: list[DayMealPlan] = []
    total_cost_usd = 0.0

    for day_idx in range(validated.days):
        day_meals: list[ScheduledMeal] = []
        day_kcal = 0.0
        day_protein = 0.0
        for slot in ("breakfast", "lunch", "dinner"):
            pool = eligible_by_slot[slot]
            chosen = pool[day_idx % len(pool)]
            macros, single_cost, _ = _compute_macros_and_cost(chosen["ingredients"])
            meal_cost = round(single_cost * validated.household_size, 2)
            total_cost_usd += meal_cost
            day_kcal += macros.calories_kcal
            day_protein += macros.protein_g
            scaled_ingredients = [
                IngredientQuantity(
                    name=ing["name"],
                    grams=round(float(ing["grams"]) * validated.household_size, 1),
                )
                for ing in chosen["ingredients"]
            ]
            day_meals.append(
                ScheduledMeal(
                    meal_type=slot,  # type: ignore[arg-type]
                    recipe_id=chosen["recipe_id"],
                    recipe_title=chosen["title"],
                    cuisine=chosen["cuisine"],
                    prep_minutes=chosen["prep_minutes"],
                    estimated_cost_usd=meal_cost,
                    macros_per_serving=macros,
                    ingredients=scaled_ingredients,
                )
            )
        schedule.append(
            DayMealPlan(
                day_number=day_idx + 1,
                day_label=DAY_LABELS[day_idx % len(DAY_LABELS)],
                meals=day_meals,
                daily_calories_per_serving=round(day_kcal, 1),
                daily_protein_g_per_serving=round(day_protein, 1),
            )
        )

    safety_notes = [
        f"Excluded allergens: {sorted(normalized_allergens) or ['none specified']}.",
        "Run audit_meal_plan_allergen_safety to independently verify this schedule before checkout.",
    ]

    return WeeklyMealScheduleResponse(
        days_scheduled=validated.days,
        household_size=validated.household_size,
        daily_calorie_target=validated.daily_calorie_target,
        schedule=schedule,
        estimated_total_cost_usd=round(total_cost_usd, 2),
        safety_notes=safety_notes,
    ).model_dump()
