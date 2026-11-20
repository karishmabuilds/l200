"""Tests for Tool & Interface Design: docstrings, naming, Pydantic schemas, and guided errors."""

from __future__ import annotations

import inspect
import unittest

from meal_planner_agent.tools import (
    audit_meal_plan_allergen_safety,
    build_deduplicated_grocery_list,
    calculate_recipe_macronutrients,
    estimate_grocery_basket_cost,
    generate_weekly_meal_schedule,
    inspect_household_pantry_inventory,
    submit_grocery_delivery_order,
)


class TestToolsAndSchemas(unittest.TestCase):
    """Verify every tool satisfies strict schema contracts, rich docstrings, and guided errors."""

    def test_all_tools_have_comprehensive_docstrings_and_descriptive_names(self) -> None:
        tools = [
            calculate_recipe_macronutrients,
            audit_meal_plan_allergen_safety,
            generate_weekly_meal_schedule,
            inspect_household_pantry_inventory,
            build_deduplicated_grocery_list,
            estimate_grocery_basket_cost,
            submit_grocery_delivery_order,
        ]
        for fn in tools:
            self.assertGreaterEqual(len(fn.__name__.split("_")), 3)
            doc = inspect.getdoc(fn) or ""
            self.assertGreater(len(doc), 150, f"Docstring too short for {fn.__name__}")
            self.assertIn("Args:", doc)
            self.assertIn("Returns:", doc)

    def test_calculate_recipe_macronutrients_valid_and_guided_error(self) -> None:
        ok = calculate_recipe_macronutrients(
            recipe_name="Chicken Quinoa Bowl",
            servings=2,
            ingredients=[
                {"name": "chicken breast", "grams": 300},
                {"name": "quinoa", "grams": 180},
                {"name": "broccoli", "grams": 150},
            ],
        )
        self.assertEqual(ok["status"], "ok")
        self.assertGreater(ok["per_serving_macros"]["protein_g"], 55.0)

        # Invalid servings should return structured recovery instructions instead of crashing
        err = calculate_recipe_macronutrients(
            recipe_name="Bad Recipe",
            servings=0,
            ingredients=[{"name": "quinoa", "grams": 100}],
        )
        self.assertEqual(err["status"], "error")
        self.assertEqual(err["error_code"], "INVALID_RECIPE_INPUT")
        self.assertIn("recovery_instructions", err)
        self.assertTrue(err["recovery_instructions"])

        # Unknown ingredients should return actionable guidance
        unknown = calculate_recipe_macronutrients(
            recipe_name="Mystery Dish",
            servings=2,
            ingredients=[{"name": "unicorn_dust", "grams": 50}],
        )
        self.assertEqual(unknown["status"], "error")
        self.assertEqual(unknown["error_code"], "UNRECOGNIZED_INGREDIENTS")
        self.assertIn("recovery_instructions", unknown)

    def test_allergen_audit_detects_peanut_and_shellfish_violations(self) -> None:
        unsafe = audit_meal_plan_allergen_safety(
            household_allergens=["peanut", "shellfish"],
            planned_recipes=[
                "Peanut Butter Power Oats",
                "Garlic Shrimp & Cherry Tomato Pasta",
            ],
        )
        self.assertEqual(unsafe["status"], "ok")
        self.assertFalse(unsafe["is_safe"])
        self.assertEqual(len(unsafe["violations"]), 2)
        self.assertIn("sunflower seed butter", unsafe["violations"][0]["safe_substitution_hint"].lower())

        safe = audit_meal_plan_allergen_safety(
            household_allergens=["peanut", "shellfish"],
            dietary_restrictions=["vegan"],
            planned_recipes=[
                "Blueberry Chia Overnight Oats",
                "Mediterranean Chickpea & Quinoa Power Bowl",
                "Spiced Red Lentil & Spinach Coconut Curry",
            ],
        )
        self.assertEqual(safe["status"], "ok")
        self.assertTrue(safe["is_safe"])
        self.assertEqual(safe["violations"], [])

    def test_generate_weekly_meal_schedule_filters_allergens_and_handles_impossible_constraints(self) -> None:
        plan = generate_weekly_meal_schedule(
            days=5,
            household_size=2,
            dietary_restrictions=["vegan"],
            household_allergens=["peanut", "soy"],
            max_prep_minutes=35,
        )
        self.assertEqual(plan["status"], "ok")
        self.assertEqual(plan["days_scheduled"], 5)
        all_titles = [m["recipe_title"] for d in plan["schedule"] for m in d["meals"]]
        audit = audit_meal_plan_allergen_safety(
            household_allergens=["peanut", "soy"],
            dietary_restrictions=["vegan"],
            planned_recipes=all_titles,
        )
        self.assertTrue(audit["is_safe"])

        # Impossible prep time triggers guided error recovery
        impossible = generate_weekly_meal_schedule(
            days=5,
            dietary_restrictions=["vegan"],
            max_prep_minutes=10,
        )
        self.assertEqual(impossible["status"], "error")
        self.assertEqual(impossible["error_code"], "NO_ELIGIBLE_RECIPES_FOR_CONSTRAINTS")
        self.assertIn("max_prep_minutes", impossible["recovery_instructions"])

    def test_pantry_deduplication_and_budget_estimation(self) -> None:
        pantry = inspect_household_pantry_inventory("default_household")
        self.assertEqual(pantry["status"], "ok")
        self.assertTrue(pantry["expiring_soon_highlights"])

        grocery = build_deduplicated_grocery_list(
            household_id="default_household",
            required_ingredients=[
                {"name": "rolled oats", "grams": 300},
                {"name": "rolled oats", "grams": 100},
                {"name": "baby spinach", "grams": 350},
                {"name": "quinoa", "grams": 500},
            ],
            subtract_pantry_inventory=True,
        )
        self.assertEqual(grocery["status"], "ok")
        self.assertIn("rolled oats", grocery["pantry_covered_ingredients"])
        spinach_item = next(
            i for i in grocery["line_items"] if i["ingredient_name"] == "baby spinach"
        )
        self.assertAlmostEqual(spinach_item["pantry_offset_grams"], 220.0)
        self.assertAlmostEqual(spinach_item["net_grams_to_buy"], 130.0)

        basket = estimate_grocery_basket_cost(
            items=[{"name": "salmon fillet", "grams": 1500}],
            weekly_budget_usd=30.0,
            delivery_zip_code="94043",
        )
        self.assertEqual(basket["status"], "ok")
        self.assertFalse(basket["within_budget"])
        self.assertTrue(basket["cost_saving_tips"])


if __name__ == "__main__":
    unittest.main()
