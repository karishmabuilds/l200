"""Multi-Agent Meal Planning Concierge architecture built with Google ADK.

Orchestration patterns implemented:
- Root Coordinator (`meal_concierge_coordinator`, Gemini Pro reasoning tier)
- Sequential Pipeline (`meal_planning_pipeline`):
    1. Parallel Context Gathering (`context_and_nutrition_gatherer`):
       - `pantry_inventory_analyst` (Gemini Flash fast tier)
       - `dietary_profile_researcher` (Gemini Flash fast tier)
    2. `chef_nutrition_planner` (Gemini Pro reasoning tier)
    3. `meal_plan_safety_auditor` (Gemini Pro reasoning tier)
- Grocery Fulfillment Specialist (`grocery_fulfillment_specialist`, Gemini Flash fast tier)
  with Human-in-the-Loop (HITL) gated checkout (`submit_grocery_delivery_order`).
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

from meal_planner_agent.config import (
    AgentSettings,
    StrategicModelRouter,
    TaskComplexity,
    get_settings,
)
from meal_planner_agent.guardrails import (
    GuardrailsAndPolicyPlugin,
    HumanInTheLoopApprovalPlugin,
)
from meal_planner_agent.memory import (
    AsyncMemoryConsolidationPlugin,
    AsyncMemoryConsolidationWorker,
    PersistentMealMemoryStore,
    build_context_cache_config,
    build_events_compaction_config,
)
from meal_planner_agent.observability import ObservabilityAndTracingPlugin, configure_tracing
from meal_planner_agent.prompts import (
    CHEF_NUTRITION_PLANNER_INSTRUCTION,
    CONCIERGE_CONSTITUTION,
    DIETARY_PROFILE_RESEARCHER_INSTRUCTION,
    GROCERY_FULFILLMENT_INSTRUCTION,
    PANTRY_ANALYST_INSTRUCTION,
    SAFETY_AUDITOR_INSTRUCTION,
)
from meal_planner_agent.secrets import get_secret_provider
from meal_planner_agent.tools import (
    audit_meal_plan_allergen_safety,
    build_deduplicated_grocery_list,
    calculate_recipe_macronutrients,
    estimate_grocery_basket_cost,
    generate_weekly_meal_schedule,
    inspect_household_pantry_inventory,
    requires_grocery_order_confirmation,
    submit_grocery_delivery_order,
)


def build_meal_planning_agents(
    settings: AgentSettings | None = None,
) -> tuple[LlmAgent, dict[str, str]]:
    """Construct the multi-agent hierarchy with strategic Flash/Pro model routing.

    Returns:
        Tuple of ``(root_agent, model_routing_summary)`` where ``model_routing_summary``
        maps each agent name to its assigned Gemini model tier.
    """
    cfg = settings or get_settings()
    router = StrategicModelRouter(cfg)

    pantry_model = router.select_model(TaskComplexity.PANTRY_LOOKUP)
    profile_model = router.select_model(TaskComplexity.PROFILE_RETRIEVAL)
    planner_model = router.select_model(TaskComplexity.MEAL_PLAN_SYNTHESIS)
    auditor_model = router.select_model(TaskComplexity.ALLERGEN_SAFETY_AUDIT)
    grocery_model = router.select_model(TaskComplexity.GROCERY_LIST_OPTIMIZATION)
    coordinator_model = router.select_model(TaskComplexity.ORCHESTRATION)

    # 1. Parallel Phase: Fast context gathering (Gemini Flash tier)
    pantry_inventory_analyst = LlmAgent(
        name="pantry_inventory_analyst",
        model=pantry_model,
        description="Inspects stocked pantry quantities and expiring perishable items.",
        instruction=PANTRY_ANALYST_INSTRUCTION,
        tools=[inspect_household_pantry_inventory],
        output_key="pantry_summary",
    )

    dietary_profile_researcher = LlmAgent(
        name="dietary_profile_researcher",
        model=profile_model,
        description="Extracts household allergies, dietary restrictions, calories, and budget.",
        instruction=DIETARY_PROFILE_RESEARCHER_INSTRUCTION,
        output_key="dietary_profile",
    )

    context_and_nutrition_gatherer = ParallelAgent(
        name="context_and_nutrition_gatherer",
        description="Concurrently retrieves household pantry status and dietary constraints.",
        sub_agents=[pantry_inventory_analyst, dietary_profile_researcher],
    )

    # 2. Planning Phase: Deep multi-constraint meal scheduling (Gemini Pro reasoning tier)
    chef_nutrition_planner = LlmAgent(
        name="chef_nutrition_planner",
        model=planner_model,
        description="Creates allergen-filtered weekly meal schedules and calculates recipe macros.",
        instruction=CHEF_NUTRITION_PLANNER_INSTRUCTION,
        tools=[generate_weekly_meal_schedule, calculate_recipe_macronutrients],
        output_key="draft_meal_plan",
    )

    # 3. Safety Verification Phase: Independent allergen & nutrition audit (Gemini Pro tier)
    meal_plan_safety_auditor = LlmAgent(
        name="meal_plan_safety_auditor",
        model=auditor_model,
        description="Independently audits proposed recipes for allergens and dietary violations.",
        instruction=SAFETY_AUDITOR_INSTRUCTION,
        tools=[audit_meal_plan_allergen_safety],
        output_key="safety_audit_report",
    )

    meal_planning_pipeline = SequentialAgent(
        name="meal_planning_pipeline",
        description=(
            "End-to-end sequential meal planning workflow: parallel context gathering -> "
            "nutritional schedule synthesis -> independent allergen safety audit."
        ),
        sub_agents=[
            context_and_nutrition_gatherer,
            chef_nutrition_planner,
            meal_plan_safety_auditor,
        ],
    )

    # 4. Grocery Fulfillment Specialist with HITL-gated delivery checkout
    hitl_order_tool = FunctionTool(
        submit_grocery_delivery_order,
        require_confirmation=requires_grocery_order_confirmation,
    )

    grocery_fulfillment_specialist = LlmAgent(
        name="grocery_fulfillment_specialist",
        model=grocery_model,
        description=(
            "Builds pantry-deduplicated grocery lists, estimates basket costs vs. budget, "
            "and manages Human-in-the-Loop confirmed grocery delivery orders."
        ),
        instruction=GROCERY_FULFILLMENT_INSTRUCTION,
        tools=[
            build_deduplicated_grocery_list,
            estimate_grocery_basket_cost,
            hitl_order_tool,
        ],
        output_key="grocery_fulfillment_summary",
    )

    # 5. Root Coordinator Agent (Gemini Pro reasoning tier)
    coordinator = LlmAgent(
        name="meal_concierge_coordinator",
        model=coordinator_model,
        description=(
            "Mise: Personal Meal Planning Concierge coordinating multi-agent meal "
            "planning, allergen safety verification, and HITL-protected grocery fulfillment."
        ),
        instruction=CONCIERGE_CONSTITUTION,
        sub_agents=[meal_planning_pipeline, grocery_fulfillment_specialist],
        tools=[
            inspect_household_pantry_inventory,
            calculate_recipe_macronutrients,
            audit_meal_plan_allergen_safety,
        ],
    )

    routing_map = {
        "meal_concierge_coordinator": coordinator_model,
        "pantry_inventory_analyst": pantry_model,
        "dietary_profile_researcher": profile_model,
        "chef_nutrition_planner": planner_model,
        "meal_plan_safety_auditor": auditor_model,
        "grocery_fulfillment_specialist": grocery_model,
    }
    return coordinator, routing_map


def build_concierge_app(
    settings: AgentSettings | None = None,
    *,
    memory_store: PersistentMealMemoryStore | None = None,
) -> App:
    """Assemble the production ADK App with compaction, caching, resumability, and plugins."""
    cfg = settings or get_settings()
    get_secret_provider().hydrate_runtime_credentials()
    configure_tracing(cfg)

    coordinator, _ = build_meal_planning_agents(cfg)
    worker = AsyncMemoryConsolidationWorker(
        store=memory_store or PersistentMealMemoryStore(db_path=cfg.memory_db_path, settings=cfg)
    )

    return App(
        name=cfg.app_name,
        root_agent=coordinator,
        plugins=[
            ObservabilityAndTracingPlugin(),
            GuardrailsAndPolicyPlugin(settings=cfg),
            HumanInTheLoopApprovalPlugin(settings=cfg),
            AsyncMemoryConsolidationPlugin(worker=worker),
        ],
        events_compaction_config=build_events_compaction_config(cfg),
        context_cache_config=build_context_cache_config(cfg),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


# Standard ADK module-level exports discovered by `adk web`, `adk run`, `adk eval`, and `agents-cli`
app = build_concierge_app()
root_agent = app.root_agent
