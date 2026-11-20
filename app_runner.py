"""Production runner exposing both an async CLI loop and a FastAPI Cloud Run service."""

from __future__ import annotations

import argparse
import asyncio
import uuid
from typing import Any

from fastapi import FastAPI
from google.adk.runners import Runner
from google.genai import types
from pydantic import BaseModel, Field

from meal_planner_agent.agent import build_concierge_app
from meal_planner_agent.config import get_settings
from meal_planner_agent.memory import build_memory_service, build_session_service
from meal_planner_agent.observability import get_structured_logger, log_structured_event
from meal_planner_agent.secrets import get_secret_provider

logger = get_structured_logger("meal_planner_agent.runner")
settings = get_settings()
concierge_app = build_concierge_app(settings)
session_service = build_session_service(settings)
memory_service = build_memory_service(settings)
runner = Runner(
    app=concierge_app,
    session_service=session_service,
    memory_service=memory_service,
)

api = FastAPI(
    title="Mise — Meal Planning Concierge Agent API",
    version="1.0.0",
    description="Multi-agent ADK Concierge API with persistent state, HITL, and OpenTelemetry.",
)


class ChatTurnRequest(BaseModel):
    """Inbound HTTP request for a meal planning turn."""

    user_id: str = Field(default="default_household", min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)


class ChatTurnResponse(BaseModel):
    """Structured HTTP response from the Meal Planning Concierge."""

    session_id: str
    response_text: str
    awaiting_human_confirmation: bool = False


async def run_concierge_turn(
    *,
    user_id: str,
    message: str,
    session_id: str | None = None,
) -> ChatTurnResponse:
    """Execute one persisted conversational turn through the ADK Runner."""
    sid = session_id or f"sess-{uuid.uuid4().hex[:10]}"
    existing = await session_service.get_session(
        app_name=settings.app_name, user_id=user_id, session_id=sid
    )
    if existing is None:
        await session_service.create_session(
            app_name=settings.app_name, user_id=user_id, session_id=sid
        )

    user_content = types.Content(role="user", parts=[types.Part.from_text(text=message)])
    collected_text: list[str] = []
    needs_confirmation = False

    async for event in runner.run_async(
        user_id=user_id,
        session_id=sid,
        new_message=user_content,
    ):
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                if getattr(part, "text", None):
                    collected_text.append(part.text)
                if getattr(part, "function_response", None):
                    resp = getattr(part.function_response, "response", {}) or {}
                    if isinstance(resp, dict) and resp.get("status") == "awaiting_human_confirmation":
                        needs_confirmation = True

    return ChatTurnResponse(
        session_id=sid,
        response_text="\n".join(collected_text).strip(),
        awaiting_human_confirmation=needs_confirmation,
    )


@api.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness and secret-hydration health endpoint for Cloud Run."""
    cred_status = get_secret_provider().hydrate_runtime_credentials()
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "fast_model": settings.fast_model_name,
        "reasoning_model": settings.reasoning_model_name,
        "credentials": cred_status,
    }


@api.post("/v1/chat", response_model=ChatTurnResponse)
async def chat_endpoint(payload: ChatTurnRequest) -> ChatTurnResponse:
    """Run a single meal-planning concierge turn via HTTP."""
    log_structured_event(
        logger,
        "http_chat_request",
        "Received HTTP chat request for Meal Planning Concierge.",
        user_id=payload.user_id,
        session_id=payload.session_id,
    )
    return await run_concierge_turn(
        user_id=payload.user_id,
        message=payload.message,
        session_id=payload.session_id,
    )


async def _interactive_cli(user_id: str, session_id: str | None) -> None:
    sid = session_id or f"cli-{uuid.uuid4().hex[:8]}"
    log_structured_event(
        logger,
        "cli_started",
        "Interactive Meal Planning Concierge CLI started.",
        user_id=user_id,
        session_id=sid,
    )
    while True:
        try:
            prompt = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in {"exit", "quit"}:
            break
        reply = await run_concierge_turn(user_id=user_id, message=prompt, session_id=sid)
        log_structured_event(
            logger,
            "cli_turn_response",
            reply.response_text,
            session_id=reply.session_id,
            awaiting_human_confirmation=reply.awaiting_human_confirmation,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mise Meal Planning Concierge Agent.")
    parser.add_argument("--user-id", default="default_household")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()
    asyncio.run(_interactive_cli(args.user_id, args.session_id))


if __name__ == "__main__":
    main()
