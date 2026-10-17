"""Agent middleware not tied to a guardrail (those live in app/ai/guardrails.py)."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, ModelResponse, before_model, dynamic_prompt, wrap_model_call
from langchain_core.messages import AIMessage, ToolMessage

from app.ai.prompts import SYSTEM_PROMPT
from app.ai.tools import request_admin_approval
from app.config import settings


def current_time_text(now: datetime) -> str:
    """Today's local date/time, and the coming week spelled out: models get
    weekday arithmetic wrong ("Wednesday" resolved to a Monday), so they look
    the date up instead of computing it."""
    week = ", ".join((now + timedelta(days=offset)).strftime("%A %Y-%m-%d") for offset in range(1, 8))
    return (
        f"Current local date/time: {now.strftime(f'%A %Y-%m-%d %H:%M ({settings.timezone})')}.\n"
        f"The next 7 days: {week}."
    )


@dynamic_prompt
def current_time_prompt(_request: ModelRequest) -> str:
    """The user agent's system prompt, stamped with the current local date/time
    on every model call so relative dates ("tomorrow", "Friday") resolve correctly."""
    return f"{SYSTEM_PROMPT}\n\n{current_time_text(datetime.now(ZoneInfo(settings.timezone)))}"


@before_model(can_jump_to=["end"])
def end_after_review(state: AgentState, _runtime) -> dict | None:
    """End the user agent's turn right after prepare_reservation produced a
    review (it set the draft): the review is the answer, verbatim. A failed
    prepare_reservation leaves no draft, so the model sees the error and
    explains it to the customer."""
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, ToolMessage) and last.name == "prepare_reservation" and state.get("draft") is not None:
        return {"jump_to": "end"}
    return None


@wrap_model_call
async def force_approval_request(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """The admin agent's first model call must propose request_admin_approval, so
    every reservation reaches the administrator's review. Later calls (after the
    decision) are left free - forcing those would loop forever."""
    if not any(isinstance(message, AIMessage) for message in request.messages):
        request = request.override(tool_choice=request_admin_approval.name)
    return await handler(request)


@wrap_model_call
async def end_after_rejection(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """After the administrator rejects the request, the admin agent's turn is
    over: answer with a fixed closing message instead of calling the model.
    (An approval needs no counterpart - request_admin_approval returns directly.)"""
    last = request.messages[-1] if request.messages else None
    if isinstance(last, ToolMessage) and last.name == request_admin_approval.name and last.status == "error":
        # HumanInTheLoopMiddleware's rejection: the tool never ran.
        return ModelResponse(result=[AIMessage("The administrator rejected the reservation.")])
    return await handler(request)
