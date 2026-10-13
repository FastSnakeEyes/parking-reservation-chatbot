"""Agent middleware not tied to a guardrail (those live in app/ai/guardrails.py)."""

from collections.abc import Awaitable, Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt, wrap_model_call
from langchain_core.messages import AIMessage

from app.ai.prompts import SYSTEM_PROMPT
from app.ai.tools import approve_reservation
from app.config import settings


@dynamic_prompt
def current_time_prompt(_request: ModelRequest) -> str:
    """The user agent's system prompt, stamped with the current local date/time
    on every model call so relative dates ("tomorrow", "Friday") resolve correctly."""
    now = datetime.now(ZoneInfo(settings.timezone)).strftime(f"%A %Y-%m-%d %H:%M ({settings.timezone})")
    return f"{SYSTEM_PROMPT}\n\nCurrent local date/time: {now}."


@wrap_model_call
async def force_approval_request(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """The admin agent's first model call must propose approve_reservation, so
    every reservation reaches the administrator's review. Later calls (after the
    decision) are left free - forcing those would loop forever."""
    if not any(isinstance(message, AIMessage) for message in request.messages):
        request = request.override(tool_choice=approve_reservation.name)
    return await handler(request)
