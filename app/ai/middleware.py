"""Agent middleware not tied to a guardrail (those live in app/ai/guardrails.py)."""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents.middleware import ModelRequest, dynamic_prompt

from app.ai.prompts import SYSTEM_PROMPT
from app.config import settings


@dynamic_prompt
def current_time_prompt(_request: ModelRequest) -> str:
    """The user agent's system prompt, stamped with the current local date/time
    on every model call so relative dates ("tomorrow", "Friday") resolve correctly."""
    now = datetime.now(ZoneInfo(settings.timezone)).strftime(f"%A %Y-%m-%d %H:%M ({settings.timezone})")
    return f"{SYSTEM_PROMPT}\n\nCurrent local date/time: {now}."
