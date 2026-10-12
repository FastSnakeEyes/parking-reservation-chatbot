"""LLM-backed parsers turning free text into structured values."""

from langchain_core.language_models import BaseChatModel

from app.ai.prompts import ADMIN_REPLY_PROMPT
from app.schemas.admin import AdminReplyDecision


def parse_admin_reply(llm: BaseChatModel, reply: str) -> dict:
    """Turn the administrator's free-text answer ("ok", "no, lot is closed")
    into a decision. Not used by today's REST console, which already sends
    a structured decision - kept for a future email/messenger channel."""
    model = llm.with_structured_output(AdminReplyDecision)
    decision = model.invoke([("system", ADMIN_REPLY_PROMPT), ("human", reply)])
    return decision.model_dump()
