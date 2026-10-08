"""The two agents in this app, built with langchain.agents.create_agent and
both run as nodes of ParkingGraph (app/ai/graph.py), which owns the
conversation's persistence and workflow:

1. The user-facing agent (`build_user_agent`) answers questions and
   collects reservation details. Its tools (app/ai/tools.py) talk to the
   real repositories/vector store; `prepare_reservation` returns directly,
   so its review (exact price, hours, availability) is the answer verbatim
   rather than the model's retelling. PII redaction and history
   summarization are its middleware.
2. The admin agent (`build_admin_agent`) drafts the administrator's
   approval-request message from a reservation's details via its
   `notify_admin` tool - the text shown in the admin console
   (app/api/routers/admin.py, ui/admin_app.py) while the conversation
   waits at ParkingGraph's `admin_decision` interrupt.
"""

from datetime import datetime
from typing import NotRequired
from zoneinfo import ZoneInfo

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.ai.guardrails import pii_middleware
from app.ai.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from app.ai.token_counting import count_tokens
from app.ai.tools import ChatContext, create_parking_tools
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class UserAgentState(AgentState):
    draft: NotRequired[dict | None]  # set by prepare_reservation's Command


def build_user_agent(llm: BaseChatModel, summary_llm: BaseChatModel | None = None):
    """Run as ParkingGraph's `agent` node (app/ai/graph.py), which owns the
    conversation's persistence; SummarizationMiddleware folds older turns into
    a summary once the history outgrows history_max_tokens. Per-request
    dependencies arrive through `context` (ChatContext)."""

    @dynamic_prompt
    def _current_time_prompt(_request) -> str:
        now = datetime.now(ZoneInfo(settings.timezone)).strftime(f"%A %Y-%m-%d %H:%M ({settings.timezone})")
        return f"{SYSTEM_PROMPT}\n\nCurrent local date/time: {now}."

    return create_agent(
        model=llm,
        tools=create_parking_tools(),
        state_schema=UserAgentState,
        context_schema=ChatContext,
        middleware=[
            _current_time_prompt,
            # PII first: summaries are built from already-redacted tool results.
            *pii_middleware(),
            SummarizationMiddleware(
                model=summary_llm or llm,
                trigger=("tokens", settings.history_max_tokens),
                keep=("tokens", settings.history_keep_tokens),
                token_counter=count_tokens,
                summary_prompt=SUMMARY_PROMPT,
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Admin agent
# --------------------------------------------------------------------------- #
ADMIN_PROMPT = (
    "You draft short reservation-approval requests for a parking operator's "
    "administrator to review. State only the facts given below; never invent "
    "or infer a detail that isn't listed. Two to four plain sentences, no "
    "greeting or sign-off. Send exactly one message, using notify_admin."
)


class AdminDecision(BaseModel):
    approved: bool = Field(description="True only if the administrator clearly approved the reservation")
    reason: str = Field(description="The administrator's reason or comment; empty string if none")


@tool
def notify_admin(text: str) -> str:
    """Send a reservation approval request to the parking administrator."""
    # No messenger/email is wired up: app/ai/graph.py's request_approval
    # node reads this call's args back out instead of it being sent anywhere.
    logger.info("Drafted admin notification: %s", text)
    return "Request delivered to the administrator."


def build_admin_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=[notify_admin], system_prompt=ADMIN_PROMPT)


def parse_admin_reply(llm: BaseChatModel, reply: str) -> dict:
    """Turn the administrator's free-text answer ("ok", "no, lot is closed")
    into a decision. Not used by today's REST console, which already sends
    a structured decision - kept for a future email/messenger channel."""
    model = llm.with_structured_output(AdminDecision)
    decision = model.invoke(
        [
            (
                "system",
                "Classify the parking administrator's reply to a reservation request. "
                "If it is unclear, treat it as not approved.",
            ),
            ("human", reply),
        ]
    )
    return decision.model_dump()
