"""The two agents in this app, built with langchain.agents.create_agent and
both run as nodes of ParkingGraph (app/ai/graph.py), which owns the
conversation's persistence and workflow:

1. The user-facing agent (`build_user_agent`) answers questions and
   collects reservation details. Its tools (app/ai/tools.py) talk to the
   real repositories/vector store; `prepare_reservation` returns directly,
   so its review (exact price, hours, availability) is the answer verbatim
   rather than the model's retelling. PII redaction and history
   summarization are its middleware.
2. The admin agent (`build_admin_agent`) is the human-in-the-loop agent:
   it proposes an `request_admin_approval` call whose summary describes the
   reservation, and HumanInTheLoopMiddleware pauses on it until the
   administrator approves or rejects it through the admin API
   (app/api/routers/admin.py, ui/admin_app.py). ParkingGraph's `record`
   node then acts on that decision.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolCall

from app.ai.guardrails import pii_middleware
from app.ai.middleware import current_time_prompt, end_after_rejection, force_approval_request
from app.ai.prompts import ADMIN_PROMPT, SUMMARY_PROMPT
from app.ai.states import UserAgentState
from app.ai.token_counting import count_tokens
from app.ai.tools import ChatContext, create_parking_tools, request_admin_approval
from app.config import settings


def build_user_agent(llm: BaseChatModel, summary_llm: BaseChatModel | None = None):
    """Run as ParkingGraph's `agent` node (app/ai/graph.py), which owns the
    conversation's persistence; SummarizationMiddleware folds older turns into
    a summary once the history outgrows history_max_tokens. Per-request
    dependencies arrive through `context` (ChatContext)."""

    return create_agent(
        model=llm,
        tools=create_parking_tools(),
        state_schema=UserAgentState,
        context_schema=ChatContext,
        middleware=[
            current_time_prompt,
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
def _review_description(tool_call: ToolCall, _state, _runtime) -> str:
    """What the administrator sees: the agent's summary of the reservation."""
    return tool_call["args"]["summary"]


def build_admin_agent(llm: BaseChatModel):
    """Run as ParkingGraph's `admin` node; its interrupt pauses the whole
    conversation until ParkingGraph.resume() delivers the decision."""
    return create_agent(
        model=llm,
        tools=[request_admin_approval],
        system_prompt=ADMIN_PROMPT,
        middleware=[
            # Exactly one model call per review: forced to request approval,
            # then never called again (return_direct / end_after_rejection).
            force_approval_request,
            end_after_rejection,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    request_admin_approval.name: {
                        "allowed_decisions": ["approve", "reject"],
                        "description": _review_description,
                    }
                }
            ),
        ],
    )
