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

from typing import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models import BaseChatModel

from app.ai.guardrails import pii_middleware
from app.ai.middleware import current_time_prompt
from app.ai.prompts import ADMIN_PROMPT, SUMMARY_PROMPT
from app.ai.token_counting import count_tokens
from app.ai.tools import ChatContext, create_parking_tools, notify_admin
from app.config import settings


class UserAgentState(AgentState):
    draft: NotRequired[dict | None]  # set by prepare_reservation's Command


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
def build_admin_agent(llm: BaseChatModel):
    return create_agent(model=llm, tools=[notify_admin], system_prompt=ADMIN_PROMPT)
