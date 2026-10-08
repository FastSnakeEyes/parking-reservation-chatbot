"""The two agents in this app, built with langchain.agents.create_agent:

1. The user-facing agent (`build_user_agent`) answers questions and
   collects reservation details. Its tools (app/ai/tools.py) talk to the
   real repositories/vector store; `prepare_reservation` returns directly,
   so its review (exact price, hours, availability) is the answer verbatim
   rather than the model's retelling. Memory, summarization, the
   cancel/confirm commands and the injection guard all live in this
   agent's checkpointer and middleware (app/ai/middleware.py).
2. The admin agent (`build_admin_agent`) drafts the administrator's
   approval-request message from a reservation's details via its
   `notify_admin` tool. Stage 2's actual approve/reject exchange still
   happens over the REST API (app/api/routers/admin.py) and the
   Streamlit console (ui/admin_app.py) - `notify_admin` only produces the
   text shown there.

app/ai/graph.py's `ApprovalPipeline` is the LangGraph graph that
orchestrates agent 2: its `request_approval` node runs this agent and
extracts the drafted text, `admin_decision` pauses for the
administrator's decision (interrupt), then `record_reservation` calls
the Stage 3 MCP tool on approval.
"""

import re
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, SummarizationMiddleware, dynamic_prompt
from langchain.agents.middleware._redaction import PIIMatch, detect_email
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field

from app.ai.guardrails import get_pii_guard
from app.ai.middleware import ChatState, session_commands
from app.ai.prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from app.ai.token_counting import count_tokens
from app.ai.tools import ChatContext, create_parking_tools
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


PHONE_PATTERN = r"(?:\+\d[\d -]{7,14}\d|\b8[ -]?6\d{2}[ -]?\d{5}\b)"  # +370 612 34567, 8 612 34567
CAR_NUMBER_PATTERN = r"\b(?!EUR|USD)[A-Z]{3}[ -]?\d{3}\b"  # ABC123, ABC 123, ABC-123


def _normalize_contact(value: str) -> str:
    return re.sub(r"[\s-]", "", value).lower()


def _skip_public_contacts(detector: Callable[[str], list[PIIMatch]]) -> Callable[[str], list[PIIMatch]]:
    """The operator's own published phone/email (settings.public_contacts, e.g. the
    support line) is meant to be shared, so it is not treated as PII."""

    def detect(content: str) -> list[PIIMatch]:
        public = {_normalize_contact(c) for c in settings.public_contacts}
        return [m for m in detector(content) if _normalize_contact(m["value"]) not in public]

    return detect


@lru_cache(maxsize=256)
def _person_parts(content: str) -> tuple[tuple[str, int, int], ...]:
    """(part, start, end) for each detected person: first word = name, the rest = surname.
    Cached so Presidio runs once per text, not once per detector."""
    parts = []
    for r in get_pii_guard().analyzer.analyze(text=content, entities=["PERSON"], language="en", score_threshold=0.6):
        words = list(re.finditer(r"\S+", content[r.start : r.end]))
        if not words:
            continue
        parts.append(("name", r.start + words[0].start(), r.start + words[0].end()))
        if len(words) > 1:
            parts.append(("surname", r.start + words[1].start(), r.start + words[-1].end()))
    return tuple(parts)


def _detect_phone(content: str) -> list[PIIMatch]:
    return [
        {"type": "phone", "value": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(PHONE_PATTERN, content)
    ]


def _detect_name(content: str) -> list[dict]:
    return [
        {"type": "name", "value": content[s:e], "start": s, "end": e}
        for part, s, e in _person_parts(content)
        if part == "name"
    ]


def _detect_surname(content: str) -> list[dict]:
    return [
        {"type": "surname", "value": content[s:e], "start": s, "end": e}
        for part, s, e in _person_parts(content)
        if part == "surname"
    ]


def _pii_guardrails() -> list[PIIMiddleware]:
    """Redact personal data before it reaches the model or the customer.

    Contact and payment data (email, phone, card) is never needed for a reservation,
    so it is redacted everywhere: customer input, tool results and the model's answer -
    except the operator's own settings.public_contacts, which customers may ask for.

    Name, surname and car number ARE needed for the customer's own reservation, so they
    are redacted only in tool results. That hides other customers' data coming back from
    Milvus / PostgreSQL while the customer's own details still pass through.

    Prompt-injection detection stays in app/ai/guardrails.py."""
    everywhere = {"strategy": "redact", "apply_to_output": True, "apply_to_tool_results": True}
    tool_results_only = {
        "strategy": "redact",
        "apply_to_input": False,
        "apply_to_output": False,
        "apply_to_tool_results": True,
    }

    return [
        PIIMiddleware("email", detector=_skip_public_contacts(detect_email), **everywhere),
        PIIMiddleware("credit_card", **everywhere),
        PIIMiddleware("phone", detector=_skip_public_contacts(_detect_phone), **everywhere),
        PIIMiddleware("name", detector=_detect_name, **tool_results_only),
        PIIMiddleware("surname", detector=_detect_surname, **tool_results_only),
        PIIMiddleware("car_number", detector=CAR_NUMBER_PATTERN, **tool_results_only),
    ]


def build_user_agent(llm: BaseChatModel, checkpointer: BaseCheckpointSaver, summary_llm: BaseChatModel | None = None):
    """Built once at startup. Conversation memory is the checkpointer's thread
    (one per chat session, see ParkingChat); SummarizationMiddleware folds
    older turns into a summary once the history outgrows history_max_tokens.
    Per-request dependencies arrive through `context` (ChatContext)."""

    @dynamic_prompt
    def _current_time_prompt(_request) -> str:
        now = datetime.now(ZoneInfo(settings.timezone)).strftime(f"%A %Y-%m-%d %H:%M ({settings.timezone})")
        return f"{SYSTEM_PROMPT}\n\nCurrent local date/time: {now}."

    return create_agent(
        model=llm,
        tools=create_parking_tools(),
        state_schema=ChatState,
        context_schema=ChatContext,
        checkpointer=checkpointer,
        middleware=[
            session_commands,
            _current_time_prompt,
            # PII first: summaries are built from already-redacted tool results.
            *_pii_guardrails(),
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
