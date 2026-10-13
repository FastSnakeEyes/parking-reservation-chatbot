"""ParkingGraph: the whole customer conversation as one LangGraph workflow,
one checkpointer thread per chat session ("chat:{session_id}").

    START -> user_agent -+-> admin -> record -> notification -> END   (after `confirm`)
                         +-> END                                       (any other turn)

- user_agent: the customer's turn. Commands are handled first, without the
  model: the admin's decision notice, prompt injection, `cancel`, and
  `confirm`, which saves the reviewed booking as a pending reservation (the
  model cannot authorize a DB write) and hands it to `admin`. Everything else
  goes to the user agent (app/ai/agents.py), which answers or prepares a review.
- admin: the human-in-the-loop admin agent proposes approving the
  reservation; its HumanInTheLoopMiddleware pauses the whole thread until the
  admin API resumes it (ParkingGraph.resume) - so while a reservation awaits
  approval, the conversation waits too.
- record: acts on the decision - records an approval through MCP, or marks
  the reservation rejected.
- notification: leaves the customer the outcome for their next message.

The graph calls ParkingService (app/services/parking.py) for every DB/MCP
side effect; the service never calls the graph. Per-request dependencies
arrive as runtime context (ChatContext).
"""

from typing import Literal
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from redis.asyncio import Redis
from starlette.concurrency import run_in_threadpool

from app.ai.agents import build_admin_agent, build_user_agent
from app.ai.guardrails import is_prompt_injection
from app.ai.states import ParkingState
from app.ai.tools import ChatContext
from app.constants import (
    ADMIN_LOCK_WAIT_SECONDS,
    CHAT_LOCK_WAIT_SECONDS,
    CHAT_THREAD_PREFIX,
    THREAD_LOCK_TIMEOUT_SECONDS,
)
from app.enums import ApprovalDecision, ChatStatus, ReservationStatus
from app.logger import get_logger

logger = get_logger(__name__)


def _reply(text: str, *, drop: str, **updates) -> dict:
    """Answer `text`. `drop` removes the triggering human message (or, with
    REMOVE_ALL_MESSAGES, the whole conversation) from the history."""
    return {"messages": [RemoveMessage(id=drop), AIMessage(content=text)], **updates}


async def _handle_command(state: ParkingState, runtime: Runtime[ChatContext], config: RunnableConfig) -> dict | None:
    """The deterministic, model-free part of the customer's turn: its state
    updates, or None when the message is for the user agent."""
    question = state["messages"][-1]
    text = str(question.content).strip()

    if state.get("notice_pending"):
        # The decision notice is already the last message (notification);
        # dropping the question makes it this turn's answer.
        return {"messages": [RemoveMessage(id=question.id)], "notice_pending": False}

    if await run_in_threadpool(is_prompt_injection, text):
        logger.warning("Blocked a suspected prompt injection")
        # Dropped from the history too, so it never reaches the model on a later turn.
        return _reply(
            "I can't process that request. Please ask about parking information or reservations.",
            drop=question.id,
        )

    command = text.casefold()

    if command == "cancel":
        return _reply("The conversation and unsent booking draft were cleared.", drop=REMOVE_ALL_MESSAGES, draft=None)

    if command != "confirm":
        return None

    if state.get("draft") is None:
        previous_id = state.get("last_reservation_id")
        if previous_id:
            return _reply(f"Your previous request was already submitted.\nReference: {previous_id}", drop=question.id)
        return _reply("There is no reviewed reservation to confirm.", drop=question.id)

    # Saved here, not in `admin`: a node's updates are only committed when it
    # finishes, and `admin` pauses until the administrator decides.
    try:
        reservation_id = await runtime.context.parking_service.book(
            state["draft"], conversation_id=config["configurable"]["thread_id"]
        )
    except ValueError as exc:
        return _reply(
            f"The request cannot be submitted: {exc}. Please provide corrected details.",
            drop=question.id,
            draft=None,
        )

    # Start fresh after a submitted booking.
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage(
                "Your reservation request was saved and is awaiting administrator approval.\n"
                f"Reference: {reservation_id}"
            ),
        ],
        "draft": None,
        "last_reservation_id": str(reservation_id),
        "reservation_status": ReservationStatus.PENDING.value,
        "submitted": True,
    }


def _after_user_agent(state: ParkingState) -> Literal["admin", "__end__"]:
    """A booking saved by this turn's `confirm` goes to the administrator."""
    return "admin" if state.get("submitted") else END


def _build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver, summary_llm: BaseChatModel | None = None):
    user_agent = build_user_agent(llm, summary_llm)
    admin_agent = build_admin_agent(llm)

    # -- Node: the customer's turn ------------------------------------------ #
    async def user_agent_node(
        state: ParkingState,
        runtime: Runtime[ChatContext],
        config: RunnableConfig,
    ) -> dict:
        # `submitted` is set fresh every turn; only `confirm` sets it to True.
        if (updates := await _handle_command(state, runtime, config)) is not None:
            return {"submitted": False, **updates}

        # Anything else invalidates an old review (a correction prepares a new one).
        result = await user_agent.ainvoke(
            {"messages": state["messages"], "draft": None},
            context=runtime.context,
        )
        # The agent's final message list replaces ours: its middleware may have
        # redacted earlier messages or folded them into a summary.
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *result["messages"]],
            "draft": result.get("draft"),
            "submitted": False,
        }

    # -- Node: the human-in-the-loop admin agent ---------------------------- #
    async def admin(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        facts = await runtime.context.parking_service.approval_facts(UUID(state["last_reservation_id"]))
        request = "\n".join(f"{key}: {value}" for key, value in facts.items())

        # Pauses inside the agent (HumanInTheLoopMiddleware) until the admin API
        # resumes the thread; this node then re-runs and the agent picks up where
        # it paused, so its model is not called again for the same request.
        await admin_agent.ainvoke({"messages": [HumanMessage(f"New reservation request:\n{request}")]})

        if state.get("decision") is None:
            # force_approval_request makes the agent propose the call, so it
            # always pauses for a decision - this means that guarantee broke.
            raise RuntimeError("The admin agent finished without the administrator's decision")
        return {}

    # -- Node: act on the administrator's decision ------------------------- #
    async def record(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        status = await runtime.context.parking_service.record_decision(
            UUID(state["last_reservation_id"]),
            approved=state["decision"] == ApprovalDecision.APPROVED,
            note=state.get("admin_note"),
        )
        return {"reservation_status": str(status)}

    # -- Node: tell the customer the outcome on their next message ---------- #
    def notification(state: ParkingState) -> dict:
        outcomes = {
            ReservationStatus.APPROVED: "approved",
            ReservationStatus.REJECTED: "rejected by the administrator",
        }
        outcome = outcomes.get(
            state["reservation_status"], "reviewed, but could not be recorded yet - we will follow up"
        )
        notice = (
            f"Update: your reservation {state['last_reservation_id']} was {outcome}.\nYou can ask your question again."
        )

        return {
            "messages": [AIMessage(notice)],
            "decision": None,
            "admin_note": None,
            "notice_pending": True,
        }

    builder = StateGraph(ParkingState, context_schema=ChatContext)
    builder.add_node("user_agent", user_agent_node)
    builder.add_node("admin", admin)
    builder.add_node("record", record)
    builder.add_node("notification", notification)

    builder.add_edge(START, "user_agent")
    builder.add_conditional_edges("user_agent", _after_user_agent, ["admin", END])
    builder.add_edge("admin", "record")
    builder.add_edge("record", "notification")
    builder.add_edge("notification", END)

    return builder.compile(checkpointer=checkpointer)


class ParkingGraph:
    """Built once at startup (app/lifespan.py); called by the API routers."""

    def __init__(
        self,
        llm: BaseChatModel,
        checkpointer: BaseCheckpointSaver,
        redis: Redis,
        summary_llm: BaseChatModel | None = None,
    ):
        """summary_llm summarizes long histories (defaults to llm); it must
        accept requests without tools - see app/lifespan.py."""
        self.graph = _build_graph(llm, checkpointer, summary_llm)
        self.redis = redis

    @staticmethod
    def thread_id(session_id: str) -> str:
        return f"{CHAT_THREAD_PREFIX}{session_id}"

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def lock(self, thread_id: str, blocking_timeout: float = 0):
        """The thread's Redis lock - held by chat turns, admin resumes and the
        idle-thread cleanup (app/ai/chat_cleanup.py), which all write the thread."""
        return self.redis.lock(
            f"parking:graph:{thread_id}:lock", timeout=THREAD_LOCK_TIMEOUT_SECONDS, blocking_timeout=blocking_timeout
        )

    async def _waiting_for(self, thread_id: str) -> dict | None:
        """{"reservation_id", "message", "stalled"} if the thread is waiting on the
        administrator: paused for their review (the admin agent's
        HumanInTheLoopMiddleware interrupt), or stalled after their decision
        because a later step failed (e.g. the MCP server was down while
        recording) - resume() retries that step."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        reservation_id = snapshot.values.get("last_reservation_id")

        for task in snapshot.tasks:
            for pending in task.interrupts:
                return {
                    "reservation_id": reservation_id,
                    "message": pending.value["action_requests"][0]["description"],
                    "stalled": False,
                }

        if snapshot.next and any(task.error for task in snapshot.tasks):
            return {
                "reservation_id": reservation_id,
                "message": (
                    f"The decision ({snapshot.values.get('decision')}) could not be recorded "
                    f"(step '{snapshot.next[0]}' failed). Submit a decision again to retry."
                ),
                "stalled": True,
            }
        return None

    async def chat(self, session_id: str, question: str, context: ChatContext) -> dict:
        """One customer turn. Returns {"status", "answer"}; status is
        "waiting_for_admin" while the conversation is paused for approval."""
        thread_id = self.thread_id(session_id)
        lock = self.lock(thread_id, blocking_timeout=CHAT_LOCK_WAIT_SECONDS)

        if not await lock.acquire():
            return {"status": ChatStatus.COMPLETE, "answer": "Your previous message is still being processed."}

        try:
            # New input would start a fresh run and abandon the paused one, so
            # a waiting conversation only answers with its status.
            waiting = await self._waiting_for(thread_id)
            if waiting is not None:
                return {
                    "status": ChatStatus.WAITING_FOR_ADMIN,
                    "answer": f"Your reservation {waiting['reservation_id']} is still awaiting administrator approval.",
                }

            try:
                result = await self.graph.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config=self._config(thread_id),
                    context=context,
                    durability="exit",  # one checkpoint per turn, not one per step
                )
            except GraphRecursionError:
                return {
                    "status": ChatStatus.COMPLETE,
                    "answer": "I could not finish processing that request. Please try a more specific question.",
                }
        finally:
            await lock.release()

        return {
            "status": ChatStatus.WAITING_FOR_ADMIN if result.get("__interrupt__") else ChatStatus.COMPLETE,
            "answer": _text(result["messages"][-1].content),
            # A reviewed booking is waiting for the customer's `confirm`.
            "awaiting_confirmation": result.get("draft") is not None,
        }

    async def updates(self, session_id: str) -> dict:
        """What the chat page polls while a reservation awaits approval:
        {"status", "notice", "reservation_status"}. The decision notice is
        returned once and then marked delivered, so the customer's next message
        goes to the user agent instead of being answered with it again."""
        thread_id = self.thread_id(session_id)
        config = self._config(thread_id)

        if await self._waiting_for(thread_id) is not None:
            return {"status": ChatStatus.WAITING_FOR_ADMIN, "notice": None, "reservation_status": None}

        snapshot = await self.graph.aget_state(config)
        if not snapshot.values.get("notice_pending"):
            return {"status": ChatStatus.COMPLETE, "notice": None, "reservation_status": None}

        lock = self.lock(thread_id, blocking_timeout=CHAT_LOCK_WAIT_SECONDS)
        if not await lock.acquire():
            # A chat turn is running - it delivers the notice itself; poll again.
            return {"status": ChatStatus.COMPLETE, "notice": None, "reservation_status": None}

        try:
            snapshot = await self.graph.aget_state(config)  # re-read under the lock
            if not snapshot.values.get("notice_pending"):
                return {"status": ChatStatus.COMPLETE, "notice": None, "reservation_status": None}

            # As if written by `notification`, so the thread still ends there.
            await self.graph.aupdate_state(config, {"notice_pending": False}, as_node="notification")
        finally:
            await lock.release()

        return {
            "status": ChatStatus.COMPLETE,
            "notice": _text(snapshot.values["messages"][-1].content),
            "reservation_status": snapshot.values.get("reservation_status"),
        }

    async def pending_message(self, thread_id: str) -> str | None:
        """The request the administrator reviews, read back from the paused
        interrupt rather than drafted again on every admin console poll."""
        waiting = await self._waiting_for(thread_id)
        return waiting["message"] if waiting else None

    async def resume(
        self,
        thread_id: str,
        reservation_id: UUID,
        decision: ApprovalDecision,
        note: str | None,
        context: ChatContext,
    ) -> ReservationStatus:
        """Resume a conversation paused for the administrator; returns the
        reservation's resulting status. Raises LookupError if that thread is
        not waiting on this reservation."""
        async with self.lock(thread_id, blocking_timeout=ADMIN_LOCK_WAIT_SECONDS):
            waiting = await self._waiting_for(thread_id)
            if waiting is None or waiting["reservation_id"] != str(reservation_id):
                raise LookupError(reservation_id)

            decision = ApprovalDecision(decision)
            config = self._config(thread_id)

            if waiting["stalled"]:
                # A step after the decision failed: record the (possibly revised)
                # decision as if `admin` had just finished, then run on from there.
                logger.warning(
                    "Retrying the stalled approval of reservation %s (decision=%s)", reservation_id, decision
                )
                await self.graph.aupdate_state(
                    config, {"decision": decision.value, "admin_note": note}, as_node="admin"
                )
                result = await self.graph.ainvoke(None, config=config, context=context, durability="exit")
                return ReservationStatus(result["reservation_status"])

            if decision == ApprovalDecision.APPROVED:
                review = {"type": "approve"}
            else:
                review = {"type": "reject", "message": note or "Rejected by the administrator."}

            logger.info("Resuming conversation for reservation %s with decision=%s", reservation_id, decision)
            result = await self.graph.ainvoke(
                # resume answers the HumanInTheLoopMiddleware interrupt; update
                # records the decision for the nodes that act on it.
                Command(
                    resume={"decisions": [review]},
                    update={"decision": decision.value, "admin_note": note},
                ),
                config=config,
                context=context,
                durability="exit",
            )

        return ReservationStatus(result["reservation_status"])


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))
