"""ParkingGraph: the whole customer conversation as one LangGraph workflow,
one checkpointer thread per chat session ("chat:{session_id}").

    START -> route_input -+-> agent ------------------------------------------> END
                          +-> book -> admin_review -+-----------------> notify_customer -> END
                          |                         +-> manual_review -^
                          +-> END   (notice / injection / cancel / nothing to confirm)

- route_input: deterministic, model-free handling of the turn - the admin's
  decision notice, prompt injection, `cancel`, `confirm` (the model cannot
  authorize a DB write). Everything else goes to the user agent.
- agent: the user agent (app/ai/agents.py) answers or prepares a review.
- book -> admin_review: on `confirm`, the reservation is saved and the
  human-in-the-loop admin agent proposes recording it; its
  HumanInTheLoopMiddleware pauses the whole thread until the admin API
  resumes it (ParkingGraph.resume) - so while a reservation awaits approval,
  the conversation waits too. Approved, the agent's tool records it (MCP);
  rejected, the tool never runs and the reservation is marked rejected.
- manual_review: the same pause, without the agent, if its model never
  proposed the call - the administrator still gets a request to decide on.
- notify_customer: leaves the customer a notice for their next message.

The graph calls ParkingService (app/services/parking.py) for every DB/MCP
side effect; the service never calls the graph. Per-request dependencies
arrive as runtime context (ChatContext).
"""

from typing import Annotated, Literal, NotRequired, TypedDict
from uuid import UUID

from langchain.agents.middleware.human_in_the_loop import ActionRequest, HITLRequest, ReviewConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from redis.asyncio import Redis
from starlette.concurrency import run_in_threadpool

from app.ai.agents import build_admin_agent, build_user_agent
from app.ai.guardrails import is_prompt_injection
from app.ai.tools import ChatContext, record_reservation
from app.constants import (
    ADMIN_LOCK_WAIT_SECONDS,
    CHAT_LOCK_WAIT_SECONDS,
    CHAT_THREAD_PREFIX,
    THREAD_LOCK_TIMEOUT_SECONDS,
)
from app.enums import ApprovalDecision, ChatStatus, ReservationStatus
from app.logger import get_logger

logger = get_logger(__name__)


# Enum fields hold the enum's plain string value: the checkpointer only loads
# its own registered types back, so enum members must not be persisted.
class ParkingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: NotRequired[dict | None]  # set by prepare_reservation, consumed by `confirm`
    last_reservation_id: NotRequired[str | None]
    decision: NotRequired[str | None]  # an ApprovalDecision value, set by ParkingGraph.resume
    admin_note: NotRequired[str | None]
    reservation_status: NotRequired[str | None]  # a ReservationStatus value
    notice_pending: NotRequired[bool]  # the decision notice is still undelivered


def _reply(text: str, *, drop: str, **updates) -> Command:
    """End the turn answering `text`. `drop` removes the triggering human message
    (or, with REMOVE_ALL_MESSAGES, the whole conversation) from the history."""
    return Command(update={"messages": [RemoveMessage(id=drop), AIMessage(content=text)], **updates}, goto=END)


def _review_request(description: str) -> HITLRequest:
    """manual_review's interrupt, shaped like HumanInTheLoopMiddleware's so the
    admin API reads and resumes both the same way."""
    return HITLRequest(
        action_requests=[ActionRequest(name=record_reservation.name, args={}, description=description)],
        review_configs=[ReviewConfig(action_name=record_reservation.name, allowed_decisions=["approve", "reject"])],
    )


def _build_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver):
    user_agent = build_user_agent(llm)
    admin_agent = build_admin_agent(llm)

    # -- Node: deterministic handling of the customer's message ------------- #
    async def route_input(
        state: ParkingState,
    ) -> Command[Literal["agent", "book", "__end__"]]:
        question = state["messages"][-1]
        text = str(question.content).strip()

        if state.get("notice_pending"):
            # The decision notice is already the last message (notify_customer);
            # dropping the question makes it this turn's answer.
            return Command(update={"messages": [RemoveMessage(id=question.id)], "notice_pending": False}, goto=END)

        if await run_in_threadpool(is_prompt_injection, text):
            logger.warning("Blocked a suspected prompt injection")
            # Dropped from the history too, so it never reaches the model on a later turn.
            return _reply(
                "I can't process that request. Please ask about parking information or reservations.",
                drop=question.id,
            )

        command = text.casefold()

        if command == "cancel":
            return _reply(
                "The conversation and unsent booking draft were cleared.",
                drop=REMOVE_ALL_MESSAGES,
                draft=None,
            )

        if command == "confirm":
            if state.get("draft") is not None:
                return Command(update={"messages": [RemoveMessage(id=question.id)]}, goto="book")

            previous_id = state.get("last_reservation_id")
            if previous_id:
                return _reply(
                    f"Your previous request was already submitted.\nReference: {previous_id}", drop=question.id
                )
            return _reply("There is no reviewed reservation to confirm.", drop=question.id)

        # Invalidate an old review before processing a correction.
        return Command(update={"draft": None}, goto="agent")

    # -- Node: the user agent answers / prepares a review -------------------- #
    async def agent(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        result = await user_agent.ainvoke(
            {"messages": state["messages"], "draft": None},
            context=runtime.context,
        )
        # The agent's final message list replaces ours: its middleware may have
        # redacted earlier messages or folded them into a summary.
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *result["messages"]],
            "draft": result.get("draft"),
        }

    # -- Node: save the reviewed booking as a pending reservation ----------- #
    async def book(
        state: ParkingState,
        runtime: Runtime[ChatContext],
        config: RunnableConfig,
    ) -> Command[Literal["admin_review", "__end__"]]:
        try:
            reservation_id = await runtime.context.parking_service.book(
                state["draft"], conversation_id=config["configurable"]["thread_id"]
            )
        except ValueError as exc:
            return Command(
                update={
                    "messages": [
                        AIMessage(f"The request cannot be submitted: {exc}. Please provide corrected details.")
                    ],
                    "draft": None,
                },
                goto=END,
            )

        # Start fresh after a submitted booking.
        return Command(
            update={
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
            },
            goto="admin_review",
        )

    async def _facts(state: ParkingState, runtime: Runtime[ChatContext]) -> str:
        facts = await runtime.context.parking_service.approval_facts(UUID(state["last_reservation_id"]))
        return "\n".join(f"{key}: {value}" for key, value in facts.items())

    # -- Node: the human-in-the-loop admin agent ---------------------------- #
    async def admin_review(
        state: ParkingState, runtime: Runtime[ChatContext]
    ) -> Command[Literal["notify_customer", "manual_review"]]:
        # Pauses inside admin_agent (HumanInTheLoopMiddleware) until the admin
        # API resumes the thread; this node then re-runs and the agent picks up
        # where it paused, so its model is not called again for the same request.
        result = await admin_agent.ainvoke(
            {
                "messages": [HumanMessage(f"New reservation request:\n{await _facts(state, runtime)}")],
                "reservation_id": state["last_reservation_id"],
            },
            context=runtime.context,
        )

        reviewed = any(
            isinstance(message, ToolMessage) and message.name == record_reservation.name
            for message in result["messages"]
        )
        if not reviewed:
            logger.warning("Admin agent never proposed %s; asking the administrator directly", record_reservation.name)
            return Command(goto="manual_review")

        # Approved, record_reservation already recorded it; rejected, it never ran.
        if state.get("decision") == ApprovalDecision.REJECTED:
            await runtime.context.parking_service.record_decision(
                UUID(state["last_reservation_id"]), approved=False, note=state.get("admin_note")
            )

        return Command(goto="notify_customer")

    # -- Node: the same pause, without the agent ---------------------------- #
    async def manual_review(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        interrupt(_review_request(f"Please approve or refuse this reservation:\n{await _facts(state, runtime)}"))

        await runtime.context.parking_service.record_decision(
            UUID(state["last_reservation_id"]),
            approved=state.get("decision") == ApprovalDecision.APPROVED,
            note=state.get("admin_note"),
        )
        return {}

    # -- Node: tell the customer the outcome on their next message ---------- #
    async def notify_customer(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        reservation_id = state["last_reservation_id"]
        status = await runtime.context.parking_service.get_status(UUID(reservation_id))

        if status == ReservationStatus.RECORDED:
            outcome = "approved and recorded"
        elif status == ReservationStatus.REJECTED:
            outcome = "rejected by the administrator"
        else:
            outcome = "reviewed, but could not be recorded yet - we will follow up"

        notice = f"Update: your reservation {reservation_id} was {outcome}.\nYou can ask your question again."

        return {
            "messages": [AIMessage(notice)],
            "reservation_status": str(status) if status else None,
            "decision": None,
            "admin_note": None,
            "notice_pending": True,
        }

    builder = StateGraph(ParkingState, context_schema=ChatContext)
    builder.add_node("route_input", route_input)
    builder.add_node("agent", agent)
    builder.add_node("book", book)
    builder.add_node("admin_review", admin_review)
    builder.add_node("manual_review", manual_review)
    builder.add_node("notify_customer", notify_customer)

    builder.add_edge(START, "route_input")
    builder.add_edge("agent", END)
    builder.add_edge("manual_review", "notify_customer")
    builder.add_edge("notify_customer", END)

    return builder.compile(checkpointer=checkpointer)


class ParkingGraph:
    """Built once at startup (app/lifespan.py); called by the API routers."""

    def __init__(self, llm: BaseChatModel, checkpointer: BaseCheckpointSaver, redis: Redis):
        self.graph = _build_graph(llm, checkpointer)
        self.redis = redis

    @staticmethod
    def thread_id(session_id: str) -> str:
        return f"{CHAT_THREAD_PREFIX}{session_id}"

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _lock(self, thread_id: str, blocking_timeout: float):
        # Shared by chat turns and admin resumes: both write the same thread.
        return self.redis.lock(
            f"parking:graph:{thread_id}:lock", timeout=THREAD_LOCK_TIMEOUT_SECONDS, blocking_timeout=blocking_timeout
        )

    async def _waiting_for(self, thread_id: str) -> dict | None:
        """{"reservation_id", "message"} if the thread is paused for the
        administrator's review (HumanInTheLoopMiddleware's or manual_review's
        interrupt - both shaped as an HITLRequest)."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        for task in snapshot.tasks:
            for pending in task.interrupts:
                return {
                    "reservation_id": snapshot.values.get("last_reservation_id"),
                    "message": pending.value["action_requests"][0]["description"],
                }
        return None

    async def chat(self, session_id: str, question: str, context: ChatContext) -> dict:
        """One customer turn. Returns {"status", "answer"}; status is
        "waiting_for_admin" while the conversation is paused for approval."""
        thread_id = self.thread_id(session_id)
        lock = self._lock(thread_id, blocking_timeout=CHAT_LOCK_WAIT_SECONDS)

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
        async with self._lock(thread_id, blocking_timeout=ADMIN_LOCK_WAIT_SECONDS):
            waiting = await self._waiting_for(thread_id)
            if waiting is None or waiting["reservation_id"] != str(reservation_id):
                raise LookupError(reservation_id)

            decision = ApprovalDecision(decision)
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
                config=self._config(thread_id),
                context=context,
                durability="exit",
            )

        return ReservationStatus(result["reservation_status"])


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))
