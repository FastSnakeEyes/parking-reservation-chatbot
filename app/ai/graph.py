"""ParkingGraph: the whole customer conversation as one LangGraph workflow,
one checkpointer thread per chat session ("chat:{session_id}").

    START -> route_input -+-> agent ------------------------------------> END
                          +-> book -> request_approval -> admin_decision -> record_decision -> END
                          +-> END   (notice / injection / cancel / nothing to confirm)

- route_input: deterministic, model-free handling of the turn - the admin's
  decision notice, prompt injection, `cancel`, `confirm` (the model cannot
  authorize a DB write). Everything else goes to the user agent.
- agent: the user agent (app/ai/agents.py) answers or prepares a review.
- book -> request_approval -> admin_decision: on `confirm`, the reservation is
  saved, the admin agent drafts the administrator's notification, and the
  thread pauses at interrupt() until the admin API resumes it - so while a
  reservation awaits approval, the conversation waits too.
- record_decision: records the decision (MCP on approval) and leaves the
  customer a notice for their next message.

The graph calls ParkingService (app/services/parking.py) for every DB/MCP
side effect; the service never calls the graph. Per-request dependencies
arrive as runtime context (ChatContext).
"""

from typing import Annotated, Literal, NotRequired, TypedDict
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from redis.asyncio import Redis
from starlette.concurrency import run_in_threadpool

from app.ai.agents import build_admin_agent, build_user_agent, notify_admin, parse_admin_reply
from app.ai.guardrails import is_prompt_injection
from app.ai.tools import ChatContext
from app.logger import get_logger

logger = get_logger(__name__)

# Prefix of chat threads in the shared checkpointer (app/ai/chat_cleanup.py finds them by it).
CHAT_THREAD_PREFIX = "chat:"


class ParkingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: NotRequired[dict | None]  # set by prepare_reservation, consumed by `confirm`
    last_reservation_id: NotRequired[str | None]
    admin_message: NotRequired[str | None]  # the admin agent's drafted notification
    decision: NotRequired[Literal["approved", "rejected"] | None]
    admin_note: NotRequired[str | None]
    reservation_status: NotRequired[str | None]  # "pending" | "recorded" | "rejected"
    notice_pending: NotRequired[bool]  # the decision notice is still undelivered


def _reply(text: str, *, drop: str, **updates) -> Command:
    """End the turn answering `text`. `drop` removes the triggering human message
    (or, with REMOVE_ALL_MESSAGES, the whole conversation) from the history."""
    return Command(update={"messages": [RemoveMessage(id=drop), AIMessage(content=text)], **updates}, goto=END)


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
            # The decision notice is already the last message (record_decision);
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
    ) -> Command[Literal["request_approval", "__end__"]]:
        try:
            reservation_id = await runtime.context.service.book(
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
                "reservation_status": "pending",
            },
            goto="request_approval",
        )

    # -- Node: the admin agent drafts the approval-request message ---------- #
    async def request_approval(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        facts = await runtime.context.service.approval_facts(UUID(state["last_reservation_id"]))
        request = "\n".join(f"{key}: {value}" for key, value in facts.items())
        result = await admin_agent.ainvoke({"messages": [HumanMessage(f"New reservation request:\n{request}")]})

        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                if call["name"] == notify_admin.name:
                    return {"admin_message": call["args"]["text"]}

        # The model skipped the tool: fall back to a plain request rather
        # than leaving the administrator with nothing to decide on.
        return {"admin_message": f"Please approve or refuse this reservation:\n{request}"}

    # -- Node: wait for the administrator (human-in-the-loop) --------------- #
    def admin_decision(state: ParkingState) -> Command[Literal["record_decision"]]:
        # Pauses here until ParkingGraph.resume() - called by the admin API.
        # Resume value: {"decision": "approved" | "rejected", "note": str | None},
        # or, for a future free-text channel, the administrator's raw reply.
        reply = interrupt({"reservation_id": state["last_reservation_id"], "message": state["admin_message"]})

        if isinstance(reply, dict):
            decision, note = reply["decision"], reply.get("note")
        else:
            parsed = parse_admin_reply(llm, str(reply))
            decision = "approved" if parsed.get("approved") else "rejected"
            note = parsed.get("reason") or None

        return Command(goto="record_decision", update={"decision": decision, "admin_note": note})

    # -- Node: persist the decision, recording approvals through MCP ------- #
    async def record_decision(state: ParkingState, runtime: Runtime[ChatContext]) -> dict:
        reservation_id = state["last_reservation_id"]
        status = await runtime.context.service.record_decision(
            UUID(reservation_id),
            approved=state["decision"] == "approved",
            note=state.get("admin_note"),
        )

        outcome = "approved and recorded" if status == "recorded" else "rejected by the administrator"
        notice = f"Update: your reservation {reservation_id} was {outcome}.\nYou can ask your question again."

        return {
            "messages": [AIMessage(notice)],
            "reservation_status": status,
            "admin_message": None,
            "decision": None,
            "admin_note": None,
            "notice_pending": True,
        }

    builder = StateGraph(ParkingState, context_schema=ChatContext)
    builder.add_node("route_input", route_input)
    builder.add_node("agent", agent)
    builder.add_node("book", book)
    builder.add_node("request_approval", request_approval)
    builder.add_node("admin_decision", admin_decision)
    builder.add_node("record_decision", record_decision)

    builder.add_edge(START, "route_input")
    builder.add_edge("agent", END)
    builder.add_edge("request_approval", "admin_decision")
    builder.add_edge("record_decision", END)

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
        return self.redis.lock(f"parking:graph:{thread_id}:lock", timeout=180, blocking_timeout=blocking_timeout)

    async def _waiting_for(self, thread_id: str) -> dict | None:
        """The pending admin_decision interrupt's payload, if the thread is paused there."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        for task in snapshot.tasks:
            for pending in task.interrupts:
                return pending.value
        return None

    async def chat(self, session_id: str, question: str, context: ChatContext) -> dict:
        """One customer turn. Returns {"status", "answer"}; status is
        "waiting_for_admin" while the conversation is paused for approval."""
        thread_id = self.thread_id(session_id)
        lock = self._lock(thread_id, blocking_timeout=1)

        if not await lock.acquire():
            return {"status": "complete", "answer": "Your previous message is still being processed."}

        try:
            # New input would start a fresh run and abandon the paused one, so
            # a waiting conversation only answers with its status.
            waiting = await self._waiting_for(thread_id)
            if waiting is not None:
                return {
                    "status": "waiting_for_admin",
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
                    "status": "complete",
                    "answer": "I could not finish processing that request. Please try a more specific question.",
                }
        finally:
            await lock.release()

        return {
            "status": "waiting_for_admin" if result.get("__interrupt__") else "complete",
            "answer": _text(result["messages"][-1].content),
        }

    async def pending_message(self, thread_id: str) -> str | None:
        """The administrator's drafted notification, read back from the paused
        interrupt rather than drafted again on every admin console poll."""
        waiting = await self._waiting_for(thread_id)
        return waiting["message"] if waiting else None

    async def resume(
        self,
        thread_id: str,
        reservation_id: UUID,
        decision: Literal["approved", "rejected"],
        note: str | None,
        context: ChatContext,
    ) -> str:
        """Resume a conversation paused for the administrator; returns the
        reservation's resulting status. Raises LookupError if that thread is
        not waiting on this reservation."""
        async with self._lock(thread_id, blocking_timeout=10):
            waiting = await self._waiting_for(thread_id)
            if waiting is None or waiting["reservation_id"] != str(reservation_id):
                raise LookupError(reservation_id)

            logger.info("Resuming conversation for reservation %s with decision=%s", reservation_id, decision)
            result = await self.graph.ainvoke(
                Command(resume={"decision": decision, "note": note}),
                config=self._config(thread_id),
                context=context,
                durability="exit",
            )

        return result["reservation_status"]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict))
