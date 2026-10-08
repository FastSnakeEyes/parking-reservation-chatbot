"""LangGraph orchestration of the Stage 2 admin-approval pipeline.

request_approval -> admin_decision (pauses for the admin) -> record_reservation

request_approval drafts the administrator's notification using the admin
agent (app/ai/agents.py); admin_decision calls interrupt() and pauses until
app/api/routers/admin.py resumes it with Command(resume=...); on approval,
record_reservation calls the Stage 3 MCP tool and marks the reservation
"recorded", otherwise it marks it "rejected".

Keyed by reservation_id, not the customer's chat session: the REST admin
endpoint only ever has a reservation_id to resume by, and an approval can
take hours or days, so this pipeline has to survive independently of the
customer's session. See app/ai/parking_chat.py's ParkingChat for the
separate, session-scoped conversation loop that calls start() below.
"""

from datetime import datetime
from typing import Literal, TypedDict
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain.messages import HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.ai.agents import build_admin_agent, notify_admin, parse_admin_reply
from app.config import settings
from app.db.repositories.reservation import ReservationRepository
from app.logger import get_logger

logger = get_logger(__name__)


class ApprovalState(TypedDict):
    reservation_id: str
    reservation: dict  # facts for the admin message: name, car_number, parking, period, ...
    message: str | None  # drafted by request_approval; read back by pending_message()
    decision: Literal["approved", "rejected"] | None
    note: str | None


def _mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "parking": {
                "transport": "streamable_http",
                "url": settings.mcp_server_url,
                "headers": {"Authorization": f"Bearer {settings.mcp_api_key.get_secret_value()}"},
            }
        }
    )


def _build_request_approval(llm: BaseChatModel):
    admin_agent = build_admin_agent(llm)

    # -- Node: the admin agent drafts the approval-request message ----------- #
    async def request_approval(state: ApprovalState) -> dict:
        request = "\n".join(f"{key}: {value}" for key, value in state["reservation"].items())
        result = await admin_agent.ainvoke({"messages": [HumanMessage(f"New reservation request:\n{request}")]})

        for message in result["messages"]:
            for call in getattr(message, "tool_calls", None) or []:
                if call["name"] == notify_admin.name:
                    return {"message": call["args"]["text"]}

        # The model skipped the tool: fall back to a plain request rather
        # than leaving the administrator with nothing to decide on.
        return {"message": f"Please approve or refuse this reservation:\n{request}"}

    return request_approval


def _build_admin_decision(llm: BaseChatModel):
    # -- Node: wait for the administrator (human-in-the-loop) ---------------- #
    def admin_decision(state: ApprovalState) -> dict:
        # The graph stops here until it is resumed with Command(resume=...)
        # from app/api/routers/admin.py. Resume value: {"decision": "approved"
        # | "rejected", "note": str | None} - or, for a future free-text
        # channel (email/messenger), the administrator's raw reply string.
        reply = interrupt({"reservation_id": state["reservation_id"], "message": state["message"]})

        if isinstance(reply, dict):
            decision, note = reply["decision"], reply.get("note")
        else:
            parsed = parse_admin_reply(llm, str(reply))
            decision = "approved" if parsed.get("approved") else "rejected"
            note = parsed.get("reason") or None

        return {"decision": decision, "note": note}

    return admin_decision


def _build_record_reservation(engine: AsyncEngine):
    local_timezone = ZoneInfo(settings.timezone)

    # -- Node: persist the decision, recording approvals through MCP -------- #
    async def record_reservation(state: ApprovalState) -> dict:
        reservation_id = UUID(state["reservation_id"])

        async with AsyncSession(engine, expire_on_commit=False) as session:
            reservations = ReservationRepository(session)
            reservation = await reservations.get_by_id(reservation_id)

            if reservation is None or reservation.status != "pending":
                logger.warning(
                    "record_reservation: reservation %s not pending (status=%s), skipping",
                    reservation_id,
                    reservation.status if reservation else "missing",
                )
                return {}

            if state["decision"] == "approved":
                approved_at = datetime.now(local_timezone)
                period = (
                    f"{reservation.starts_at.astimezone(local_timezone).isoformat()} - "
                    f"{reservation.ends_at.astimezone(local_timezone).isoformat()}"
                )

                client = _mcp_client()
                tools = {tool.name: tool for tool in await client.get_tools()}
                await tools["record_reservation"].ainvoke(
                    {
                        "name": f"{reservation.first_name} {reservation.surname}",
                        "car_number": reservation.car_number,
                        "period": period,
                        "approval_time": approved_at.isoformat(),
                    }
                )

                await reservations.update(reservation_id, {"status": "recorded", "approved_at": approved_at})
                logger.info("Reservation %s approved and recorded via MCP", reservation_id)
            else:
                await reservations.update(reservation_id, {"status": "rejected"})
                logger.info("Reservation %s rejected (note=%s)", reservation_id, state.get("note"))

            await session.commit()

        return {}

    return record_reservation


def _build_approval_graph(llm: BaseChatModel, engine: AsyncEngine, checkpointer: BaseCheckpointSaver):
    builder = StateGraph(ApprovalState)
    builder.add_node("request_approval", _build_request_approval(llm))
    builder.add_node("admin_decision", _build_admin_decision(llm))
    builder.add_node("record_reservation", _build_record_reservation(engine))

    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "admin_decision")
    builder.add_edge("admin_decision", "record_reservation")
    builder.add_edge("record_reservation", END)

    return builder.compile(checkpointer=checkpointer)


class ApprovalPipeline:
    def __init__(self, llm: BaseChatModel, engine: AsyncEngine, checkpointer: BaseCheckpointSaver):
        self.graph = _build_approval_graph(llm, engine, checkpointer)

    @staticmethod
    def _config(reservation_id: UUID) -> dict:
        return {"configurable": {"thread_id": str(reservation_id)}}

    async def start(self, reservation_id: UUID, reservation: dict) -> None:
        """Start the approval graph; returns once it pauses at admin_decision.
        `reservation` is the facts dict request_approval drafts the admin's
        notification from (e.g. Reservation ID, Parking, Customer, Period)."""
        logger.info("Starting approval graph for reservation %s", reservation_id)
        await self.graph.ainvoke(
            {
                "reservation_id": str(reservation_id),
                "reservation": reservation,
                "message": None,
                "decision": None,
                "note": None,
            },
            config=self._config(reservation_id),
        )

    async def resume(
        self,
        reservation_id: UUID,
        decision: Literal["approved", "rejected"],
        note: str | None = None,
    ) -> None:
        """Resume a paused approval graph with the administrator's decision."""
        logger.info("Resuming approval graph for reservation %s with decision=%s", reservation_id, decision)
        await self.graph.ainvoke(
            Command(resume={"decision": decision, "note": note}),
            config=self._config(reservation_id),
        )

    async def pending_message(self, reservation_id: UUID) -> str | None:
        """The drafted notification for a reservation paused at admin_decision.

        Reads it back from the already-captured interrupt rather than
        drafting it again on every admin console poll.
        """
        snapshot = await self.graph.aget_state(self._config(reservation_id))

        for task in snapshot.tasks:
            for pending in task.interrupts:
                return pending.value["message"]

        return None
