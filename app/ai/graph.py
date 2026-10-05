"""Stage 4: LangGraph orchestration.

Two graphs, not one monolithic synchronous graph, because the admin
approval step is asynchronous and can take hours:

- `run_chat_turn` wraps ParkingChat's existing tool-calling loop in a
  trivial 1-node StateGraph, so "user interaction" is an explicit
  LangGraph node (per the spec's example structure) rather than a bare
  method call. The loop itself is kept as-is; this does not change its
  behavior.
- `ApprovalPipeline` is the durable admin-approval + recording graph.
  `admin_approval` calls `interrupt()` and pauses - persisted by a
  sqlite checkpointer, since the wait for a human decision can take
  hours or days - and `start()` returns as soon as that happens.
  `record_or_close` resumes via `Command(resume=...)` from the admin
  decision endpoint (app/api/routers/admin.py): on approval it calls
  the Stage 3 MCP tool and marks the reservation "recorded"; on
  rejection it just marks it "rejected".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, TypedDict
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import settings
from app.db.repositories.reservation import ReservationRepository


class ChatState(TypedDict):
    session_id: str
    question: str
    result: dict


async def run_chat_turn(chat, session_id: str, question: str) -> dict:
    """Run one user-interaction turn through a 1-node LangGraph graph."""

    async def user_interaction(state: ChatState) -> dict:
        return {"result": await chat.ask(state["session_id"], state["question"])}

    workflow = StateGraph(ChatState)
    workflow.add_node("user_interaction", user_interaction)
    workflow.add_edge(START, "user_interaction")
    workflow.add_edge("user_interaction", END)

    final_state = await workflow.compile().ainvoke({"session_id": session_id, "question": question, "result": {}})

    return final_state["result"]


class ApprovalState(TypedDict):
    reservation_id: str
    message: str
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


async def admin_approval(state: ApprovalState) -> dict:
    decision = interrupt(
        {
            "reservation_id": state["reservation_id"],
            "message": state["message"],
        }
    )

    return {"decision": decision["decision"], "note": decision.get("note")}


def _build_record_or_close(engine: AsyncEngine):
    local_timezone = ZoneInfo(settings.timezone)

    async def record_or_close(state: ApprovalState) -> dict:
        reservation_id = UUID(state["reservation_id"])

        async with AsyncSession(engine, expire_on_commit=False) as session:
            reservations = ReservationRepository(session)
            reservation = await reservations.get_by_id(reservation_id)

            if reservation is None or reservation.status != "pending":
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

                await reservations.update(
                    reservation_id,
                    {"status": "recorded", "approved_at": approved_at},
                )
            else:
                await reservations.update(reservation_id, {"status": "rejected"})

            await session.commit()

        return {}

    return record_or_close


def _build_approval_graph(engine: AsyncEngine, checkpointer: BaseCheckpointSaver):
    workflow = StateGraph(ApprovalState)
    workflow.add_node("admin_approval", admin_approval)
    workflow.add_node("record_or_close", _build_record_or_close(engine))
    workflow.add_edge(START, "admin_approval")
    workflow.add_edge("admin_approval", "record_or_close")
    workflow.add_edge("record_or_close", END)

    return workflow.compile(checkpointer=checkpointer)


class ApprovalPipeline:
    def __init__(self, engine: AsyncEngine, checkpointer: BaseCheckpointSaver):
        self.graph = _build_approval_graph(engine, checkpointer)

    @staticmethod
    def _config(reservation_id: UUID) -> dict:
        return {"configurable": {"thread_id": str(reservation_id)}}

    async def start(self, reservation_id: UUID, message: str) -> None:
        """Start the approval graph; returns once it pauses at admin_approval."""
        await self.graph.ainvoke(
            {
                "reservation_id": str(reservation_id),
                "message": message,
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
        await self.graph.ainvoke(
            Command(resume={"decision": decision, "note": note}),
            config=self._config(reservation_id),
        )
