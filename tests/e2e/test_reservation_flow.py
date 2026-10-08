"""Full Stage 1-4 flow through ParkingGraph: the customer gets a review,
confirms, the conversation pauses for the administrator, and the decision
either records the reservation through the real MCP server (approved) or
just closes it out (rejected) - end to end, against a real Postgres and a
real MCP server subprocess, with the app's Postgres checkpointer. Only the
LLM is scripted.
"""

from datetime import UTC, datetime, timedelta

import pytest
from app.ai.graph import ParkingGraph
from app.ai.tools import ChatContext
from app.db.models import Reservation
from app.db.repositories.parking import ParkingRepository
from app.db.repositories.reservation import ReservationRepository
from app.services.parking import ParkingService
from langchain_core.messages import AIMessage
from sqlalchemy.ext.asyncio import AsyncSession
from tests.e2e.conftest import ADMIN_DRAFT_MESSAGE, ADMIN_TURN, ScriptedModel

FUTURE_START = datetime.now(UTC) + timedelta(days=1)
FUTURE_END = FUTURE_START + timedelta(hours=3)
SESSION = "e2e-session"


def _prepare_call(parking_id: int) -> AIMessage:
    args = {
        "parking_id": parking_id,
        "space_type": "standard",
        "first_name": "Jane",
        "surname": "Doe",
        "car_number": "E2E-123",
        "starts_at": FUTURE_START.isoformat(),
        "ends_at": FUTURE_END.isoformat(),
    }
    return AIMessage(content="", tool_calls=[{"name": "prepare_reservation", "args": args, "id": "c1"}])


def _context(session: AsyncSession) -> ChatContext:
    return ChatContext(parking_service=ParkingService(ParkingRepository(session), ReservationRepository(session)))


async def _submit(engine, checkpointer, real_parking, lock_redis) -> tuple[ParkingGraph, Reservation]:
    """Customer turns: review, then confirm - the conversation pauses for the admin."""
    parking_graph = ParkingGraph(
        ScriptedModel(responses=[_prepare_call(real_parking.id), *ADMIN_TURN]), checkpointer, lock_redis
    )

    async with AsyncSession(engine, expire_on_commit=False) as session:
        review = await parking_graph.chat(SESSION, "book it", _context(session))
        assert review["answer"].startswith("Please review your reservation request:")

        confirmed = await parking_graph.chat(SESSION, "confirm", _context(session))
        assert confirmed["status"] == "waiting_for_admin"

        reservations = await ReservationRepository(session).list_pending()
        reservation = next(r for r in reservations if r.conversation_id == parking_graph.thread_id(SESSION))

    assert await parking_graph.pending_message(reservation.conversation_id) == ADMIN_DRAFT_MESSAGE
    return parking_graph, reservation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_approval_records_reservation_via_mcp_server(
    engine, checkpointer, real_parking, mcp_settings, lock_redis
):
    parking_graph, reservation = await _submit(engine, checkpointer, real_parking, lock_redis)

    # Admin request: its own DB session, as in the app.
    async with AsyncSession(engine, expire_on_commit=False) as session:
        status = await parking_graph.resume(
            reservation.conversation_id, reservation.id, "approved", None, _context(session)
        )
        stored = await session.get(Reservation, reservation.id, populate_existing=True)

    assert status == "recorded"
    assert stored.status == "recorded"
    assert stored.approved_at is not None

    contents = mcp_settings["file"].read_text()
    assert "Jane Doe | E2E-123" in contents

    async with AsyncSession(engine, expire_on_commit=False) as session:
        notice = await parking_graph.chat(SESSION, "any news?", _context(session))

    assert notice["answer"].startswith(f"Update: your reservation {reservation.id} was approved")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejection_closes_without_calling_mcp_server(
    engine, checkpointer, real_parking, mcp_settings, lock_redis
):
    parking_graph, reservation = await _submit(engine, checkpointer, real_parking, lock_redis)

    before = mcp_settings["file"].read_text() if mcp_settings["file"].exists() else ""

    async with AsyncSession(engine, expire_on_commit=False) as session:
        status = await parking_graph.resume(
            reservation.conversation_id, reservation.id, "rejected", None, _context(session)
        )
        stored = await session.get(Reservation, reservation.id, populate_existing=True)

    assert status == "rejected"
    assert stored.status == "rejected"
    assert stored.approved_at is None

    after = mcp_settings["file"].read_text() if mcp_settings["file"].exists() else ""
    assert after == before
