"""ParkingService.record_decision (app/services/parking.py) against real
Postgres: an approval is recorded through MCP exactly once, however many
decisions arrive at the same moment."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import sqlalchemy as sa
from app.db.models import Reservation
from app.db.repositories.parking import ParkingRepository
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput
from app.services import parking as parking_module
from app.services.parking import ParkingService
from sqlalchemy.ext.asyncio import AsyncSession


class SlowRecordTool:
    """Stands in for the MCP tool; slow enough that concurrent decisions overlap."""

    name = "record_reservation"

    def __init__(self):
        self.calls: list[dict] = []

    async def ainvoke(self, arguments):
        await asyncio.sleep(0.3)
        self.calls.append(arguments)


@pytest.fixture
def mcp_tool(monkeypatch):
    tool = SlowRecordTool()

    class Client:
        async def get_tools(self):
            return [tool]

    monkeypatch.setattr(parking_module, "_mcp_client", Client)
    return tool


async def _pending_reservation(engine, parking_id: int):
    starts_at = datetime.now(UTC) + timedelta(days=3)
    reservation_id = uuid4()
    async with AsyncSession(engine, expire_on_commit=False) as session:
        await ReservationRepository(session).submit_pending(
            reservation_id,
            BookingInput(
                parking_id=parking_id,
                space_type="standard",
                first_name="Jane",
                surname="Doe",
                car_number="ABC123",
                starts_at=starts_at,
                ends_at=starts_at + timedelta(hours=1),
            ),
        )
    return reservation_id


async def _decide(engine, reservation_id, *, approved: bool):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        service = ParkingService(ParkingRepository(session), ReservationRepository(session))
        return await service.record_decision(reservation_id, approved=approved)


async def _delete(engine, reservation_id):
    async with AsyncSession(engine) as session:
        await session.execute(sa.delete(Reservation).where(Reservation.id == reservation_id))
        await session.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_approvals_record_through_mcp_once(engine, real_parking, mcp_tool):
    """A double-click in the admin console: two approvals at the same moment."""
    reservation_id = await _pending_reservation(engine, real_parking.id)

    try:
        results = await asyncio.gather(
            _decide(engine, reservation_id, approved=True), _decide(engine, reservation_id, approved=True)
        )
    finally:
        await _delete(engine, reservation_id)

    assert len(mcp_tool.calls) == 1
    assert results == ["approved", "approved"]  # the second just reports the outcome


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_decision_on_a_decided_reservation_changes_nothing(engine, real_parking, mcp_tool):
    reservation_id = await _pending_reservation(engine, real_parking.id)

    try:
        first = await _decide(engine, reservation_id, approved=False)
        second = await _decide(engine, reservation_id, approved=True)  # too late: already rejected
    finally:
        await _delete(engine, reservation_id)

    assert (first, second) == ("rejected", "rejected")
    assert mcp_tool.calls == []
