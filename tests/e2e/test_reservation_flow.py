"""Full Stage 2/3/4 flow: a booking is submitted, handed to the approval
graph, and the administrator's decision either records it through the real
MCP server (approved) or just closes it out (rejected) - end to end, against
a real Postgres and a real MCP server subprocess.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.ai.graph import ApprovalPipeline
from app.db.models import Reservation
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput

FUTURE_START = datetime.now(UTC) + timedelta(days=1)
FUTURE_END = FUTURE_START + timedelta(hours=3)


def _booking(parking_id: int) -> BookingInput:
    return BookingInput(
        parking_id=parking_id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="E2E-123",
        starts_at=FUTURE_START,
        ends_at=FUTURE_END,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_approval_records_reservation_via_mcp_server(
    real_session,
    real_parking,
    approval_pipeline: ApprovalPipeline,
    mcp_test_server,
):
    reservation_id = uuid4()
    booking = _booking(real_parking.id)
    await ReservationRepository(real_session).submit_pending(reservation_id, booking)

    await approval_pipeline.start(reservation_id, message="Please approve Jane Doe's reservation.")
    assert await approval_pipeline.pending_message(reservation_id) == "Please approve Jane Doe's reservation."

    await approval_pipeline.resume(reservation_id, "approved")

    reservation = await real_session.get(Reservation, reservation_id, populate_existing=True)
    assert reservation.status == "recorded"
    assert reservation.approved_at is not None

    contents = mcp_test_server["file"].read_text()
    assert "Jane Doe | E2E-123" in contents
    assert contents.count("\n") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejection_closes_without_calling_mcp_server(
    real_session,
    real_parking,
    approval_pipeline: ApprovalPipeline,
    mcp_test_server,
):
    reservation_id = uuid4()
    booking = _booking(real_parking.id)
    await ReservationRepository(real_session).submit_pending(reservation_id, booking)

    before = mcp_test_server["file"].read_text() if mcp_test_server["file"].exists() else ""

    await approval_pipeline.start(reservation_id, message="Please approve Jane Doe's reservation.")
    await approval_pipeline.resume(reservation_id, "rejected")

    reservation = await real_session.get(Reservation, reservation_id, populate_existing=True)
    assert reservation.status == "rejected"
    assert reservation.approved_at is None

    after = mcp_test_server["file"].read_text() if mcp_test_server["file"].exists() else ""
    assert after == before
