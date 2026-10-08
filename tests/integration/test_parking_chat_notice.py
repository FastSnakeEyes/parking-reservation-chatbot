"""The customer-facing notice app/ai/middleware.py's decision_notice delivers
once the administrator has decided on their last reservation - otherwise
the customer has no way to learn the outcome."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.ai.middleware import decision_notice
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput


def _booking(parking_id: int) -> BookingInput:
    return BookingInput(
        parking_id=parking_id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=datetime.now(UTC) + timedelta(days=1),
        ends_at=datetime.now(UTC) + timedelta(days=1, hours=2),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_notice_without_a_last_reservation(db_session):
    assert await decision_notice(ReservationRepository(db_session), None) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_notice_while_still_pending(db_session, sample_parking):
    reservation_id = uuid4()
    reservations = ReservationRepository(db_session)
    await reservations.submit_pending(reservation_id, _booking(sample_parking.id))

    assert await decision_notice(reservations, str(reservation_id)) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notice_on_approval(db_session, sample_parking):
    reservation_id = uuid4()
    reservations = ReservationRepository(db_session)
    await reservations.submit_pending(reservation_id, _booking(sample_parking.id))
    await reservations.update(
        reservation_id,
        {"status": "recorded", "approved_at": datetime.now(UTC)},
    )

    notice = await decision_notice(reservations, str(reservation_id))

    assert notice is not None
    assert "approved" in notice
    assert str(reservation_id) in notice


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notice_on_rejection(db_session, sample_parking):
    reservation_id = uuid4()
    reservations = ReservationRepository(db_session)
    await reservations.submit_pending(reservation_id, _booking(sample_parking.id))
    await reservations.update(reservation_id, {"status": "rejected"})

    notice = await decision_notice(reservations, str(reservation_id))

    assert notice is not None
    assert "rejected" in notice
