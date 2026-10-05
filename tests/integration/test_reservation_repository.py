from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import sqlalchemy as sa
from app.db.models import Reservation
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput

FUTURE_START = datetime.now(UTC) + timedelta(days=1)
FUTURE_END = FUTURE_START + timedelta(hours=2)


def _booking(parking_id: int) -> BookingInput:
    return BookingInput(
        parking_id=parking_id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=FUTURE_START,
        ends_at=FUTURE_END,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_submit_pending_creates_a_pending_reservation(db_session, sample_parking):
    repository = ReservationRepository(db_session)
    reservation_id = uuid4()

    result_id = await repository.submit_pending(reservation_id, _booking(sample_parking.id))

    assert result_id == reservation_id

    saved = await db_session.get(Reservation, reservation_id)
    assert saved.status == "pending"
    assert saved.car_number == "ABC123"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_submit_pending_is_idempotent_on_retry(db_session, sample_parking):
    repository = ReservationRepository(db_session)
    reservation_id = uuid4()
    booking = _booking(sample_parking.id)

    first = await repository.submit_pending(reservation_id, booking)
    second = await repository.submit_pending(reservation_id, booking)

    assert first == second

    count = await db_session.scalar(
        sa.select(sa.func.count()).select_from(Reservation).where(Reservation.id == reservation_id)
    )
    assert count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_request_rejects_past_start(db_session, sample_parking):
    repository = ReservationRepository(db_session)
    booking = BookingInput(
        parking_id=sample_parking.id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=datetime.now(UTC) - timedelta(days=1),
        ends_at=datetime.now(UTC) + timedelta(hours=1),
    )

    with pytest.raises(ValueError, match="future"):
        await repository.validate_request(booking)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_request_rejects_unknown_parking(db_session):
    repository = ReservationRepository(db_session)

    with pytest.raises(ValueError, match="not found"):
        await repository.validate_request(_booking(parking_id=999999))
