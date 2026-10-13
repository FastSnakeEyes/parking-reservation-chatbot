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


def _at(hours: int) -> datetime:
    """A moment `hours` after the start of tomorrow's test window."""
    return FUTURE_START + timedelta(hours=hours)


async def _reserve(db_session, parking_id: int, start: int, end: int, status: str) -> None:
    db_session.add(
        Reservation(
            id=uuid4(),
            parking_id=parking_id,
            space_type="standard",
            first_name="Other",
            surname="Customer",
            car_number="XYZ987",
            starts_at=_at(start),
            ends_at=_at(end),
            status=status,
            approved_at=datetime.now(UTC) if status == "approved" else None,
        )
    )
    await db_session.flush()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_availability_counts_active_overlapping_reservations(db_session, sample_parking):
    """sample_parking has 5 bookable standard spaces."""
    await _reserve(db_session, sample_parking.id, 0, 2, "pending")
    await _reserve(db_session, sample_parking.id, 1, 3, "approved")
    await _reserve(db_session, sample_parking.id, 0, 3, "rejected")  # holds no space
    await _reserve(db_session, sample_parking.id, 5, 6, "approved")  # outside the window

    availability = await ReservationRepository(db_session).availability(sample_parking.id, "standard", _at(0), _at(4))

    assert availability == {"bookable_spaces": 5, "reserved": 2, "available": 3}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_back_to_back_reservations_count_once(db_session, sample_parking):
    await _reserve(db_session, sample_parking.id, 0, 1, "approved")
    await _reserve(db_session, sample_parking.id, 1, 2, "approved")  # starts as the first ends

    availability = await ReservationRepository(db_session).availability(sample_parking.id, "standard", _at(0), _at(2))

    assert availability["reserved"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_capacity_override_lowers_availability(db_session, sample_parking):
    from app.db.models import CapacityOverride

    db_session.add(
        CapacityOverride(
            parking_id=sample_parking.id,
            space_type="standard",
            starts_at=_at(0),
            ends_at=_at(24),
            bookable_capacity=1,
            note="maintenance",
        )
    )
    await _reserve(db_session, sample_parking.id, 0, 2, "pending")

    availability = await ReservationRepository(db_session).availability(sample_parking.id, "standard", _at(0), _at(2))

    assert availability == {"bookable_spaces": 1, "reserved": 1, "available": 0}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_submit_pending_refuses_when_no_space_is_free(db_session, sample_parking):
    for _ in range(5):  # fill all 5 bookable spaces for the booking's period
        await _reserve(db_session, sample_parking.id, 0, 2, "approved")

    with pytest.raises(ValueError, match="No spaces"):
        await ReservationRepository(db_session).submit_pending(uuid4(), _booking(sample_parking.id))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_submissions_cannot_both_take_the_last_space(engine, real_parking, monkeypatch):
    """Two customers submit at once for the last free space: the row lock in
    submit_pending lets exactly one through."""
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    # Widen the race window between the capacity check and the insert, so both
    # submissions would pass the check if nothing serialized them.
    check = ReservationRepository.validate_request

    async def slow_check(self, booking, **kwargs):
        result = await check(self, booking, **kwargs)
        await asyncio.sleep(0.3)
        return result

    monkeypatch.setattr(ReservationRepository, "validate_request", slow_check)

    created: list = []
    async with AsyncSession(engine, expire_on_commit=False) as session:
        for _ in range(4):  # real_parking has 5 bookable standard spaces: leave one
            reservation = Reservation(
                id=uuid4(),
                parking_id=real_parking.id,
                space_type="standard",
                first_name="Other",
                surname="Customer",
                car_number="XYZ987",
                starts_at=_at(0),
                ends_at=_at(2),
                status="approved",
                approved_at=datetime.now(UTC),
            )
            session.add(reservation)
            created.append(reservation.id)
        await session.commit()

    async def submit():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            reservation_id = uuid4()
            created.append(reservation_id)
            await ReservationRepository(session).submit_pending(reservation_id, _booking(real_parking.id))

    try:
        results = await asyncio.gather(submit(), submit(), return_exceptions=True)
    finally:
        async with AsyncSession(engine) as session:
            await session.execute(sa.delete(Reservation).where(Reservation.id.in_(created)))
            await session.commit()

    assert sum(result is None for result in results) == 1
    assert [type(result) for result in results if result is not None] == [ValueError]
