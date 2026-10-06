"""CapacityAllocationRepository: the database-level protection against
overselling bookable capacity (see its module docstring for why a plain
Postgres EXCLUDE constraint is the wrong tool for N-capacity booking)."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from app.db.models import Parking, ParkingSpaceType, Reservation
from app.db.repositories.capacity_allocation import CapacityAllocationRepository
from sqlalchemy.ext.asyncio import AsyncSession

START = datetime.now(UTC) + timedelta(days=1)
END = START + timedelta(hours=2)


async def _make_parking(session: AsyncSession, *, bookable_spaces: int) -> Parking:
    parking = Parking(
        slug=f"cap-test-{uuid4()}",
        name="Capacity Test Parking",
        district="Center",
        latitude=54.68,
        longitude=25.27,
        total_spaces=bookable_spaces,
        bookable_spaces=bookable_spaces,
    )
    session.add(parking)
    await session.flush()

    session.add(
        ParkingSpaceType(
            parking_id=parking.id,
            space_type="standard",
            total_spaces=bookable_spaces,
            bookable_spaces=bookable_spaces,
        )
    )
    await session.commit()

    return parking


async def _make_reservation(
    session: AsyncSession,
    parking_id: int,
    starts_at: datetime = START,
    ends_at: datetime = END,
) -> UUID:
    """capacity_allocations.reservation_id has a real FK to reservations,
    so allocation tests need an actual (pending) reservation row to
    reference, not just a bare uuid4()."""
    reservation = Reservation(
        parking_id=parking_id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=starts_at,
        ends_at=ends_at,
    )
    session.add(reservation)
    await session.commit()

    return reservation.id


async def _allocate_and_settle(
    engine,
    parking_id: int,
    reservation_id: UUID,
) -> bool:
    """Open its own session (its own connection), allocate, and
    commit/rollback accordingly - mirroring how ParkingService.allocate_capacity
    actually uses CapacityAllocationRepository.allocate (flush-only; caller
    decides commit vs rollback)."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        repository = CapacityAllocationRepository(session)
        allocated = await repository.allocate(parking_id, "standard", reservation_id, START, END)

        if allocated:
            await session.commit()
        else:
            await session.rollback()

        return allocated


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allocate_up_to_capacity_then_next_fails(real_session):
    parking = await _make_parking(real_session, bookable_spaces=2)
    repository = CapacityAllocationRepository(real_session)

    first_id = await _make_reservation(real_session, parking.id)
    second_id = await _make_reservation(real_session, parking.id)
    third_id = await _make_reservation(real_session, parking.id)

    first = await repository.allocate(parking.id, "standard", first_id, START, END)
    second = await repository.allocate(parking.id, "standard", second_id, START, END)
    third = await repository.allocate(parking.id, "standard", third_id, START, END)

    assert (first, second, third) == (True, True, False)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_allocate_unknown_space_type_fails_cleanly(real_session):
    parking = await _make_parking(real_session, bookable_spaces=2)
    repository = CapacityAllocationRepository(real_session)
    reservation_id = await _make_reservation(real_session, parking.id)

    allocated = await repository.allocate(parking.id, "ev", reservation_id, START, END)

    assert allocated is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_overlapping_periods_do_not_compete_for_capacity(real_session):
    parking = await _make_parking(real_session, bookable_spaces=1)
    repository = CapacityAllocationRepository(real_session)

    later_start = END + timedelta(hours=1)
    later_end = later_start + timedelta(hours=2)

    first_id = await _make_reservation(real_session, parking.id, START, END)
    second_id = await _make_reservation(real_session, parking.id, later_start, later_end)

    first = await repository.allocate(parking.id, "standard", first_id, START, END)
    second = await repository.allocate(parking.id, "standard", second_id, later_start, later_end)

    assert first is True
    assert second is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_requests_for_the_last_space(engine, real_session):
    parking = await _make_parking(real_session, bookable_spaces=1)

    first_id = await _make_reservation(real_session, parking.id)
    second_id = await _make_reservation(real_session, parking.id)

    results = await asyncio.gather(
        _allocate_and_settle(engine, parking.id, first_id),
        _allocate_and_settle(engine, parking.id, second_id),
    )

    # The row lock in CapacityAllocationRepository.allocate serializes the
    # two attempts: exactly one sees capacity available, the other sees it
    # already taken once its own lock is granted. Neither oversells.
    assert sorted(results) == [False, True]
