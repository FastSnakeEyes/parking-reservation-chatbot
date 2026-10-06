"""AdminDecisionRepository: the database-level protection against
conflicting administrator decisions on the same reservation (two admins,
or one admin double-clicking approve/reject)."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.db.models import Parking, ParkingSpaceType, Reservation
from app.db.repositories.admin_decision import AdminDecisionRepository, DecisionConflict
from sqlalchemy.ext.asyncio import AsyncSession

START = datetime.now(UTC) + timedelta(days=1)
END = START + timedelta(hours=2)


async def _make_reservation(session: AsyncSession) -> Reservation:
    parking = Parking(
        slug=f"decision-test-{uuid4()}",
        name="Decision Test Parking",
        district="Center",
        latitude=54.68,
        longitude=25.27,
        total_spaces=2,
        bookable_spaces=2,
    )
    session.add(parking)
    await session.flush()

    session.add(
        ParkingSpaceType(
            parking_id=parking.id,
            space_type="standard",
            total_spaces=2,
            bookable_spaces=2,
        )
    )

    reservation = Reservation(
        parking_id=parking.id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=START,
        ends_at=END,
    )
    session.add(reservation)
    await session.commit()

    return reservation


@pytest.mark.integration
@pytest.mark.asyncio
async def test_record_decision_succeeds(real_session):
    reservation = await _make_reservation(real_session)
    repository = AdminDecisionRepository(real_session)

    decision = await repository.record(reservation.id, reservation.version, "approved", "ok", f"decide-{uuid4()}")

    assert decision.reservation_id == reservation.id
    assert decision.decision == "approved"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_second_decision_on_same_reservation_conflicts(real_session):
    reservation = await _make_reservation(real_session)
    repository = AdminDecisionRepository(real_session)

    await repository.record(reservation.id, reservation.version, "approved", "ok", f"decide-{uuid4()}")
    await real_session.commit()

    with pytest.raises(DecisionConflict):
        await repository.record(reservation.id, reservation.version, "rejected", "actually no", f"decide-{uuid4()}")
