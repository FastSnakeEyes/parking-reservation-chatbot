"""ParkingService.record_decision: the decision write, the reservation
status transition, and the resume-command enqueue must commit (or roll
back) as a single unit - never leaving an orphaned decision or a command
enqueued for a decision that didn't actually take effect."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import sqlalchemy as sa
from app.db.models import AdminDecision, Command, Parking, ParkingSpaceType, Reservation
from app.db.repositories.admin_decision import DecisionConflict
from app.db.repositories.reservation import StaleReservation
from app.services.parking_service import ParkingService, ResumeCommand
from sqlalchemy.ext.asyncio import AsyncSession

START = datetime.now(UTC) + timedelta(days=1)
END = START + timedelta(hours=2)


async def _make_reservation(session: AsyncSession) -> Reservation:
    parking = Parking(
        slug=f"service-test-{uuid4()}",
        name="Service Test Parking",
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
async def test_record_decision_commits_decision_status_and_command_together(real_session):
    reservation = await _make_reservation(real_session)
    service = ParkingService(real_session)
    command_key = f"resume-{uuid4()}"

    await service.record_decision(
        reservation.id,
        reservation.version,
        "approved",
        "ok",
        f"decide-{uuid4()}",
        resume_command=ResumeCommand(command_key, "resume_user", "thread-1", {}),
    )

    await real_session.refresh(reservation)
    assert reservation.status == "approved"

    decision_count = await real_session.scalar(
        sa.select(sa.func.count()).select_from(AdminDecision).where(AdminDecision.reservation_id == reservation.id)
    )
    assert decision_count == 1

    command = await real_session.scalar(sa.select(Command).where(Command.idempotency_key == command_key))
    assert command is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_second_decision_conflict_rolls_back_without_orphaned_command(real_session):
    reservation = await _make_reservation(real_session)
    reservation_id, reservation_version = reservation.id, reservation.version
    service = ParkingService(real_session)

    await service.record_decision(reservation_id, reservation_version, "approved", "ok", f"decide-{uuid4()}")

    retry_key = f"resume-{uuid4()}"

    with pytest.raises(DecisionConflict):
        await service.record_decision(
            reservation_id,
            reservation_version,
            "rejected",
            "actually no",
            f"decide-{uuid4()}",
            resume_command=ResumeCommand(retry_key, "resume_user", "thread-1", {}),
        )

    # record_decision's rollback on conflict expires every object tracked by
    # real_session, including `reservation` - query by the id captured
    # earlier rather than touching the now-stale ORM instance.
    decision_count = await real_session.scalar(
        sa.select(sa.func.count()).select_from(AdminDecision).where(AdminDecision.reservation_id == reservation_id)
    )
    assert decision_count == 1

    orphaned_command = await real_session.scalar(sa.select(Command).where(Command.idempotency_key == retry_key))
    assert orphaned_command is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_version_rolls_back_the_already_flushed_decision(real_session):
    """mark_decided fails on the stale version *after* decisions.record()
    already succeeded (flush-only, not committed) - the whole transaction
    must roll back, leaving zero decision rows, not one."""
    reservation = await _make_reservation(real_session)
    reservation_id, reservation_version = reservation.id, reservation.version
    service = ParkingService(real_session)
    command_key = f"resume-{uuid4()}"

    with pytest.raises(StaleReservation):
        await service.record_decision(
            reservation_id,
            reservation_version + 1,
            "approved",
            "ok",
            f"decide-{uuid4()}",
            resume_command=ResumeCommand(command_key, "resume_user", "thread-1", {}),
        )

    # record_decision's rollback expires every object tracked by
    # real_session, including `reservation` - query fresh by id instead of
    # touching the now-stale ORM instance.
    decision_count = await real_session.scalar(
        sa.select(sa.func.count()).select_from(AdminDecision).where(AdminDecision.reservation_id == reservation_id)
    )
    assert decision_count == 0

    orphaned_command = await real_session.scalar(sa.select(Command).where(Command.idempotency_key == command_key))
    assert orphaned_command is None

    refreshed = await real_session.get(Reservation, reservation_id)
    assert refreshed.status == "pending"
