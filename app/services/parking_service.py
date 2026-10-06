"""ParkingService: reservation rules, ownership, administrator scope,
submission, decision recording, and atomic capacity allocation.

No FastAPI imports and no HTTP request objects here - this is called by
GraphRunner / graph nodes, never by a route handler directly.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.repositories.admin_decision import AdminDecisionRepository, DecisionConflict
from app.db.repositories.capacity_allocation import CapacityAllocationRepository
from app.db.repositories.command import CommandRepository
from app.db.repositories.reservation import ReservationRepository, StaleReservation
from app.schemas.chat import BookingInput

__all__ = ["DecisionConflict", "ParkingService", "ResumeCommand", "StaleReservation"]


@dataclass
class ResumeCommand:
    """A command to enqueue in the same transaction as the decision it
    results from - see ParkingService.record_decision."""

    idempotency_key: str
    command_type: str
    thread_id: str
    payload: dict


class ParkingService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.reservations = ReservationRepository(session)
        self.allocations = CapacityAllocationRepository(session)
        self.decisions = AdminDecisionRepository(session)
        self.timezone = ZoneInfo(settings.timezone)

    async def submit_reservation(self, reservation_id: UUID, booking: BookingInput) -> UUID:
        """Idempotent: a retried submission with the same reservation_id
        returns the existing reservation instead of erroring or duplicating."""
        return await self.reservations.submit_pending(reservation_id, booking)

    async def record_decision(
        self,
        reservation_id: UUID,
        expected_version: int,
        decision: str,
        note: str | None,
        idempotency_key: str,
        resume_command: ResumeCommand | None = None,
    ) -> None:
        """Atomically: record the admin decision, transition the
        reservation's status (pending -> approved/rejected), and - if given -
        enqueue the resume command, all in one transaction. Raises
        DecisionConflict if this reservation already has a decision, or
        StaleReservation if it isn't pending at expected_version; either way
        nothing partial is left committed.
        """
        new_status = "approved" if decision == "approved" else "rejected"
        approved_at = datetime.now(UTC) if decision == "approved" else None

        try:
            await self.decisions.record(reservation_id, expected_version, decision, note, idempotency_key)
            await self.reservations.mark_decided(reservation_id, expected_version, new_status, approved_at)

            if resume_command is not None:
                commands = CommandRepository(self.session)
                await commands.enqueue(
                    resume_command.idempotency_key,
                    resume_command.command_type,
                    resume_command.thread_id,
                    resume_command.payload,
                )
        except (DecisionConflict, StaleReservation):
            await self.session.rollback()
            raise
        else:
            await self.session.commit()

    async def allocate_capacity(self, reservation_id: UUID) -> bool:
        """Atomically: claim capacity for an approved reservation and mark
        it 'recorded' (the system's confirmed-booking state) - independent
        of MCP export, which is tracked separately via export_status.

        Returns False (never raises) if the reservation isn't approved, or
        if capacity is full; both are normal business outcomes, not errors.
        """
        reservation = await self.reservations.get_for_update(reservation_id)

        if reservation is None or reservation.status != "approved":
            return False

        allocated = await self.allocations.allocate(
            reservation.parking_id,
            reservation.space_type,
            reservation.id,
            reservation.starts_at,
            reservation.ends_at,
        )

        if not allocated:
            await self.session.rollback()
            return False

        reservation.status = "recorded"
        await self.session.commit()

        return True
