from datetime import UTC, datetime
from uuid import UUID
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import ParkingSpaceType, Reservation
from app.db.repositories.postgres import PostgresRepository
from app.schemas.chat import BookingInput



class StaleReservation(Exception):
    """Raised when an operation references a reservation revision/status
    that no longer matches what's in the database (e.g. a decision made
    against a reservation that was since decided, or doesn't exist)."""


class ReservationRepository(PostgresRepository[Reservation]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=Reservation)
        self.timezone = ZoneInfo(settings.timezone)

    async def validate_request(self, booking: BookingInput) -> None:
        if booking.starts_at <= datetime.now(UTC):
            raise ValueError("Reservation start must be in the future")

        capacity = await self.session.scalar(
            sa.select(ParkingSpaceType).where(
                ParkingSpaceType.parking_id == booking.parking_id,
                ParkingSpaceType.space_type == booking.space_type,
            )
        )

        if capacity is None:
            raise ValueError("Parking or space type not found")

        if capacity.bookable_spaces == 0:
            raise ValueError("This space type is not bookable")

    async def submit_pending(
        self,
        reservation_id: UUID,
        booking: BookingInput,
    ) -> UUID:
        # Return the receipt on a retry after a successful DB commit.
        existing = await self.session.get(
            Reservation,
            reservation_id,
        )

        if existing is not None:
            return existing.id

        await self.validate_request(booking)

        statement = (
            insert(Reservation.__table__)
            .values(
                id=reservation_id,
                **booking.model_dump(),
                status="pending",
            )
            .on_conflict_do_nothing(
                index_elements=[Reservation.__table__.c.id],
            )
        )

        try:
            await self.session.execute(statement)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise

        return reservation_id

    async def get_for_update(self, reservation_id: UUID) -> Reservation | None:
        return await self.session.scalar(
            sa.select(Reservation).where(Reservation.id == reservation_id).with_for_update()
        )

    async def mark_decided(
        self,
        reservation_id: UUID,
        expected_version: int,
        status: str,
        approved_at: datetime | None = None,
    ) -> Reservation:
        """Flush-only (no commit): composes inside ParkingService's atomic
        decision transaction, alongside AdminDecisionRepository.record().

        Locks the row, then requires it still be 'pending' at exactly
        expected_version - this is the staleness check for "approval must
        reference the exact immutable reservation revision reviewed."
        """
        reservation = await self.get_for_update(reservation_id)

        if reservation is None:
            raise StaleReservation(f"Reservation {reservation_id} not found")

        if reservation.status != "pending" or reservation.version != expected_version:
            raise StaleReservation(
                f"Reservation {reservation_id} is no longer pending at version {expected_version} "
                f"(status={reservation.status}, version={reservation.version})"
            )

        reservation.status = status

        if approved_at is not None:
            reservation.approved_at = approved_at

        await self.session.flush()

        return reservation
