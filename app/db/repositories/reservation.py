from datetime import UTC, datetime
from uuid import UUID
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import CapacityOverride, ParkingSpaceType, Reservation
from app.db.repositories.postgres import PostgresRepository
from app.enums import ReservationStatus
from app.schemas.chat import BookingInput

# Reservations that hold a space: submitted (awaiting the admin) or approved.
ACTIVE_STATUSES = (ReservationStatus.PENDING, ReservationStatus.APPROVED)


def _peak_overlap(periods: list[tuple[datetime, datetime]]) -> int:
    """The most periods that overlap at any single moment."""
    events = sorted([(start, 1) for start, _ in periods] + [(end, -1) for _, end in periods])
    # At equal times ends (-1) sort before starts (+1): back-to-back periods don't overlap.
    peak = current = 0
    for _, change in events:
        current += change
        peak = max(peak, current)
    return peak


class ReservationRepository(PostgresRepository[Reservation]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=Reservation)
        self.timezone = ZoneInfo(settings.timezone)

    async def get_for_update(self, reservation_id: UUID) -> Reservation | None:
        """The reservation with its row locked until the transaction ends
        (SELECT ... FOR UPDATE), freshly read so a concurrent decision's
        status change is seen."""
        return await self.session.scalar(
            sa.select(Reservation)
            .where(Reservation.id == reservation_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )

    async def list_pending(self) -> list[Reservation]:
        rows = await self.session.scalars(
            sa.select(Reservation)
            .where(Reservation.status == ReservationStatus.PENDING)
            .order_by(Reservation.created_at)
        )
        return list(rows.all())

    async def availability(
        self,
        parking_id: int,
        space_type: str,
        starts_at: datetime,
        ends_at: datetime,
        *,
        lock: bool = False,
    ) -> dict | None:
        """Free spaces of one space type for a period; None if the parking or
        space type doesn't exist.

        bookable_spaces: the space type's bookable capacity, or the smallest
          capacity override overlapping the period.
        reserved: the peak number of active (pending or approved) reservations
          held at the same moment within the period - back-to-back bookings
          don't count twice.
        available: bookable_spaces - reserved, never below 0.

        lock=True locks the space type's row until the transaction ends
        (SELECT ... FOR UPDATE), so concurrent submissions check and insert
        one at a time - see submit_pending.
        """
        capacity_query = sa.select(ParkingSpaceType).where(
            ParkingSpaceType.parking_id == parking_id,
            ParkingSpaceType.space_type == space_type,
        )
        if lock:
            capacity_query = capacity_query.with_for_update()

        capacity = await self.session.scalar(capacity_query)
        if capacity is None:
            return None

        override = await self.session.scalar(
            sa.select(sa.func.min(CapacityOverride.bookable_capacity)).where(
                CapacityOverride.parking_id == parking_id,
                CapacityOverride.space_type == space_type,
                CapacityOverride.starts_at < ends_at,
                CapacityOverride.ends_at > starts_at,
            )
        )
        bookable = capacity.bookable_spaces if override is None else min(capacity.bookable_spaces, override)

        overlapping = await self.session.execute(
            sa.select(Reservation.starts_at, Reservation.ends_at).where(
                Reservation.parking_id == parking_id,
                Reservation.space_type == space_type,
                Reservation.status.in_(ACTIVE_STATUSES),
                Reservation.starts_at < ends_at,
                Reservation.ends_at > starts_at,
            )
        )
        reserved = _peak_overlap(overlapping.all())

        return {"bookable_spaces": bookable, "reserved": reserved, "available": max(bookable - reserved, 0)}

    async def validate_request(self, booking: BookingInput, *, lock: bool = False) -> dict:
        """Raise ValueError if the booking can't be made; return its availability."""
        if booking.starts_at <= datetime.now(UTC):
            raise ValueError("Reservation start must be in the future")

        availability = await self.availability(
            booking.parking_id, booking.space_type, booking.starts_at, booking.ends_at, lock=lock
        )

        if availability is None:
            raise ValueError("Parking or space type not found")

        if availability["bookable_spaces"] == 0:
            raise ValueError("This space type is not bookable")

        if availability["available"] == 0:
            raise ValueError("No spaces of this type are available for that period")

        return availability

    async def submit_pending(
        self,
        reservation_id: UUID,
        booking: BookingInput,
        conversation_id: str | None = None,
    ) -> UUID:
        # Return the receipt on a retry after a successful DB commit.
        existing = await self.session.get(
            Reservation,
            reservation_id,
        )

        if existing is not None:
            return existing.id

        # Locks the space type's row until the commit below, so two concurrent
        # submissions can't both take the last free space.
        try:
            await self.validate_request(booking, lock=True)
        except ValueError:
            await self.session.rollback()  # release the row lock right away
            raise

        statement = (
            insert(Reservation.__table__)
            .values(
                id=reservation_id,
                **booking.model_dump(),
                status=ReservationStatus.PENDING,
                conversation_id=conversation_id,
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
