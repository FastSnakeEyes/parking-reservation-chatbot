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
