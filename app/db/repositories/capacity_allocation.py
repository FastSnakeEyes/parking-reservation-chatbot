from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CapacityAllocation, ParkingSpaceType
from app.db.repositories.postgres import PostgresRepository


class CapacityAllocationRepository(PostgresRepository[CapacityAllocation]):
    """Database-level protection against overselling bookable capacity.

    This system is capacity-counted (N bookable spaces per type), not
    single-resource exclusive booking, so a Postgres EXCLUDE constraint (the
    textbook tool for "no two intervals overlap") is the wrong tool here -
    it would forbid any two overlapping reservations even when capacity is
    10. Instead: lock the capacity row for (parking_id, space_type), count
    existing overlapping allocations inside that lock, and only insert if
    count < bookable_capacity - all in one transaction. The row lock is what
    makes two concurrent requests for the last space resolve safely: one
    blocks until the other's transaction commits or rolls back, then sees
    the up-to-date count.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=CapacityAllocation)

    async def allocate(
        self,
        parking_id: int,
        space_type: str,
        reservation_id: UUID,
        starts_at: datetime,
        ends_at: datetime,
    ) -> bool:
        """Flush-only (no commit): composes inside ParkingService's atomic
        allocate+confirm transaction. Returns False (never raises) when
        capacity is full or missing, so callers handle it as a normal
        business outcome rather than an error."""
        capacity = await self.session.scalar(
            sa.select(ParkingSpaceType)
            .where(
                ParkingSpaceType.parking_id == parking_id,
                ParkingSpaceType.space_type == space_type,
            )
            .with_for_update()
        )

        if capacity is None:
            return False

        # Half-open interval overlap test: [starts_at, ends_at) intersects
        # [a.starts_at, a.ends_at) iff a.starts_at < ends_at and a.ends_at > starts_at.
        overlapping = await self.session.scalar(
            sa.select(sa.func.count())
            .select_from(CapacityAllocation)
            .where(
                CapacityAllocation.parking_id == parking_id,
                CapacityAllocation.space_type == space_type,
                CapacityAllocation.starts_at < ends_at,
                CapacityAllocation.ends_at > starts_at,
            )
        )

        if overlapping >= capacity.bookable_spaces:
            return False

        self.session.add(
            CapacityAllocation(
                parking_id=parking_id,
                space_type=space_type,
                reservation_id=reservation_id,
                starts_at=starts_at,
                ends_at=ends_at,
            )
        )
        await self.session.flush()

        return True
