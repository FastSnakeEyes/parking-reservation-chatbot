"""Seed pending reservations for the AdminUser load test, and remove them.

    uv run python -m load_tests.seed_reservations --count 50
    uv run python -m load_tests.seed_reservations --cleanup

Seeded reservations are named "LoadTest" (the load test only decides those)
and have no conversation, so a decision on them goes straight to
ParkingService.record_decision. Each one gets its own far-future day, so the
capacity check never refuses them.
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import sqlalchemy as sa
from app.config import settings
from app.db.models import ParkingSpaceType, Reservation
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Must match load_tests/locustfile.py's LOADTEST_FIRST_NAME (not imported:
# that module needs Locust, which this script doesn't).
LOADTEST_FIRST_NAME = "LoadTest"


async def seed(count: int) -> None:
    engine = create_async_engine(settings.database_url)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            space = await session.scalar(
                sa.select(ParkingSpaceType).where(ParkingSpaceType.bookable_spaces > 0).limit(1)
            )
            if space is None:
                raise SystemExit("No bookable space type - run scripts.ingest_dynamic first")

            repository = ReservationRepository(session)
            first_day = datetime.now(UTC).replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=365)

            for day in range(count):
                starts_at = first_day + timedelta(days=day)
                await repository.submit_pending(
                    uuid4(),
                    BookingInput(
                        parking_id=space.parking_id,
                        space_type=space.space_type,
                        first_name=LOADTEST_FIRST_NAME,
                        surname="User",
                        car_number="LOAD001",
                        starts_at=starts_at,
                        ends_at=starts_at + timedelta(hours=1),
                    ),
                )
        print(f"Seeded {count} pending {LOADTEST_FIRST_NAME} reservations")
    finally:
        await engine.dispose()


async def cleanup() -> None:
    engine = create_async_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(sa.delete(Reservation).where(Reservation.first_name == LOADTEST_FIRST_NAME))
        print(f"Deleted {result.rowcount} {LOADTEST_FIRST_NAME} reservations")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--cleanup", action="store_true", help="delete the seeded reservations instead")
    args = parser.parse_args()

    asyncio.run(cleanup() if args.cleanup else seed(args.count))


if __name__ == "__main__":
    main()
