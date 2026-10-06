"""Shared test fixtures.

Sets required env vars before any `app.*` import happens anywhere in the
suite (app.config.settings is a module-level singleton, constructed on
first import), so tests never need a real .env file or real OpenAI/MCP
credentials unless a specific test opts into them.
"""

import os

os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")
os.environ.setdefault("MCP_API_KEY", "test-mcp-key")
os.environ.setdefault("MCP_SERVER_URL", "http://localhost:0/mcp")
os.environ.setdefault("USE_INJECTION_MODEL", "false")
# The sandbox this project was built in can't bind-mount the docker socket
# testcontainers' Ryuk reaper needs; disabling it just means test containers
# are cleaned up by the fixture teardown instead of a watchdog container.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

from collections.abc import AsyncIterator
from datetime import date, time

import fakeredis
import pytest
import pytest_asyncio
from app.db.models import Base, Parking, ParkingSpaceType, Price, WorkingHours
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_container() -> AsyncIterator[str]:
    with PostgresContainer("postgres:18", driver="asyncpg") as container:
        yield container.get_connection_url()


@pytest_asyncio.fixture(scope="session")
async def engine(postgres_container: str) -> AsyncIterator[AsyncEngine]:
    test_engine = create_async_engine(postgres_container)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """One test per outer transaction; app code's own commits only close a
    savepoint, so nothing persists once the fixture rolls the outer
    transaction back."""
    connection = await engine.connect()
    transaction = await connection.begin()

    session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    yield session

    await session.close()
    await transaction.rollback()
    await connection.close()


@pytest.fixture
def fake_redis():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


@pytest_asyncio.fixture
async def sample_parking(db_session: AsyncSession) -> Parking:
    """One bookable parking with a standard space type and a flat price,
    open every day, for tests that need a valid parking_id to book against."""
    parking = Parking(
        slug="test-parking",
        name="Test Parking",
        district="Center",
        latitude=54.68,
        longitude=25.27,
        total_spaces=10,
        bookable_spaces=5,
    )
    db_session.add(parking)
    await db_session.flush()

    db_session.add(
        ParkingSpaceType(
            parking_id=parking.id,
            space_type="standard",
            total_spaces=10,
            bookable_spaces=5,
        )
    )

    for day in range(1, 8):
        db_session.add(
            WorkingHours(
                parking_id=parking.id,
                day_of_week=day,
                open_time=time(0, 0),
                close_time=time(23, 59),
                is_closed=False,
            )
        )

    for day_type in ("weekday", "weekend"):
        db_session.add(
            Price(
                parking_id=parking.id,
                space_type="standard",
                day_type=day_type,
                valid_from=date(2020, 1, 1),
                price_per_hour=1,
                daily_max=10,
            )
        )

    await db_session.commit()

    return parking
