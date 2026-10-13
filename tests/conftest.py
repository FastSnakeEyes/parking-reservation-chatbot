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
from uuid import uuid4

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


class FakeLock:
    async def acquire(self, *args, **kwargs):
        return True

    async def release(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None


class LockOnlyRedis:
    """Only ParkingGraph's per-thread lock. A real redis-py lock releases via a Lua
    script, which in-memory Redis fakes can't run - so tests stub the lock itself."""

    def lock(self, *args, **kwargs):
        return FakeLock()


class CounterRedis:
    """In-memory stand-in for the Redis counters require_admin uses (get/incr/expire)."""

    def __init__(self):
        self.values: dict[str, int] = {}

    async def get(self, key):
        return self.values.get(key)

    async def incr(self, key):
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key, seconds):
        return True


@pytest.fixture
def lock_redis():
    return LockOnlyRedis()


@pytest_asyncio.fixture
async def checkpointer(postgres_container):
    """The real AsyncPostgresSaver the app uses, on the test Postgres."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    url = postgres_container.replace("postgresql+asyncpg://", "postgresql://")

    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        yield saver


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


@pytest_asyncio.fixture
async def real_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A plainly-committing session (unlike `db_session`'s rollback-savepoint
    isolation), so the whole flow runs against committed data, as in the app."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session


@pytest_asyncio.fixture
async def real_parking(real_session: AsyncSession) -> Parking:
    parking = Parking(
        slug=f"e2e-parking-{uuid4()}",
        name="E2E Test Parking",
        district="Center",
        latitude=54.68,
        longitude=25.27,
        total_spaces=10,
        bookable_spaces=5,
    )
    real_session.add(parking)
    await real_session.flush()

    real_session.add(
        ParkingSpaceType(
            parking_id=parking.id,
            space_type="standard",
            total_spaces=10,
            bookable_spaces=5,
        )
    )

    for day in range(1, 8):
        real_session.add(
            WorkingHours(
                parking_id=parking.id,
                day_of_week=day,
                open_time=time(0, 0),
                close_time=time(23, 59),
                is_closed=False,
            )
        )

    for day_type in ("weekday", "weekend"):
        real_session.add(
            Price(
                parking_id=parking.id,
                space_type="standard",
                day_type=day_type,
                valid_from=date(2020, 1, 1),
                price_per_hour=1,
                daily_max=10,
            )
        )

    await real_session.commit()

    return parking
