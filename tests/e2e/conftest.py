"""Fixtures specific to the full Stage 2/3/4 flow: a real MCP server
subprocess (the same `python -m mcp_server.server` entrypoint docker-compose
runs) and an ApprovalPipeline pointed at it and at the shared test Postgres.
"""

import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from datetime import date
from datetime import time as time_
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from app.ai.graph import ApprovalPipeline
from app.config import settings
from app.db.models import Parking, ParkingSpaceType, Price, WorkingHours
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

MCP_TEST_PORT = 9401


@pytest.fixture(scope="session")
def mcp_test_server(tmp_path_factory) -> Iterator[dict]:
    reservations_file = tmp_path_factory.mktemp("mcp") / "approved_reservations.txt"

    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        env={
            "PATH": os.environ["PATH"],
            "MCP_API_KEY": "test-mcp-key",
            "RESERVATIONS_FILE": str(reservations_file),
            "MCP_PORT": str(MCP_TEST_PORT),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://localhost:{MCP_TEST_PORT}"
    deadline = time.monotonic() + 15

    try:
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass

            time.sleep(0.2)
        else:
            raise RuntimeError("mcp_server did not become healthy in time")

        yield {"url": f"{base_url}/mcp", "api_key": "test-mcp-key", "file": reservations_file}
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest_asyncio.fixture
async def approval_pipeline(engine, mcp_test_server, monkeypatch) -> ApprovalPipeline:
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "mcp_server_url", mcp_test_server["url"])
    monkeypatch.setattr(settings, "mcp_api_key", SecretStr(mcp_test_server["api_key"]))

    return ApprovalPipeline(engine, InMemorySaver())


@pytest_asyncio.fixture
async def real_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A plainly-committing session (unlike `db_session`'s rollback-savepoint
    isolation), needed here because ApprovalPipeline's record_or_close opens
    its own fresh connection and must see what this fixture writes."""
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
                open_time=time_(0, 0),
                close_time=time_(23, 59),
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
