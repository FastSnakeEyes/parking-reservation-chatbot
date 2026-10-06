"""Fixtures specific to the full Stage 2/3/4 flow: a real MCP server
subprocess (the same `python -m mcp_server.server` entrypoint docker-compose
runs) and an ApprovalPipeline pointed at it and at the shared test Postgres.

real_session/real_parking (plainly-committing, as opposed to db_session's
rollback-savepoint isolation) live in the root tests/conftest.py, shared
with tests/integration/.
"""

import os
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
import pytest_asyncio
from app.ai.graph import ApprovalPipeline
from app.config import settings
from langgraph.checkpoint.memory import InMemorySaver

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
