"""Fixtures specific to the full Stage 2/3/4 flow: a real MCP server
subprocess (the same `python -m mcp_server.server` entrypoint docker-compose
runs), and settings pointed at it, for driving ParkingGraph end to end
against the shared test Postgres.
"""

import os
import subprocess
import sys
import time
from collections.abc import Iterator

import httpx
import pytest
from app.ai.tools import request_admin_approval
from app.config import settings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

MCP_TEST_PORT = 9401

ADMIN_DRAFT_MESSAGE = "Please approve Jane Doe's reservation."


class ScriptedModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel has no bind_tools - create_agent calls it
    unconditionally, so a fake model used with it needs a no-op override.
    Both agents share it, so e2e tests script every model turn up front
    instead of depending on (or paying for) a real OpenAI call."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


# The admin agent's only model call: it proposes request_admin_approval, which
# pauses for the administrator's review. After the decision the agent ends
# without calling the model again.
ADMIN_TURN = [
    AIMessage(
        content="",
        tool_calls=[{"name": request_admin_approval.name, "args": {"summary": ADMIN_DRAFT_MESSAGE}, "id": "call_1"}],
    ),
]


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


@pytest.fixture
def mcp_settings(mcp_test_server, monkeypatch) -> dict:
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "mcp_server_url", mcp_test_server["url"])
    monkeypatch.setattr(settings, "mcp_api_key", SecretStr(mcp_test_server["api_key"]))

    return mcp_test_server
