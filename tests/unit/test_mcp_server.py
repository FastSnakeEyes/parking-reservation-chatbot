"""Stage 3's MCP server (mcp_server/server.py): it only accepts the shared
bearer token, and writes the spec's line format.

The e2e tests (tests/e2e/) drive it as a real subprocess through the whole
approval flow; these test its pieces in-process."""

import pytest
from mcp_server import server
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Module-scoped: FastMCP's session manager can only be started once per process.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MCP_API_KEY", "secret-token")
        app = server.build_app()

    with TestClient(app) as test_client:
        yield test_client


def _call_tool():
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    }


MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong-token", "secret-token", "Bearer secret-token-but-longer"],
    ids=["missing", "wrong", "no-bearer-prefix", "prefix-of-longer"],
)
def test_rejects_requests_without_the_right_token(client, authorization):
    headers = {**MCP_HEADERS, **({"Authorization": authorization} if authorization else {})}

    response = client.post("/mcp", json=_call_tool(), headers=headers)

    assert response.status_code == 401


def test_accepts_the_right_token(client):
    response = client.post("/mcp", json=_call_tool(), headers={**MCP_HEADERS, "Authorization": "Bearer secret-token"})

    assert response.status_code == 200


def test_health_needs_no_token(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.delenv("MCP_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MCP_API_KEY"):
        server.build_app()


@pytest.mark.asyncio
async def test_record_reservation_appends_the_spec_line_format(monkeypatch, tmp_path):
    reservations_file = tmp_path / "approved_reservations.txt"
    monkeypatch.setattr(server, "RESERVATIONS_FILE", reservations_file)

    for car in ("ABC123", "XYZ987"):
        await server.record_reservation(
            name="Jane Doe",
            car_number=car,
            period="2026-10-20T10:00:00+03:00 - 2026-10-20T12:00:00+03:00",
            approval_time="2026-10-13T18:00:00+03:00",
        )

    # Name | Car Number | Reservation Period | Approval Time - one line per approval
    assert reservations_file.read_text().splitlines() == [
        "Jane Doe | ABC123 | 2026-10-20T10:00:00+03:00 - 2026-10-20T12:00:00+03:00 | 2026-10-13T18:00:00+03:00",
        "Jane Doe | XYZ987 | 2026-10-20T10:00:00+03:00 - 2026-10-20T12:00:00+03:00 | 2026-10-13T18:00:00+03:00",
    ]
