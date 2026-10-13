"""Load tests for each component (Stage 4): the chatbot, the administrator's
confirmation API, and the MCP server's recording tool.

Usage (pick the user classes to run; see load_tests/README section in the
project README for the full procedure):

    # chatbot in interactive dialogue mode (calls the real LLM)
    uv run --group eval locust -f load_tests/locustfile.py ChatUser --host http://localhost:8000

    # administrator confirmation (seed reservations first: load_tests/seed_reservations.py)
    ADMIN_API_KEY=... uv run --group eval locust -f load_tests/locustfile.py AdminUser --host http://localhost:8000

    # MCP server recording - point it at a test instance, it appends to the reservations file
    MCP_API_KEY=... MCP_LOAD_URL=http://localhost:9100 uv run --group eval locust -f load_tests/locustfile.py MCPUser
"""

import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    case["question"] for case in json.loads((Path(__file__).parent.parent / "evals/golden_qa.json").read_text())
]

# Seeded by load_tests/seed_reservations.py; the admin user only decides these.
LOADTEST_FIRST_NAME = "LoadTest"


class ChatUser(HttpUser):
    """A customer chatting: each simulated user is one chat session."""

    # >= 5.5 s between requests keeps a session under cookie_rate_limit's default
    # 2 requests / 10 s even when an answer comes back instantly.
    wait_time = between(5.5, 8)

    def on_start(self):
        self.client.get("/")  # the session cookie, as a real browser gets it

    @task
    def ask_a_question(self):
        self.client.post("/chat", json={"question": random.choice(QUESTIONS)})  # noqa: S311 (not cryptographic)


class AdminUser(HttpUser):
    """An administrator console: polls pending reservations and decides the
    seeded LoadTest ones. Rejects by default (no MCP write); LOADTEST_APPROVE=1
    approves instead, which also records each one through the MCP server."""

    wait_time = between(1, 2)

    def on_start(self):
        self.client.headers["X-Admin-Key"] = os.environ["ADMIN_API_KEY"]
        self.decision = "approved" if os.environ.get("LOADTEST_APPROVE") == "1" else "rejected"

    @task(4)
    def list_pending(self):
        self.client.get("/admin/reservations")

    @task(1)
    def decide_one(self):
        pending = self.client.get("/admin/reservations", name="/admin/reservations").json()
        seeded = [reservation for reservation in pending if reservation["first_name"] == LOADTEST_FIRST_NAME]
        if not seeded:
            return

        reservation = random.choice(seeded)  # noqa: S311 (not cryptographic)
        with self.client.post(
            f"/admin/reservations/{reservation['id']}/decision",
            json={"decision": self.decision, "note": "load test"},
            name="/admin/reservations/[id]/decision",
            catch_response=True,
        ) as response:
            # Another simulated admin may have decided it first.
            if response.status_code == 404:
                response.success()


class MCPUser(HttpUser):
    """The recording node calling the MCP server's record_reservation tool,
    over the same streamable-HTTP JSON-RPC protocol the app's MCP client uses."""

    host = os.environ.get("MCP_LOAD_URL", "http://localhost:9000")
    wait_time = between(0.5, 1)

    def on_start(self):
        self.client.headers.update(
            {
                "Authorization": f"Bearer {os.environ['MCP_API_KEY']}",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
        )
        response = self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "locust", "version": "1"},
            },
            name="mcp initialize",
        )
        self.client.headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        self.client.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, name="mcp initialized")

    def _rpc(self, method: str, params: dict, name: str):
        return self.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": random.randint(1, 10**9), "method": method, "params": params},  # noqa: S311
            name=name,
        )

    @task
    def record_reservation(self):
        now = datetime.now(UTC).isoformat()
        with self._rpc(
            "tools/call",
            {
                "name": "record_reservation",
                "arguments": {
                    "name": f"{LOADTEST_FIRST_NAME} User",
                    "car_number": "LOAD001",
                    "period": f"{now} - {now}",
                    "approval_time": now,
                },
            },
            name="mcp tools/call record_reservation",
        ) as response:
            # Streamable HTTP answers as JSON or as a server-sent event.
            body = response.text
            payload = json.loads(body.split("data: ", 1)[1]) if body.startswith("event:") else json.loads(body)
            if payload.get("result", {}).get("isError"):
                response.failure(f"tool error: {payload['result']}")
