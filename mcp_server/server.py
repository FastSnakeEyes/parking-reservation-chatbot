"""Stage 3: MCP server that records administrator-approved reservations.

Standalone networked service (not an in-process stdio subprocess), run via
`python -m mcp_server.server`. It deliberately does not import app.config:
it is a separate deployable with its own minimal env (MCP_API_KEY,
RESERVATIONS_FILE, MCP_PORT), not the full app Settings model.

Exposes one tool, record_reservation, which appends one line per approved
reservation: "Name | Car Number | Reservation Period | Approval Time".
"""

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

RESERVATIONS_FILE = Path(os.environ.get("RESERVATIONS_FILE", "./data/approved_reservations.txt"))

server = FastMCP(name="parking-reservation-mcp", host="0.0.0.0")  # noqa: S104 (containerized service)

_write_lock = asyncio.Lock()


@server.tool()
async def record_reservation(
    name: str,
    car_number: str,
    period: str,
    approval_time: str,
) -> str:
    """Append an administrator-approved reservation to the reservations file.

    Writes one line: "Name | Car Number | Reservation Period | Approval Time".
    """
    line = f"{name} | {car_number} | {period} | {approval_time}"

    async with _write_lock:
        await asyncio.to_thread(_append_line, line)

    return f"Recorded: {line}"


def _append_line(line: str) -> None:
    RESERVATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESERVATIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@server.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


UNAUTHENTICATED_PATHS = {"/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Static shared-secret check.

    mcp's built-in AuthSettings models a full OAuth protected-resource
    server (issuer_url, resource metadata endpoints, ...), which is more
    than this internal service needs. A shared bearer token between this
    server and the one LangGraph client that calls it is enough to satisfy
    "resistant to unauthorized access" here.
    """

    def __init__(self, app, token: str):
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path in UNAUTHENTICATED_PATHS:
            return await call_next(request)

        if request.headers.get("authorization") != f"Bearer {self._token}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


def build_app() -> Starlette:
    api_key = os.environ.get("MCP_API_KEY")

    if not api_key:
        raise RuntimeError("MCP_API_KEY is required")

    app = server.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=api_key)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "9000")))  # noqa: S104


if __name__ == "__main__":
    main()
