"""Stage 3: MCP server that records administrator-approved reservations.

Standalone networked service (not an in-process stdio subprocess), run via
`python -m mcp_server.server`. It deliberately does not import app.config:
it is a separate deployable with its own minimal env (MCP_API_KEY,
RESERVATIONS_FILE, MCP_PORT), not the full app Settings model.

Exposes one tool, record_reservation, which appends one line per approved
reservation: "Name | Car Number | Reservation Period | Approval Time".
"""

import asyncio
import logging
import os
import secrets
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Self-contained logging setup: this service deliberately doesn't import
# app.logger/app.config (see module docstring), so it configures its own.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Mirrors app/constants/admin.py.
DEV_ONLY_SECRET = "local-dev-only-change-me"  # noqa: S105 (a known dev value production rejects)
MIN_PRODUCTION_SECRET_LENGTH = 16
MIN_PRODUCTION_SECRET_DISTINCT_CHARS = 8


def is_weak_secret(value: str) -> bool:
    """Mirrors app/config.py's is_weak_secret (this service deliberately
    doesn't import app.*; tests/unit/test_mcp_server.py keeps the two in step)."""
    secret = value.strip()
    return (
        secret == DEV_ONLY_SECRET
        or len(secret) < MIN_PRODUCTION_SECRET_LENGTH
        or len(set(secret)) < MIN_PRODUCTION_SECRET_DISTINCT_CHARS
    )


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

    logger.info("Recorded reservation: car=%s, period=%s", car_number, period)

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

        # Constant-time comparison: the token can't be guessed from response timing.
        supplied = request.headers.get("authorization", "")
        if not secrets.compare_digest(supplied.encode(), f"Bearer {self._token}".encode()):
            logger.warning("Rejected an unauthenticated request to %s", request.url.path)
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


def build_app() -> Starlette:
    api_key = os.environ.get("MCP_API_KEY")

    if not api_key:
        raise RuntimeError("MCP_API_KEY is required")

    # The same rule as app/config.py (this service deliberately doesn't import
    # it): in production, never run with the repo's dev default or a short key.
    if os.environ.get("APP_ENV") == "production" and is_weak_secret(api_key):
        raise RuntimeError(
            "APP_ENV=production needs a real MCP_API_KEY: a unique random value of at least "
            f"{MIN_PRODUCTION_SECRET_LENGTH} characters, e.g. `openssl rand -hex 32`"
        )

    app = server.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=api_key)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "9000")))  # noqa: S104


if __name__ == "__main__":
    main()
