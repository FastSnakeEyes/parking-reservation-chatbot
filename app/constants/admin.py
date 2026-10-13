"""The admin API's key check (app/dependencies.py's require_admin)."""

# Failed X-Admin-Key attempts allowed per client IP within the window, after
# which that IP is refused (even with the right key) until the window expires.
ADMIN_AUTH_MAX_FAILURES = 10
ADMIN_AUTH_WINDOW_SECONDS = 300

# Shared-secret defaults for local development only (ADMIN_API_KEY, MCP_API_KEY).
# They're published in this repository, so production refuses to start with them
# (app/config.py, mcp_server/server.py) - or with any key shorter than this.
DEV_ONLY_SECRET = "local-dev-only-change-me"  # noqa: S105 (a known dev value production rejects)
MIN_PRODUCTION_SECRET_LENGTH = 16
