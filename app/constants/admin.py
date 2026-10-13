"""The admin API's key check (app/dependencies.py's require_admin)."""

# Failed X-Admin-Key attempts allowed per client IP within the window, after
# which that IP is refused (even with the right key) until the window expires.
ADMIN_AUTH_MAX_FAILURES = 10
ADMIN_AUTH_WINDOW_SECONDS = 300

# Shared-secret defaults for local development only (ADMIN_API_KEY, MCP_API_KEY).
# They're published in this repository, so in production the API and the MCP
# server refuse to start with them (app/config.py's check_production_secrets,
# mcp_server/server.py) - or with a key that is too short or too repetitive.
DEV_ONLY_SECRET = "local-dev-only-change-me"  # noqa: S105 (a known dev value production rejects)
MIN_PRODUCTION_SECRET_LENGTH = 16
MIN_PRODUCTION_SECRET_DISTINCT_CHARS = 8  # rejects "aaaa...", "1234123412341234"
