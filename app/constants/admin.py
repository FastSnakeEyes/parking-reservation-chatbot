"""The admin API's key check (app/dependencies.py's require_admin)."""

# Failed X-Admin-Key attempts allowed per client IP within the window, after
# which that IP is refused (even with the right key) until the window expires.
ADMIN_AUTH_MAX_FAILURES = 10
ADMIN_AUTH_WINDOW_SECONDS = 300
