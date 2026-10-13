ARG PYTHON_VERSION=3.13.5
FROM python:${PYTHON_VERSION}-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.18 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
COPY mcp_server ./mcp_server
COPY scripts ./scripts
COPY data ./data
COPY alembic.ini ./
COPY migrations ./migrations

# Run as an unprivileged user, not root. /data is the MCP server's reservations
# volume: created here owned by `app`, so a new named volume starts writable
# by it (Docker copies the image directory's ownership on first use).
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data
USER app

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]