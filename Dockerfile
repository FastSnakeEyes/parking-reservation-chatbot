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

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]