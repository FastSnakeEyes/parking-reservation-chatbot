# parking-reservation-chatbot

An intelligent chatbot for a fictional Vilnius parking operator (VilniusPark). It answers
questions about parking spaces, prices, hours and rules; collects reservation details
interactively; escalates each request to a human administrator for approval; and records
approved reservations through an MCP server - built as the deliverable for the EPAM "AI
Engineering: Fast Track" course assignment described in [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Architecture

| Stage | What it is | Where |
| --- | --- | --- |
| 1. RAG chatbot | LangChain tool-calling agent over a hybrid store: Milvus for static docs (general info, rules, booking process, per-car-park pages), Postgres for dynamic data (hours, prices, capacity, reservations) | `app/ai/`, `app/db/` |
| 1. Guardrails | Presidio+spaCy PII detection/redaction on search queries and answers; optional transformer-based prompt-injection screening | `app/ai/guardrails.py` |
| 2. Human-in-the-loop admin agent | A LangChain chain drafts the approval-request message; a REST API + Streamlit console is the administrator's side of the exchange | `app/ai/admin_agent.py`, `app/api/routers/admin.py`, `ui/admin_app.py` |
| 3. MCP server | Standalone server (streamable-HTTP, bearer-token auth) exposing one tool, `record_reservation`, which appends approved reservations to a file | `mcp_server/server.py` |
| 4. LangGraph orchestration | `user_interaction` node wraps the chat loop; `admin_approval` (durable `interrupt()`, sqlite-checkpointed) + `record_or_close` form the approval pipeline that calls the MCP tool | `app/ai/graph.py` |

### Request flow

1. A user chats via the web UI (`app/templates/chat.html`) → `POST /chat`. The agent answers
   from Milvus/Postgres, or collects booking details and asks the user to `confirm`.
2. On `confirm`, the reservation is saved to Postgres with `status="pending"`, and
   `ParkingChat.start_approval()` drafts a notification and starts the approval graph, which
   immediately pauses at `admin_approval` (an `interrupt()`, persisted by a sqlite checkpointer
   so it survives a restart).
3. The administrator polls `GET /admin/reservations` (via `ui/admin_app.py`) and approves or
   rejects. `POST /admin/reservations/{id}/decision` resumes the graph with that decision.
4. On approval, `record_or_close` calls the MCP server's `record_reservation` tool (writes
   `Name | Car Number | Reservation Period | Approval Time` to a file) and sets the
   reservation's status to `recorded`; on rejection it's set to `rejected`.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (pins the Python version automatically)
- Docker (and Docker Compose) for Postgres, Redis, Milvus, and the MCP server
- An OpenAI API key

## Setup

1. Clone the repository using SSH:

   ```bash
   # Default (no SSH host alias configured)
   git clone git@github.com:FastSnakeEyes/parking-reservation-chatbot.git

   # If you have a github-personal host alias set up in your SSH config
   git clone git@github-personal:FastSnakeEyes/parking-reservation-chatbot.git
   ```

2. Copy `.env.example` to `.env.development` and fill in real values (at minimum
   `OPENAI_API_KEY`, and your own `ADMIN_API_KEY`/`MCP_API_KEY`):

   ```bash
   cp .env.example .env.development
   ```

3. Start the stack:

   ```bash
   docker compose --env-file .env.development up --build -d
   ```

4. Run migrations:

   ```bash
   docker compose --env-file .env.development run --rm --no-deps api uv run --frozen --no-sync alembic upgrade head
   ```

5. Ingest the static docs (Milvus) and dynamic data (Postgres):

   ```bash
   docker compose --env-file .env.development run --rm --no-deps api python -m scripts.ingest_static
   docker compose --env-file .env.development run --rm --no-deps api python -m scripts.ingest_dynamic
   ```

## Usage

- **Chat UI**: open `http://localhost:${API_PORT}/` (default `8000`).
- **Admin console** (Stage 2, approve/reject pending reservations) - two options,
  same `/admin/*` API underneath:
  - Web page: open `http://localhost:${API_PORT}/admin` and enter your `ADMIN_API_KEY`
    (saved in the browser's local storage).
  - Streamlit:

    ```bash
    ADMIN_API_KEY=... API_BASE_URL=http://localhost:8000 make admin-ui
    ```

- **Tests** (unit + integration + e2e; spins up its own ephemeral Postgres via
  testcontainers and fakes Redis, so no running stack is required):

  ```bash
  make test
  ```

- **RAG evaluation** (Precision@K/Recall@K against a hand-labelled golden set, plus
  retrieval latency; requires the static docs to already be ingested):

  ```bash
  make eval
  ```

- **Load test** (Locust, against a running stack):

  ```bash
  make load-test
  ```

## Project structure

```
app/
  ai/            chat agent, LangGraph orchestration, admin agent, guardrails, tools
  api/routers/   /chat and /admin REST endpoints
  db/            SQLAlchemy models and repositories (Postgres)
  schemas/       Pydantic request/response models
  services/      Redis-backed chat session store
  static/, templates/   the chat web UI
mcp_server/      Stage 3 MCP server (records approved reservations to a file)
ui/              Stage 2 Streamlit admin console
migrations/      Alembic migrations
scripts/         Milvus/Postgres data ingestion
evals/           RAG evaluation (golden set + script)
load_tests/      Locust load test
tests/           unit / integration / e2e tests
docs/            the course specification this repo implements
```

## Linting & Formatting

Run Ruff from the project root:

```bash
uv run ruff check .
```

Check formatting without changing files:

```bash
uv run ruff format --check .
```

Automatically fix safe linting problems:

```bash
uv run ruff check . --fix
```

Format the code:

```bash
uv run ruff format .
```

Recommended full sequence:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
```
