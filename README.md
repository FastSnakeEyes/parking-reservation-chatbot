# parking-reservation-chatbot

An intelligent chatbot for a fictional Vilnius parking operator (VilniusPark). It answers
questions about parking spaces, prices, hours and rules; collects reservation details
interactively; escalates each request to a human administrator for approval; and records
approved reservations through an MCP server - built as the deliverable for the EPAM "AI
Engineering: Fast Track" course assignment described in [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Architecture

| Stage | What it is | Where |
| --- | --- | --- |
| 1. RAG chatbot | `langchain.agents.create_agent` user agent over a hybrid store: Milvus for static docs (general info, rules, booking process, per-car-park pages), Postgres for dynamic data (hours, prices, capacity, reservations) | `app/ai/agents.py`, `app/ai/tools.py`, `app/db/` |
| 1. Guardrails | PIIMiddleware redacts the agent's own output; Presidio+spaCy additionally screens search queries and blocks PII from reaching the vector store; optional transformer-based prompt-injection screening | `app/ai/agents.py`, `app/ai/guardrails.py` |
| 2. Human-in-the-loop admin agent | A second `create_agent` drafts the approval-request message; a REST API + Streamlit console is the administrator's side of the exchange | `app/ai/agents.py`, `app/api/routers/admin.py`, `ui/admin_app.py` |
| 3. MCP server | Standalone server (streamable-HTTP, bearer-token auth) exposing one tool, `record_reservation`, which appends approved reservations to a file | `mcp_server/server.py` |
| 4. LangGraph orchestration | `user_interaction` node wraps the chat loop; `admin_approval` (durable `interrupt()`, Postgres-checkpointed) + `record_or_close` form the approval pipeline that calls the MCP tool | `app/ai/graph.py` |

### Logging and tracing

Standard library logging, configured once in `app/logger.py` (level from `LOG_LEVEL`,
default `INFO`) and used throughout `app/` for the reservation/approval/guardrail lifecycle
(submitted, approved, rejected, PII redacted, injection blocked, admin auth failures).
`mcp_server/server.py` configures its own logger the same way, since it's a separate
deployable that doesn't import `app.config`/`app.logger`.

[LangSmith](https://smith.langchain.com) tracing for the LangChain/LangGraph calls is off by
default; set `LANGSMITH_TRACING=true` plus `LANGSMITH_API_KEY` (and optionally
`LANGSMITH_PROJECT`) to enable it - wired up in `app/lifespan.py`.

### Request flow

`POST /chat` → `ParkingGraph` (`app/ai/graph.py`) → graph nodes → `ParkingService`
(`app/services/parking.py`) → repositories. Routes only handle HTTP (validation, auth,
response shape); the graph owns the conversation workflow, one Postgres-checkpointed thread
per chat session (`chat:{session_id}`); the service is a capability the graph calls, and
never calls the graph.

```
START -> route_input -+-> user_agent -------------------------------------> END
                      +-> admin_agent -> recording -> send_notification -> END
                      +-> END   (notice / injection / cancel / nothing to confirm)
```

1. A user chats via the web UI (`app/templates/chat.html`) → `POST /chat` → `ParkingGraph.chat()`.
   The `route_input` node handles the injection guard and the `cancel`/`confirm` commands;
   anything else goes to the `user_agent` node, which answers from Milvus/Postgres or collects
   booking details and shows a review, asking the user to `confirm`.
2. On `confirm`, `route_input` calls `ParkingService.book()`: the reservation is saved with
   `status="pending"` and its `conversation_id`. The `admin_agent` node then runs the
   **human-in-the-loop admin agent**: it is made to propose an `approve_reservation` call whose
   summary describes the reservation, and LangChain's `HumanInTheLoopMiddleware` pauses on that
   call - and with it the whole conversation (persisted by the checkpointer, so it survives a
   restart). `/chat` answers `status: "waiting_for_admin"` until the admin decides.
3. The administrator polls `GET /admin/reservations` (via `ui/admin_app.py`), sees the agent's
   summary, and approves or rejects. `POST /admin/reservations/{id}/decision` looks up the
   reservation's `conversation_id` and resumes that conversation (`ParkingGraph.resume()`).
4. The `recording` node acts on the decision through `ParkingService.record_decision()`:
   approved, it calls the MCP server's `record_reservation` tool (writes
   `Name | Car Number | Reservation Period | Approval Time` to a file) and sets the
   reservation's status to `recorded`; rejected, it sets it to `rejected`.
5. The `send_notification` node leaves the outcome for the customer: their next message in that
   session is answered with the approved/rejected notice.

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
  ai/            ParkingGraph (the conversation workflow, incl. the admin-approval
                 interrupt), user/admin agents, guardrails, tools, idle-thread cleanup
  api/routers/   /chat, /admin REST endpoints, and pages.py (/, /admin, /health)
  services/      ParkingService: book, approval facts, pending reservations, record a
                 decision (MCP) - called by ParkingGraph's nodes and the admin API
  db/            SQLAlchemy models and repositories (Postgres)
  schemas/       Pydantic request/response models
  static/, templates/   the chat web UI
  lifespan.py    startup/shutdown: DB/Redis/Milvus/checkpointer setup, LangSmith config
  logger.py      logging setup (LOG_LEVEL)
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
