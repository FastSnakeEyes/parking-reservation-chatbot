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
| 1. Guardrails | `PIIMiddleware` redacts contact/payment data everywhere and other customers' names/car numbers from tool results; Presidio+spaCy keeps PII out of vector-search queries and the ingested docs; optional transformer-based prompt-injection screening. The operator's own published contacts (`PUBLIC_CONTACTS`) are never redacted | `app/ai/guardrails.py` |
| 2. Human-in-the-loop admin agent | A second `create_agent` proposes an `request_admin_approval` call; LangChain's `HumanInTheLoopMiddleware` pauses on it until the administrator approves or rejects it through the REST API (web page or Streamlit console) | `app/ai/agents.py`, `app/api/routers/admin.py`, `ui/admin_app.py` |
| 3. MCP server | Standalone server (streamable-HTTP, bearer-token auth) exposing one tool, `record_reservation`, which appends approved reservations to a file | `mcp_server/server.py` |
| 4. LangGraph orchestration | `ParkingGraph`: `user_agent → admin → record → notification`, one Postgres-checkpointed thread per chat session, so a conversation paused for approval survives restarts | `app/ai/graph.py` |

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
START -> user_agent -+-> admin -> record -> notification -> END   (after `confirm`)
                     +-> END                                       (any other turn)
```

1. A user chats via the web UI (`app/templates/chat.html`) → `POST /chat` → `ParkingGraph.chat()`.
   The `user_agent` node first handles the injection guard and the `cancel`/`confirm` commands
   without the model; anything else goes to the user agent, which answers from Milvus/Postgres
   or collects booking details and shows a review. The review comes with **Confirm** and
   **Cancel** buttons (`/chat` returns `awaiting_confirmation: true`); typing `confirm` /
   `cancel` works too.
2. On `confirm`, `user_agent` calls `ParkingService.book()`: the reservation is saved with
   `status="pending"` and its `conversation_id`. The `admin` node then runs the
   **human-in-the-loop admin agent**: it is made to propose an `request_admin_approval` call whose
   summary describes the reservation, and LangChain's `HumanInTheLoopMiddleware` pauses on that
   call - and with it the whole conversation (persisted by the checkpointer, so it survives a
   restart). `/chat` answers `status: "waiting_for_admin"` until the admin decides.
3. The administrator polls `GET /admin/reservations` (via `ui/admin_app.py`), sees the agent's
   summary, and approves or rejects. `POST /admin/reservations/{id}/decision` looks up the
   reservation's `conversation_id` and resumes that conversation (`ParkingGraph.resume()`).
4. The `record` node acts on the decision through `ParkingService.record_decision()`:
   approved, it calls the MCP server's `record_reservation` tool (writes
   `Name | Car Number | Reservation Period | Approval Time` to a file) and sets the
   reservation's status to `approved`; rejected, it sets it to `rejected`.
5. The `notification` node leaves the outcome for the customer. While a reservation awaits
   approval, the chat page polls `GET /chat/updates` every 5 seconds and shows the decision as
   a pop-up ("Reservation approved" / "Reservation rejected") as soon as it arrives - also when
   the page is reopened later. The notice is delivered once.

Pending reservations with no conversation awaiting approval (e.g. submitted before
reservations stored a `conversation_id`) can still be decided: the admin API then records the
decision directly, without a customer notice.

### Conversation memory

Each chat session's history lives in its checkpointer thread. The user agent's
`SummarizationMiddleware` folds older turns into a summary once the history outgrows
`HISTORY_MAX_TOKENS`, keeping the newest `HISTORY_KEEP_TOKENS` verbatim. A background job
(`app/ai/chat_cleanup.py`) deletes chat threads idle longer than `COOKIE_TTL`, every
`CHAT_CLEANUP_INTERVAL` seconds - never one still waiting for an admin decision.

### HTTP API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Chat page (sets the session cookie) |
| `POST /chat` | One chat turn: `{question}` → `{status, answer, sources, awaiting_confirmation}` |
| `GET /chat/updates` | Polled while a reservation awaits approval: `{status, notice, reservation_status}` |
| `GET /admin` | Admin console page |
| `GET /admin/reservations` | Pending reservations with the admin agent's summary (`X-Admin-Key`) |
| `POST /admin/reservations/{id}/decision` | `{decision: approved \| rejected, note?}` → `{status}` (`X-Admin-Key`) |
| `GET /health` | API, Postgres and Redis health |

Reservation statuses: `pending` → `approved` (also written through MCP) or `rejected`.

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
   `OPENAI_API_KEY`, and your own `ADMIN_API_KEY`/`MCP_API_KEY`; `PUBLIC_CONTACTS` must match the
   support phone/email in `data/vilnius-parking-data/static/general_info.md`):

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

- **Chat UI**: open `http://localhost:${API_PORT}/` (default `8000`). After code changes,
  rebuild the API container: `docker compose --env-file .env.development up -d --build api`
  (and run migrations if new ones were added).
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
  ai/
    graph.py         ParkingGraph: the conversation workflow (user_agent -> admin -> record
                     -> notification) and its chat/updates/resume API
    agents.py        the user agent and the human-in-the-loop admin agent
    states.py        state schemas: ParkingState (the graph's), UserAgentState
    tools.py         the agents' tools and ChatContext (per-request dependencies)
    middleware.py    agent middleware: current-time prompt, forced approval request
    guardrails.py    PII redaction (PIIGuard, PIIMiddleware) and prompt-injection check
    prompts.py       system, summary and admin prompts
    parsers.py       LLM parser for a free-text admin reply (future email/messenger channel)
    chat_cleanup.py  background deletion of idle chat threads
  api/routers/   /chat, /admin REST endpoints, and pages.py (/, /admin, /health)
  services/      ParkingService: book, approval facts, pending reservations, record a
                 decision (MCP) - called by ParkingGraph's nodes and the admin API
  db/            SQLAlchemy models and repositories (Postgres)
  schemas/       Pydantic request/response models
  constants/     fixed values, one module per domain (paths, chat, llm, guardrails)
  enums/         all enums, one module per domain (statuses, space/day types, PII types, ...)
  static/, templates/   the chat and admin web pages
  config.py      settings, from .env.<APP_ENV>
  dependencies.py  FastAPI dependencies: DB session, ParkingService/ParkingGraph,
                 cookie/admin-key checks, rate limits
  lifespan.py    startup/shutdown: DB/Redis/Milvus/checkpointer setup, ParkingGraph,
                 cleanup job, LangSmith config
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
