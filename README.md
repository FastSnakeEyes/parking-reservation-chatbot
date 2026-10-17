# parking-reservation-chatbot

An intelligent chatbot for a fictional Vilnius parking operator (VilniusPark). It answers
questions about parking spaces, prices, hours and rules; collects reservation details
interactively; escalates each request to a human administrator for approval; and records
approved reservations through an MCP server - built as the deliverable for the EPAM "AI
Engineering: Fast Track" course assignment described in [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Architecture

| Stage | What it is | Where |
| --- | --- | --- |
| 1. RAG chatbot | `langchain.agents.create_agent` user agent over a hybrid store: Milvus for static docs (general info, rules, booking process, per-car-park pages) searched with hybrid search - semantic vectors + BM25 keywords, weighted 0.7 / 0.3 - and Postgres for dynamic data (hours, prices, capacity, reservations) | `app/ai/agents.py`, `app/ai/tools.py`, `app/db/` |
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
`LANGSMITH_PROJECT`) to enable it - wired up in `app/lifespan.py`. For an EU-region
LangSmith account also set `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` (otherwise
every upload fails with 403); traces then appear at eu.smith.langchain.com. The startup log
says whether tracing is enabled. In LangSmith each chat turn is a `LangGraph` run whose
trace shows the nodes (`user_agent`, `admin`, `record`, `notification`) and their model/tool
calls; the **Threads** tab groups a conversation's runs by its `thread_id`.

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

The diagrams in [`docs/graph.md`](docs/graph.md) - this graph plus the user and admin agents'
own graphs - are rendered from the compiled code: `uv run python -m scripts.export_graph`
(add `--png` for `docs/graph.png`, rendered via the public mermaid.ink service).

1. A user chats via the web UI (`app/templates/chat.html`) → `POST /chat` → `ParkingGraph.chat()`.
   The `user_agent` node first handles the injection guard and the `cancel`/`confirm` commands
   without the model; anything else goes to the user agent, which answers from Milvus/Postgres
   (including free spaces for a period via `check_availability`) or collects booking details
   and shows a review. The review comes with **Confirm** and
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

If recording fails after the decision (e.g. the MCP server is down), the reservation stays
`pending` and the conversation keeps waiting; the admin console shows the failure, and
submitting the decision again retries the failed step (the decision may be revised).

### Conversation memory

Each chat session's history lives in its checkpointer thread. The user agent's
`SummarizationMiddleware` folds older turns into a summary once the history outgrows
`HISTORY_MAX_TOKENS`, keeping the newest `HISTORY_KEEP_TOKENS` verbatim. A background job
(`app/ai/chat_cleanup.py`) deletes chat threads idle longer than `COOKIE_TTL`, every
`CHAT_CLEANUP_INTERVAL` seconds - never one still waiting for an admin decision, and never
one whose chat turn is running (it takes the conversation's lock and re-checks first). While
the chat page waits for a decision, its polling keeps the session cookie alive.

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

The admin endpoints check `X-Admin-Key` in constant time; after 10 failed attempts within
5 minutes, a client IP is refused (429) until the window expires.

Reservation statuses: `pending` → `approved` (also written through MCP) or `rejected`.

Capacity: a space type's free spaces for a period are its bookable capacity (or the smallest
overlapping `capacity_overrides` entry) minus the peak number of `pending` + `approved`
reservations at the same moment. A submission is refused when none is free; the check and
the insert run under a row lock, so two customers can't both take the last space.

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

2. **Install uv** (the package and Python manager this project uses), once per machine - any one of:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
   brew install uv                                    # macOS with Homebrew
   python3 -m pip install uv==0.9.2                   # anywhere with Python
   uv --version                                       # check: uv 0.9.x
   ```

3. **Create the Python virtualenv and install the dependencies.** The stack itself runs in Docker,
   but tests, evaluations, the Streamlit admin UI and the scripts run on your machine:

   ```bash
   uv venv                # creates .venv with Python 3.13 (from .python-version; uv downloads it if needed)
   uv sync                # installs the app and the dev tools (pytest, ruff) into .venv
   uv sync --group eval   # optional: also Evidently and pandas, for `make eval-evidently`
   ```

   `uv venv` is needed once (`uv sync` alone also creates `.venv` if it's missing); run `uv sync`
   again after pulling changes to `pyproject.toml`. `make install` does both in one step (it
   installs uv first if it's missing). Activating the
   virtualenv is optional - `uv run <command>` and the `make` targets use `.venv` automatically:

   ```bash
   source .venv/bin/activate   # optional; then plain `pytest`, `python -m ...` work
   deactivate                  # to leave it
   ```

4. **Create `.env.development`** from the template:

   ```bash
   cp .env.example .env.development
   ```

   The app reads `.env.development` by default, `.env.production` with `APP_ENV=production`, or
   any file named by `ENV_FILE`. Those files are git-ignored; **`.env.example` is committed, so
   never put a real key in it** - keep it as a template with empty or placeholder values.

5. **Fill in the keys** in `.env.development`:

   | Variable | Required | What to put | Without it |
   | --- | --- | --- | --- |
   | `OPENAI_API_KEY` | yes | your key from https://platform.openai.com/api-keys | chat, embeddings and ingestion fail |
   | `ADMIN_API_KEY` | yes | a random value: `openssl rand -hex 32` | the admin console refuses every request (503) |
   | `MCP_API_KEY` | yes | another random value - not the same as `ADMIN_API_KEY` | the MCP server doesn't start: approvals aren't recorded |
   | `POSTGRES_PASSWORD` | yes | any password for the local database | Postgres doesn't start |
   | `LANGSMITH_API_KEY` | no | your key from https://smith.langchain.com, with `LANGSMITH_TRACING=true` (and `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` for an EU account) | no tracing |
   | `MILVUS_TOKEN` | no | only for a Milvus server with authentication (e.g. Zilliz Cloud) | - |

   Everything else has working defaults for Docker. `PUBLIC_CONTACTS` must match the support
   phone/email in `data/vilnius-parking-data/static/general_info.md`. In production
   (`APP_ENV=production`) the API refuses to start with a short, default or shared admin/MCP key.

6. Start the stack:

   ```bash
   docker compose --env-file .env.development up --build -d
   ```

7. Run migrations:

   ```bash
   docker compose --env-file .env.development run --rm --no-deps api uv run --frozen --no-sync alembic upgrade head
   ```

8. Ingest the static docs (Milvus) and dynamic data (Postgres):

   ```bash
   docker compose --env-file .env.development run --rm --no-deps api python -m scripts.ingest_static
   # after upgrading from an older version (hybrid search, chunk headings), or after removing/shortening docs:
   docker compose --env-file .env.development run --rm --no-deps api python -m scripts.ingest_static --recreate
   docker compose --env-file .env.development run --rm --no-deps api python -m scripts.ingest_dynamic
   ```

## Usage

### Links

With the stack running locally on the default ports (`API_PORT=8000`, `MCP_SERVER_PUBLISHED_PORT=9000`):

| What | Link |
| --- | --- |
| Chat UI | [http://localhost:8000/](http://localhost:8000/) |
| Admin console (approve / reject reservations) | [http://localhost:8000/admin](http://localhost:8000/admin) - enter your `ADMIN_API_KEY` |
| Admin console, Streamlit version (`make admin-ui`) | [http://localhost:8501/](http://localhost:8501/) |
| API docs - try the chat and admin endpoints (Swagger UI) | [http://localhost:8000/docs](http://localhost:8000/docs) |
| API docs (ReDoc) | [http://localhost:8000/redoc](http://localhost:8000/redoc) |
| Chat API | `POST` [http://localhost:8000/chat](http://localhost:8000/chat), `GET` [http://localhost:8000/chat/updates](http://localhost:8000/chat/updates) |
| Admin API (`X-Admin-Key` header) | `GET` [http://localhost:8000/admin/reservations](http://localhost:8000/admin/reservations), `POST` `http://localhost:8000/admin/reservations/{id}/decision` |
| API health | [http://localhost:8000/health](http://localhost:8000/health) |
| MCP server health | [http://localhost:9000/health](http://localhost:9000/health) |
| LangSmith traces (if enabled) | [https://smith.langchain.com](https://smith.langchain.com) (EU: [https://eu.smith.langchain.com](https://eu.smith.langchain.com)) |

The chat API needs the session cookie that `GET /` sets, so open the Chat UI first (or call `/`
in the same client). Request and response formats: [HTTP API](#http-api).

### How to use

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

- **Tests**: see [Running tests](#running-tests).

- **RAG evaluation** (Precision@K/Recall@K against a hand-labelled golden set, plus
  retrieval and end-to-end `/chat` latency; requires the static docs to already be ingested).
  The golden set has 30 questions: 15 single-fact ones, plus 15 harder ones (paraphrases,
  multi-fact, out of scope, private data, prompt injection, not in the docs) that the system was
  never tuned on. Retrieval is scored on the 23 with a relevant doc. Latest results:
  [`evals/report.md`](evals/report.md) - at K=3, Recall@3 0.96, MRR 0.91, Precision@1 0.87,
  Precision@3 0.62 (the 15 single-fact questions alone: Recall@3 1.00, MRR 0.92). Measured on
  those 15 questions: hybrid search took Recall from 0.93 (semantic only) to 1.00; giving every chunk its heading path
  ("# VilniusPark Old Town / ## Facilities") took Precision@1 from 0.73 to 0.87. The report also
  compares K=2..5 (`--k-range`) and recommends the smallest K with the best recall: K=2 misses
  an answer, K=4 and 5 add nothing over 3, so `RETRIEVER_K=3`.

  `SEARCH_TYPE` picks the search: `hybrid` (default), `semantic` (embeddings only) or `keyword`
  (BM25 only, no embedding call). On the 15 single-fact questions at K=3:

  | `SEARCH_TYPE` | Recall@3 | MRR | Precision@1 |
  | --- | --- | --- | --- |
  | `hybrid` | 1.00 | 0.92 | 0.87 |
  | `semantic` | 0.93 | 0.86 | 0.80 |
  | `keyword` | 1.00 | 0.83 | 0.73 |

  ```bash
  make eval          # retrieval accuracy and latency
  make eval-chat     # + end-to-end /chat latency (calls the chatbot)
  SEARCH_TYPE=semantic make eval   # the same with another search type
  ```

- The evaluations run on your machine, not in Docker, so they reach Milvus on `localhost`
  (the `make` targets set `VECTOR_DB_URI=http://localhost:19530`; override with
  `make eval EVAL_MILVUS_URI=...`). Running the scripts directly with only `.env.development`
  fails: its `milvus:19530` hostname exists only inside Docker.
- **Evidently AI evaluation** (LLM-as-judge on the real chatbot's answers: context quality,
  faithfulness, completeness, refusals, PII in answers, answer length; needs the stack running
  and costs a few cents of OpenAI calls). Writes an interactive
  [`evals/evidently_report.html`](evals/evidently_report.html), every verdict with the judge's
  reasoning in `evals/evidently_rows.csv`, and a summary appended to `evals/report.md`. Its first
  run found a real bug: the chatbot added the brand name to search queries, which pulled in every
  doc's generic intro chunk instead of the relevant one - 6 of 15 answers wrongly said the docs
  didn't cover the question. Search queries now drop the brand name. Latest run, on all 30
  questions: 77% pass every check (single-fact 100%, paraphrases 20%, multi-fact 33%, out of scope,
  private data and injection 100%), no personal data in any answer. Reviewing each failure: 2 are
  real bugs (the Airport's EV bay count, where the database and the docs disagree; Užupis's 2
  accessible spaces missed) and 5 are judge errors - e.g. "the docs don't mention a monthly pass"
  isn't counted as declining, and the judges see only the eval's top-3 chunks, not the ones the
  chatbot actually searched. Completeness and PII
  use custom judges: Evidently's built-in Completeness wants an answer to repeat every retrieved
  chunk (it failed "10 km/h" for not listing EV charger penalties; 7%), and its PII judge flagged
  "a booking needs your name and car number".

  ```bash
  make eval-evidently
  ```

- **Load tests** (Locust, against a running stack) for each component: the chatbot
  (`ChatUser`), administrator confirmation (`AdminUser`) and MCP recording (`MCPUser`).
  Procedure and latest results: [`load_tests/report.md`](load_tests/report.md).

  ```bash
  make load-test        # the chatbot; see the report for AdminUser / MCPUser
  ```

## Running tests

The suite needs no running stack: integration and e2e tests start their own throwaway
Postgres in Docker (testcontainers) - so **Docker must be running** - and the e2e tests also
start a real MCP server process. Redis, Milvus and the LLM are stubbed or faked; no OpenAI
key or network is needed.

```bash
make test                          # everything (same as `uv run pytest -s`)
uv run pytest                      # everything, quieter
uv run pytest tests/unit           # unit tests only - fast, no Docker needed
uv run pytest tests/integration    # integration tests (Postgres in Docker)
uv run pytest tests/e2e            # full flow: chat -> admin approval -> MCP file
uv run pytest -k availability      # tests whose name matches a keyword
uv run pytest tests/unit/test_parking_graph.py::test_cancel_clears_history_and_draft  # one test
uv run pytest --cov=app --cov=mcp_server --cov-report=term-missing   # with coverage
```

| Folder | What it covers |
| --- | --- |
| `tests/unit/` | the graph and agents (with a scripted fake model), tools, guardrails, middleware, MCP server, config, rate limits, parsers, ingestion, evaluation metrics |
| `tests/integration/` | repositories, availability and the no-overbooking lock, admin and chat APIs, chat cleanup, the dynamic-data import - against real Postgres |
| `tests/e2e/` | the whole reservation flow through the real graph, Postgres checkpointer and MCP server |

CI (`.github/workflows/ci.yml`) runs Ruff and the full suite on every push and pull request.
Latest local run: 168 tests passing, 88% coverage of `app/` and `mcp_server/`.

## Checking approved reservations

Each approval is stored in two places:

- **Postgres**, `reservations` table - approved ones have `status = 'approved'` and an
  `approved_at` time (stored in UTC):

  ```sql
  SELECT first_name, surname, car_number, starts_at, ends_at, status, approved_at
  FROM reservations ORDER BY created_at DESC LIMIT 10;
  ```

- **The MCP server's file** (`Name | Car Number | Reservation Period | Approval Time`), which
  lives in the `mcp_reservations` Docker volume, not in the project folder:

  ```bash
  # read it
  docker compose --env-file .env.development exec mcp-server cat /data/approved_reservations.txt
  # watch new approvals arrive (Ctrl+C to stop)
  docker compose --env-file .env.development exec mcp-server tail -f /data/approved_reservations.txt
  # copy it into the project folder (a snapshot - run again to refresh; it is git-ignored)
  docker compose --env-file .env.development cp mcp-server:/data/approved_reservations.txt ./approved_reservations.txt
  ```

## Deployment

The same Docker Compose stack runs in production; what changes is configuration.

1. **Environment file.** Create `.env.production` from `.env.example` and set
   `APP_ENV=production` - this marks the session cookie `Secure`, so serve the app over
   HTTPS (behind a TLS-terminating reverse proxy such as nginx, Caddy or a cloud load
   balancer). Use strong, unique secrets for `ADMIN_API_KEY`, `MCP_API_KEY` and
   `POSTGRES_PASSWORD`; never commit the file. In production the API and the MCP server
   refuse to start with a missing, default (`local-dev-only-change-me`), short (< 16
   characters) or repetitive (< 8 different characters) admin/MCP key, or with the same key
   for both - generate each with e.g. `openssl rand -hex 32`. Migrations and ingestion
   scripts don't check them, since they never use these keys.
2. **Network exposure.** Publish only the API port. Postgres, Redis, Milvus and the MCP
   server bind to `BIND_ADDRESS` (keep `127.0.0.1`) and are reached by the API over the
   Compose network. The admin endpoints are protected by `X-Admin-Key` (constant-time
   check, lockout after repeated failures); restrict `/admin` at the proxy too if possible.
3. **Start and migrate** on every deploy - migrations are idempotent:

   ```bash
   docker compose --env-file .env.production up -d --build
   docker compose --env-file .env.production run --rm --no-deps api \
       uv run --frozen --no-sync alembic upgrade head
   ```

4. **Data.** Ingest the static docs (Milvus) and dynamic data (Postgres) once, and again
   whenever they change (`scripts.ingest_static` / `scripts.ingest_dynamic`, see Setup).
   Back up the `postgres_data` volume (reservations and conversation checkpoints) and the
   MCP server's `mcp_reservations` volume (the approved-reservations file).
5. **Verify.** `GET /health` should report the API, Postgres and Redis as `ok`; the startup
   log shows whether LangSmith tracing is enabled.

Every service has `restart: unless-stopped` and a health check (the API's polls `/health`), so
a crashed container, a hung API or a host reboot brings the stack back on its own. The API and
MCP server containers run as an unprivileged `app` user, not root. Upgrading an existing
deployment from a root-run image: give the MCP volume to that user once -
`docker run --rm -v <project>_mcp_reservations:/data alpine chown -R 10001:10001 /data`.

CI (`.github/workflows/ci.yml`) runs Ruff and the full test suite on every push and pull
request.

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
scripts/         Milvus/Postgres data ingestion, LangGraph diagram export
evals/           RAG evaluation (golden set, script, latest report)
load_tests/      Locust load tests per component, seed script, latest report
tests/           unit / integration / e2e tests
docs/            the course specification, the presentation, LangGraph diagrams (graph.md)
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
