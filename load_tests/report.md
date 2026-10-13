# Load test report

Stage 4 asks for load tests of each component: the chatbot in interactive dialogue mode,
the administrator confirmation, and the MCP server's recording. All three are user classes
in [`locustfile.py`](locustfile.py); this report is from one run of each.

**Run:** 2026-10-13, local Docker Compose stack on an Apple M1 Pro laptop; chat model
`gpt-4o-mini` (the `LLM_MODEL` default) via the OpenAI API. Locust headless, `--only-summary`.

| Component | Users | Duration | Requests | Failures | Median | p95 | Max | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chatbot: `POST /chat` | 3 | 75 s | 25 | **0** | 3.1 s | 4.4 s | 5.8 s | 0.34 req/s |
| Admin: `GET /admin/reservations` | 5 | 45 s | 145 | **0** | 13 ms | 24 ms | 160 ms | ~3.2 req/s |
| Admin: `POST /admin/reservations/{id}/decision` | (same 5) | | 29 | **0** | 10 ms | 16 ms | 24 ms | ~0.6 req/s |
| MCP: `tools/call record_reservation` | 20 | 45 s | 1112 | **0** | 6 ms | 13 ms | 30 ms | 26 req/s |

## Findings

- **Chatbot latency is the model.** `/chat` time is dominated by the OpenAI calls (one or
  more per turn, plus tool calls); the app's own work is in the milliseconds, as the admin
  and MCP numbers show. The chat page shows a "processing" status while waiting.
- **No lost writes under concurrency.** The MCP run made 1112 tool calls from 20 concurrent
  clients, and the reservations file had exactly 1112 lines afterwards, each in the spec's
  `Name | Car Number | Reservation Period | Approval Time` format - the server's write lock
  serializes appends.
- **The admin API stays fast while deciding:** listing pending reservations (each with the
  admin agent's summary read back from the paused conversation) and deciding are tens of
  milliseconds.
- **Bug found and fixed by this load test.** The first chatbot run failed 2 of 23 requests
  with HTTP 500: once a session's history grew past `HISTORY_MAX_TOKENS`, summarization
  reused the agents' model, whose `parallel_tool_calls=False` OpenAI rejects on a request
  without tools. Summarization now has its own model client (`app/lifespan.py`); the re-run
  above has 0 failures, and `tests/unit/test_parking_graph.py` covers the wiring.
- A first re-run also showed 3 x 429: the test paced requests 2-5 s apart, which breaks the
  2-requests-per-10-s session limit when an answer is instant. That was the test, not the
  app; `ChatUser` now waits 5.5-8 s.

## How to reproduce

```bash
# Chatbot (calls the real LLM - keep it short)
uv run --group eval locust -f load_tests/locustfile.py ChatUser \
    --host http://localhost:8000 --headless -u 3 -r 3 -t 75s

# Administrator confirmation: seed "LoadTest" pending reservations, run, clean up.
# Rejects by default; LOADTEST_APPROVE=1 approves (and records through the MCP server).
uv run python -m load_tests.seed_reservations --count 40
ADMIN_API_KEY=... uv run --group eval locust -f load_tests/locustfile.py AdminUser \
    --host http://localhost:8000 --headless -u 5 -r 5 -t 45s
uv run python -m load_tests.seed_reservations --cleanup

# MCP recording: against a separate server instance, so the real reservations file stays clean
MCP_API_KEY=loadtest-token RESERVATIONS_FILE=/tmp/loadtest.txt MCP_PORT=9100 \
    uv run python -m mcp_server.server &
MCP_API_KEY=loadtest-token MCP_LOAD_URL=http://localhost:9100 \
    uv run --group eval locust -f load_tests/locustfile.py MCPUser --headless -u 20 -r 10 -t 45s
```

Running the seed script from the host needs the published Postgres port, e.g.
`POSTGRES_HOST=localhost POSTGRES_PORT=5432`.
