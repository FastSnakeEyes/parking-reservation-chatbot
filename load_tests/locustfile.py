"""Load test for the chat endpoint.

Usage:
    uv run locust -f load_tests/locustfile.py --host http://localhost:8000

Each simulated user first loads "/" (to get the session cookie, same as a
real browser), then repeatedly asks questions over POST /chat. Locust's
HttpUser keeps cookies across requests, so each user acts as one
consistent chat session - matching the per-session rate limit in
app/dependencies.py (cookie_rate_limit).
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    case["question"] for case in json.loads((Path(__file__).parent.parent / "evals/golden_qa.json").read_text())
]


class ChatUser(HttpUser):
    wait_time = between(2, 5)  # stays under cookie_rate_limit's default 2 req / 10s

    def on_start(self):
        self.client.get("/")

    @task
    def ask_a_question(self):
        self.client.post("/chat", json={"question": random.choice(QUESTIONS)})  # noqa: S311 (not cryptographic)
