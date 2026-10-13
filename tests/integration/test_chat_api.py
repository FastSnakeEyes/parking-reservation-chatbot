"""The chat router's HTTP behavior (the graph itself is stubbed)."""

from uuid import uuid4

import pytest
from app.api.routers.chat import router as chat_router
from app.config import settings
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


class StubGraph:
    async def updates(self, session_id):
        return {"status": "waiting_for_admin", "notice": None, "reservation_status": None}


@pytest.mark.asyncio
async def test_polling_for_updates_renews_the_session_cookie():
    """A customer waiting on the page longer than cookie_ttl must keep their
    session, or the approval pop-up would never arrive."""
    app = FastAPI()
    app.include_router(chat_router)
    app.state.parking_graph = StubGraph()
    session_id = str(uuid4())

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", cookies={settings.cookie_name: session_id}
    ) as client:
        response = await client.get("/chat/updates")

    assert response.status_code == 200
    assert response.json()["status"] == "waiting_for_admin"
    set_cookie = response.headers["set-cookie"]
    assert f"{settings.cookie_name}={session_id}" in set_cookie
    assert f"Max-Age={settings.cookie_ttl}" in set_cookie
