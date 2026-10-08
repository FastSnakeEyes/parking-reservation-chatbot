"""The /admin HTML console (app/main.py) and its static assets.

Uses the real app object (not a fresh FastAPI()) since the /admin route and
the static mount are wired there, not on the admin router alone.
"""

import pytest
from app.dependencies import get_sql_session
from app.main import app
from httpx import ASGITransport, AsyncClient


class StubGraph:
    async def pending_message(self, thread_id):
        return None


@pytest.fixture
def admin_console_client(db_session):
    async def override_session():
        yield db_session

    app.dependency_overrides[get_sql_session] = override_session
    app.state.parking_graph = StubGraph()
    app.state.vector_repository = None

    try:
        yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    finally:
        app.dependency_overrides.pop(get_sql_session, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_console_page_renders(admin_console_client):
    async with admin_console_client as client:
        response = await client.get("/admin")

    assert response.status_code == 200
    assert "admin.js" in response.text
    assert "admin.css" in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_console_static_assets_are_served(admin_console_client):
    async with admin_console_client as client:
        js_response = await client.get("/static/admin.js")
        css_response = await client.get("/static/admin.css")

    assert js_response.status_code == 200
    assert css_response.status_code == 200
