from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.api.routers.admin import router as admin_router
from app.db.repositories.reservation import ReservationRepository
from app.dependencies import get_sql_session
from app.schemas.chat import BookingInput
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

THREAD_ID = "chat:test-session"


class StubGraph:
    """Stands in for ParkingGraph: a conversation paused for every reservation."""

    def __init__(self):
        self.resumed = []

    async def pending_message(self, thread_id):
        return "drafted message"

    async def resume(self, thread_id, reservation_id, decision, note, context):
        self.resumed.append((thread_id, reservation_id, decision, note))
        return "recorded" if decision == "approved" else "rejected"


def build_test_app(db_session, graph) -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router)
    app.state.parking_graph = graph
    app.state.vector_repository = None  # only the user agent's tools need it

    async def override_session():
        yield db_session

    app.dependency_overrides[get_sql_session] = override_session

    return app


def _booking(parking_id: int) -> BookingInput:
    return BookingInput(
        parking_id=parking_id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=datetime.now(UTC) + timedelta(days=1),
        ends_at=datetime.now(UTC) + timedelta(days=1, hours=2),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_pending_requires_admin_key(db_session):
    app = build_test_app(db_session, StubGraph())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/admin/reservations")

    assert response.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_pending_returns_reservations_with_drafted_message(db_session, sample_parking):
    reservation_id = uuid4()
    await ReservationRepository(db_session).submit_pending(reservation_id, _booking(sample_parking.id), THREAD_ID)

    app = build_test_app(db_session, StubGraph())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/admin/reservations",
            headers={"X-Admin-Key": "test-admin-key"},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == str(reservation_id)
    assert body[0]["message"] == "drafted message"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_decision_unknown_reservation_returns_404(db_session):
    app = build_test_app(db_session, StubGraph())

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/admin/reservations/{uuid4()}/decision",
            headers={"X-Admin-Key": "test-admin-key"},
            json={"decision": "approved"},
        )

    assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_decision_resumes_the_reservations_conversation(db_session, sample_parking):
    reservation_id = uuid4()
    await ReservationRepository(db_session).submit_pending(reservation_id, _booking(sample_parking.id), THREAD_ID)

    graph = StubGraph()
    app = build_test_app(db_session, graph)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/admin/reservations/{reservation_id}/decision",
            headers={"X-Admin-Key": "test-admin-key"},
            json={"decision": "approved"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "recorded"}
    assert graph.resumed == [(THREAD_ID, reservation_id, "approved", None)]
