"""The customer-facing notice ParkingChat.pending_decision_notice delivers
once the administrator has decided on their last reservation - otherwise
the customer has no way to learn the outcome."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from app.ai.parking_chat import ParkingChat
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput


def _chat(db_session, fake_redis) -> ParkingChat:
    return ParkingChat(
        session=db_session,
        redis=fake_redis,
        vector_repository=None,
        llm=None,
        approval_pipeline=None,
    )


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
async def test_no_notice_without_a_last_reservation(db_session, fake_redis):
    chat = _chat(db_session, fake_redis)

    assert await chat.pending_decision_notice({}) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_notice_while_still_pending(db_session, fake_redis, sample_parking):
    reservation_id = uuid4()
    await ReservationRepository(db_session).submit_pending(reservation_id, _booking(sample_parking.id))

    chat = _chat(db_session, fake_redis)
    state = {"last_reservation_id": str(reservation_id)}

    assert await chat.pending_decision_notice(state) is None
    assert state["last_reservation_id"] == str(reservation_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notice_on_approval_and_clears_state(db_session, fake_redis, sample_parking):
    reservation_id = uuid4()
    reservations = ReservationRepository(db_session)
    await reservations.submit_pending(reservation_id, _booking(sample_parking.id))
    await reservations.update(
        reservation_id,
        {"status": "recorded", "approved_at": datetime.now(UTC)},
    )

    chat = _chat(db_session, fake_redis)
    state = {"last_reservation_id": str(reservation_id)}

    notice = await chat.pending_decision_notice(state)

    assert notice is not None
    assert "approved" in notice
    assert str(reservation_id) in notice
    assert state["last_reservation_id"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notice_on_rejection(db_session, fake_redis, sample_parking):
    reservation_id = uuid4()
    reservations = ReservationRepository(db_session)
    await reservations.submit_pending(reservation_id, _booking(sample_parking.id))
    await reservations.update(reservation_id, {"status": "rejected"})

    chat = _chat(db_session, fake_redis)
    state = {"last_reservation_id": str(reservation_id)}

    notice = await chat.pending_decision_notice(state)

    assert notice is not None
    assert "rejected" in notice
