"""ChatThreadRepository (app/db/repositories/chat_thread.py) against a real
AsyncPostgresSaver: which chat threads count as idle."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import sqlalchemy as sa
from app.db.models import Reservation
from app.db.repositories.chat_thread import ChatThreadRepository
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy.ext.asyncio import AsyncSession

CUTOFF = datetime.now(UTC) - timedelta(hours=1)


async def _checkpoint(checkpointer, thread_id: str, age: timedelta) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["ts"] = (datetime.now(UTC) - age).isoformat()
    await checkpointer.aput({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}, checkpoint, {}, {})


async def _idle_ids(engine) -> list[str]:
    async with AsyncSession(engine) as session:
        return await ChatThreadRepository(session).idle_thread_ids(CUTOFF)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_idle_thread_ids_uses_each_threads_newest_checkpoint(engine, checkpointer):
    idle = f"chat:{uuid4()}"
    active = f"chat:{uuid4()}"
    not_a_chat = str(uuid4())
    await _checkpoint(checkpointer, idle, age=timedelta(hours=2))
    await _checkpoint(checkpointer, active, age=timedelta(hours=2))
    await _checkpoint(checkpointer, active, age=timedelta(minutes=5))  # active recently
    await _checkpoint(checkpointer, not_a_chat, age=timedelta(days=3))

    idle_ids = await _idle_ids(engine)

    assert idle in idle_ids
    assert active not in idle_ids
    assert not_a_chat not in idle_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_thread_waiting_for_the_admin_is_never_idle(engine, checkpointer, real_parking):
    waiting = f"chat:{uuid4()}"
    await _checkpoint(checkpointer, waiting, age=timedelta(days=2))

    starts_at = datetime.now(UTC) + timedelta(days=4)
    reservation_id = uuid4()
    async with AsyncSession(engine) as session:
        await ReservationRepository(session).submit_pending(
            reservation_id,
            BookingInput(
                parking_id=real_parking.id,
                space_type="standard",
                first_name="Jane",
                surname="Doe",
                car_number="ABC123",
                starts_at=starts_at,
                ends_at=starts_at + timedelta(hours=1),
            ),
            conversation_id=waiting,
        )

    try:
        async with AsyncSession(engine) as session:
            threads = ChatThreadRepository(session)
            idle_ids = await threads.idle_thread_ids(CUTOFF)
            waiting_is_idle = await threads.is_idle(waiting, CUTOFF)
    finally:
        async with AsyncSession(engine) as session:
            await session.execute(sa.delete(Reservation).where(Reservation.id == reservation_id))
            await session.commit()

    assert waiting not in idle_ids
    assert waiting_is_idle is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_is_idle_rechecks_a_single_thread(engine, checkpointer):
    thread = f"chat:{uuid4()}"
    await _checkpoint(checkpointer, thread, age=timedelta(hours=2))

    async with AsyncSession(engine) as session:
        threads = ChatThreadRepository(session)
        before = await threads.is_idle(thread, CUTOFF)
        await _checkpoint(checkpointer, thread, age=timedelta(seconds=1))  # a chat turn just ran
        after = await threads.is_idle(thread, CUTOFF)

    assert (before, after) == (True, False)
