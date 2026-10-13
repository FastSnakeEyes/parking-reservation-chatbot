"""app/ai/chat_cleanup.py against a real AsyncPostgresSaver: only chat
threads idle past the cutoff are deleted - never active chats, never a
conversation still waiting for the administrator, and never non-chat threads."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import sqlalchemy as sa
from app.ai.chat_cleanup import delete_idle_chat_threads
from app.db.models import Reservation
from app.db.repositories.reservation import ReservationRepository
from app.schemas.chat import BookingInput
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy.ext.asyncio import AsyncSession
from tests.conftest import FakeLock


async def _write_checkpoint(checkpointer, thread_id: str, age: timedelta) -> None:
    checkpoint = empty_checkpoint()
    checkpoint["ts"] = (datetime.now(UTC) - age).isoformat()

    await checkpointer.aput(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
        checkpoint,
        {},
        {},
    )


async def _exists(checkpointer, thread_id: str) -> bool:
    return await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}}) is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deletes_only_idle_chat_threads(engine, checkpointer, real_parking):
    idle_chat = f"chat:{uuid4()}"
    active_chat = f"chat:{uuid4()}"
    awaiting_admin = f"chat:{uuid4()}"
    other_thread = str(uuid4())  # not a chat thread

    await _write_checkpoint(checkpointer, idle_chat, age=timedelta(hours=2))
    await _write_checkpoint(checkpointer, active_chat, age=timedelta(hours=2))
    await _write_checkpoint(checkpointer, active_chat, age=timedelta(minutes=1))  # newest one counts
    await _write_checkpoint(checkpointer, awaiting_admin, age=timedelta(days=2))
    await _write_checkpoint(checkpointer, other_thread, age=timedelta(days=30))

    # The admin hasn't decided yet: the conversation must survive to be resumed.
    starts_at = datetime.now(UTC) + timedelta(days=1)
    booking = BookingInput(
        parking_id=real_parking.id,
        space_type="standard",
        first_name="Jane",
        surname="Doe",
        car_number="ABC123",
        starts_at=starts_at,
        ends_at=starts_at + timedelta(hours=2),
    )
    reservation_id = uuid4()
    # Committed for real (the cleanup query runs on its own connection), then removed.
    async with AsyncSession(engine) as session:
        await ReservationRepository(session).submit_pending(reservation_id, booking, awaiting_admin)

    try:
        deleted = await delete_idle_chat_threads(engine, checkpointer, lambda _: FakeLock(), idle_seconds=3600)
    finally:
        async with AsyncSession(engine) as session:
            await session.execute(sa.delete(Reservation).where(Reservation.id == reservation_id))
            await session.commit()

    assert deleted >= 1
    assert not await _exists(checkpointer, idle_chat)
    assert await _exists(checkpointer, active_chat)
    assert await _exists(checkpointer, awaiting_admin)
    assert await _exists(checkpointer, other_thread)


class HeldLock(FakeLock):
    """A thread whose chat turn is running right now."""

    async def acquire(self, *args, **kwargs):
        return False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skips_a_thread_whose_turn_is_running(engine, checkpointer):
    busy_chat = f"chat:{uuid4()}"
    await _write_checkpoint(checkpointer, busy_chat, age=timedelta(hours=2))

    await delete_idle_chat_threads(
        engine, checkpointer, lambda thread_id: HeldLock() if thread_id == busy_chat else FakeLock(), idle_seconds=3600
    )

    assert await _exists(checkpointer, busy_chat)  # left for the next sweep
