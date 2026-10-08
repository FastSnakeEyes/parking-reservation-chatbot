"""app/ai/chat_cleanup.py against a real AsyncPostgresSaver: only chat
threads idle past the cutoff are deleted - never active chats, and never
the approval graph's reservation threads, however old."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from app.ai.chat_cleanup import delete_idle_chat_threads
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@pytest_asyncio.fixture
async def checkpointer(postgres_container):
    url = postgres_container.replace("postgresql+asyncpg://", "postgresql://")

    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()
        yield saver


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
async def test_deletes_only_idle_chat_threads(engine, checkpointer):
    idle_chat = f"chat:{uuid4()}"
    active_chat = f"chat:{uuid4()}"
    old_approval = str(uuid4())  # approval graph threads are keyed by bare reservation_id

    await _write_checkpoint(checkpointer, idle_chat, age=timedelta(hours=2))
    await _write_checkpoint(checkpointer, active_chat, age=timedelta(hours=2))
    await _write_checkpoint(checkpointer, active_chat, age=timedelta(minutes=1))  # newest one counts
    await _write_checkpoint(checkpointer, old_approval, age=timedelta(days=30))

    deleted = await delete_idle_chat_threads(engine, checkpointer, idle_seconds=3600)

    assert deleted >= 1
    assert not await _exists(checkpointer, idle_chat)
    assert await _exists(checkpointer, active_chat)
    assert await _exists(checkpointer, old_approval)
