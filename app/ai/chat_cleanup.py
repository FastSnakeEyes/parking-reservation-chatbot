"""Deletes abandoned chat threads from the Postgres checkpointer.

The chat cookie expires after cookie_ttl seconds without a request, after
which its thread can never be reached again - but AsyncPostgresSaver has
no TTL, so without this job those threads would stay forever.

Only `chat:`-prefixed threads are touched, and never one still waiting for
an administrator's decision on its reservation. Each thread is deleted under
the same lock chat turns hold (ParkingGraph.lock), after re-checking it is
still idle, so a turn that starts at the cutoff can't lose its history.

Which threads are idle comes from ChatThreadRepository
(app/db/repositories/chat_thread.py); this module only orchestrates.
"""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import settings
from app.db.repositories.chat_thread import ChatThreadRepository
from app.logger import get_logger

logger = get_logger(__name__)


async def delete_idle_chat_threads(
    engine: AsyncEngine,
    checkpointer: BaseCheckpointSaver,
    thread_lock: Callable[[str], object],
    idle_seconds: int,
) -> int:
    """Delete idle chat threads; returns how many. `thread_lock` is
    ParkingGraph.lock: a thread whose lock is held (a turn is running) is
    skipped until the next sweep."""
    cutoff = datetime.now(UTC) - timedelta(seconds=idle_seconds)

    # A short session per query: no transaction stays open while the sweep
    # waits on locks or deletes threads.
    async with AsyncSession(engine) as session:
        thread_ids = await ChatThreadRepository(session).idle_thread_ids(cutoff)

    deleted = 0
    for thread_id in thread_ids:
        lock = thread_lock(thread_id)
        if not await lock.acquire(blocking=False):
            continue

        try:
            async with AsyncSession(engine) as session:
                still_idle = await ChatThreadRepository(session).is_idle(thread_id, cutoff)
            if still_idle:
                await checkpointer.adelete_thread(thread_id)
                deleted += 1
        finally:
            await lock.release()

    if deleted:
        logger.info("Deleted %d idle chat threads", deleted)

    return deleted


async def run_chat_cleanup(
    engine: AsyncEngine, checkpointer: BaseCheckpointSaver, thread_lock: Callable[[str], object]
) -> None:
    """Background loop started by app/lifespan.py; cancelled on shutdown.
    Safe to run in every worker at once - each thread is deleted under its lock."""
    while True:
        try:
            await delete_idle_chat_threads(engine, checkpointer, thread_lock, settings.cookie_ttl)
        except Exception:
            logger.exception("Chat thread cleanup failed; retrying next interval")

        await asyncio.sleep(settings.chat_cleanup_interval)
