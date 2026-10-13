"""Deletes abandoned chat threads from the Postgres checkpointer.

The chat cookie expires after cookie_ttl seconds without a request, after
which its thread can never be reached again - but AsyncPostgresSaver has
no TTL, so without this job those threads would stay forever.

Only `chat:`-prefixed threads are touched, and never one still waiting for
an administrator's decision on its reservation. Each thread is deleted under
the same lock chat turns hold (ParkingGraph.lock), after re-checking it is
still idle, so a turn that starts at the cutoff can't lose its history.
"""

import asyncio
from collections.abc import Callable

import sqlalchemy as sa
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import settings
from app.constants import CHAT_THREAD_PREFIX
from app.logger import get_logger

logger = get_logger(__name__)

# A thread is idle when even its newest checkpoint is older than the cutoff.
# Threads paused for an administrator's decision are kept however long it takes:
# the admin API resumes them by reservations.conversation_id.
_IDLE_THREADS = """
    SELECT thread_id
    FROM checkpoints
    WHERE thread_id LIKE :prefix {only_thread}
      AND thread_id NOT IN (
          SELECT conversation_id FROM reservations
          WHERE status = 'pending' AND conversation_id IS NOT NULL
      )
    GROUP BY thread_id
    HAVING max((checkpoint ->> 'ts')::timestamptz) < now() - make_interval(secs => :idle_seconds)
"""
IDLE_THREADS_SQL = sa.text(_IDLE_THREADS.format(only_thread=""))
# The same check for one thread, repeated under its lock just before deleting it.
STILL_IDLE_SQL = sa.text(_IDLE_THREADS.format(only_thread="AND thread_id = :thread_id"))


async def delete_idle_chat_threads(
    engine: AsyncEngine,
    checkpointer: BaseCheckpointSaver,
    thread_lock: Callable[[str], object],
    idle_seconds: int,
) -> int:
    """Delete idle chat threads; returns how many. `thread_lock` is
    ParkingGraph.lock: a thread whose lock is held (a turn is running) is
    skipped until the next sweep."""
    params = {"prefix": f"{CHAT_THREAD_PREFIX}%", "idle_seconds": idle_seconds}

    async with engine.connect() as conn:
        thread_ids = (await conn.execute(IDLE_THREADS_SQL, params)).scalars().all()

    deleted = 0
    for thread_id in thread_ids:
        lock = thread_lock(thread_id)
        if not await lock.acquire(blocking=False):
            continue

        try:
            async with engine.connect() as conn:
                still_idle = (await conn.execute(STILL_IDLE_SQL, {**params, "thread_id": thread_id})).first()
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
