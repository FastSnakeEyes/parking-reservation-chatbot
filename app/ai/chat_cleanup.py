"""Deletes abandoned chat threads from the Postgres checkpointer.

The chat cookie expires after cookie_ttl seconds without a request, after
which its thread can never be reached again - but AsyncPostgresSaver has
no TTL, so without this job those threads would stay forever.

Only `chat:`-prefixed threads are touched: the approval graph shares the
checkpointer (threads keyed by reservation_id) and its threads must survive
however long the administrator takes to decide.
"""

import asyncio

import sqlalchemy as sa
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncEngine

from app.ai.parking_chat import CHAT_THREAD_PREFIX
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# A thread is idle when even its newest checkpoint is older than the cutoff.
IDLE_THREADS_SQL = sa.text(
    """
    SELECT thread_id
    FROM checkpoints
    WHERE thread_id LIKE :prefix
    GROUP BY thread_id
    HAVING max((checkpoint ->> 'ts')::timestamptz) < now() - make_interval(secs => :idle_seconds)
    """
)


async def delete_idle_chat_threads(engine: AsyncEngine, checkpointer: BaseCheckpointSaver, idle_seconds: int) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(
            IDLE_THREADS_SQL, {"prefix": f"{CHAT_THREAD_PREFIX}%", "idle_seconds": idle_seconds}
        )
        thread_ids = result.scalars().all()

    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)

    if thread_ids:
        logger.info("Deleted %d idle chat threads", len(thread_ids))

    return len(thread_ids)


async def run_chat_cleanup(engine: AsyncEngine, checkpointer: BaseCheckpointSaver) -> None:
    """Background loop started by app/lifespan.py; cancelled on shutdown.
    Safe to run in every worker at once - deleting a thread is idempotent."""
    while True:
        try:
            await delete_idle_chat_threads(engine, checkpointer, settings.cookie_ttl)
        except Exception:
            logger.exception("Chat thread cleanup failed; retrying next interval")

        await asyncio.sleep(settings.chat_cleanup_interval)
