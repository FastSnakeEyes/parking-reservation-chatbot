"""Chat threads in LangGraph's Postgres checkpointer - the database side of the
idle-thread cleanup (app/ai/chat_cleanup.py).

The checkpoint tables are created and owned by AsyncPostgresSaver.setup(),
not by this app's models or migrations. `checkpoints` is only described here,
as a lightweight Core table (deliberately outside Base.metadata, so nothing
ever tries to create it), for reading which threads went idle. Deleting a
thread goes through the checkpointer's public adelete_thread(), not SQL.
"""

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import CHAT_THREAD_PREFIX
from app.db.models import Reservation
from app.enums import ReservationStatus

# One row per checkpoint; a thread's latest one carries its last activity in checkpoint->'ts'.
checkpoints = sa.table(
    "checkpoints",
    sa.column("thread_id", sa.Text),
    sa.column("checkpoint", JSONB),
)


class ChatThreadRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _idle_threads(cutoff: datetime) -> sa.Select:
        """Chat threads whose newest checkpoint is older than `cutoff`, except
        those waiting for an administrator's decision on a pending reservation
        - the admin API resumes them by reservations.conversation_id, however
        long that takes."""
        last_activity = sa.func.max(sa.cast(checkpoints.c.checkpoint["ts"].astext, sa.DateTime(timezone=True)))
        awaiting_admin = sa.select(Reservation.conversation_id).where(
            Reservation.status == ReservationStatus.PENDING,
            Reservation.conversation_id.is_not(None),
        )

        return (
            sa.select(checkpoints.c.thread_id)
            .where(
                checkpoints.c.thread_id.startswith(CHAT_THREAD_PREFIX, autoescape=True),
                checkpoints.c.thread_id.not_in(awaiting_admin),
            )
            .group_by(checkpoints.c.thread_id)
            .having(last_activity < cutoff)
        )

    async def idle_thread_ids(self, cutoff: datetime) -> list[str]:
        """Every chat thread idle since before `cutoff`."""
        result = await self.session.execute(self._idle_threads(cutoff))
        return list(result.scalars().all())

    async def is_idle(self, thread_id: str, cutoff: datetime) -> bool:
        """The same check for one thread - repeated under its lock just before
        deleting it, in case a chat turn touched it in the meantime."""
        statement = self._idle_threads(cutoff).where(checkpoints.c.thread_id == thread_id)
        return (await self.session.execute(statement)).first() is not None
