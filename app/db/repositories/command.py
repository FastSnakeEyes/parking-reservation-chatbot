from datetime import UTC, datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Command
from app.db.repositories.postgres import PostgresRepository


class CommandRepository(PostgresRepository[Command]):
    """WorkflowService's durable outbox: GraphRunner enqueues a command
    instead of running the graph inline in the request; the in-process
    worker claims and executes it."""

    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=Command)

    async def enqueue(
        self,
        idempotency_key: str,
        command_type: str,
        thread_id: str,
        payload: dict,
    ) -> Command:
        """Idempotent: a retried enqueue with the same key returns the
        existing command instead of creating a duplicate. Flush-only (no
        commit) so this composes inside a caller's larger transaction, e.g.
        ParkingService recording a decision and enqueueing its resume
        command atomically."""
        statement = (
            insert(Command.__table__)
            .values(
                id=uuid4(),
                idempotency_key=idempotency_key,
                command_type=command_type,
                thread_id=thread_id,
                payload=payload,
            )
            .on_conflict_do_nothing(index_elements=[Command.__table__.c.idempotency_key])
        )
        await self.session.execute(statement)
        await self.session.flush()

        return await self.session.scalar(sa.select(Command).where(Command.idempotency_key == idempotency_key))

    async def claim_next(self) -> Command | None:
        """Atomic claim-and-commit: self-contained, meant to be called by
        the worker loop in its own session per iteration. `SKIP LOCKED` lets
        multiple claimers (today: one worker, but this makes the pattern
        correct if that ever changes) each get a different pending command
        instead of blocking on each other."""
        statement = (
            sa.select(Command)
            .where(Command.status == "pending")
            .order_by(Command.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        command = await self.session.scalar(statement)

        if command is None:
            return None

        command.status = "claimed"
        command.claimed_at = datetime.now(UTC)
        await self.session.commit()

        return command

    async def complete(self, command_id: UUID, result: dict | None) -> None:
        command = await self.get_by_id(command_id)

        if command is None:
            return

        command.status = "succeeded"
        command.result = result
        command.completed_at = datetime.now(UTC)
        await self.session.commit()

    async def fail(self, command_id: UUID, error: str) -> None:
        command = await self.get_by_id(command_id)

        if command is None:
            return

        command.status = "failed"
        command.error = error
        command.completed_at = datetime.now(UTC)
        await self.session.commit()
