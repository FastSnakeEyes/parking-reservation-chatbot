"""WorkflowService: durable commands, idempotency, and operation status.

Thin façade over CommandRepository - this is what GraphRunner (Phase 2)
calls to enqueue work instead of running the graph inline in the request.
"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Command
from app.db.repositories.command import CommandRepository


class WorkflowService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.commands = CommandRepository(session)

    async def enqueue(
        self,
        idempotency_key: str,
        command_type: str,
        thread_id: str,
        payload: dict,
    ) -> Command:
        command = await self.commands.enqueue(idempotency_key, command_type, thread_id, payload)
        await self.session.commit()

        return command

    async def get_status(self, command_id: UUID) -> Command | None:
        return await self.commands.get_by_id(command_id)
