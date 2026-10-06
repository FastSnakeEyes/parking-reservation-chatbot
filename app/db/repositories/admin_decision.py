from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AdminDecision
from app.db.repositories.postgres import PostgresRepository


class DecisionConflict(Exception):
    """Raised when a reservation already has a recorded admin decision."""


class AdminDecisionRepository(PostgresRepository[AdminDecision]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=AdminDecision)

    async def record(
        self,
        reservation_id: UUID,
        version: int,
        decision: str,
        note: str | None,
        idempotency_key: str,
    ) -> AdminDecision:
        """Flush-only (no commit): composes inside ParkingService's atomic
        decision+enqueue transaction. UNIQUE(reservation_id) at the database
        level is what actually prevents two decisions on one reservation,
        even under concurrent decide() calls - this just translates that
        into a typed exception instead of a bare IntegrityError."""
        entity = AdminDecision(
            reservation_id=reservation_id,
            reservation_version=version,
            decision=decision,
            note=note,
            idempotency_key=idempotency_key,
        )
        self.session.add(entity)

        try:
            await self.session.flush()
        except IntegrityError as exc:
            raise DecisionConflict(f"Reservation {reservation_id} already has a decision") from exc

        return entity
