"""Stage 2: REST API for the administrator to review and decide on pending
reservations. ui/admin_app.py (Streamlit) is the console that calls this.
"""

from uuid import UUID

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.ai.graph import ApprovalPipeline
from app.config import settings
from app.db.models import Reservation
from app.db.repositories.reservation import ReservationRepository
from app.dependencies import SQLSessionDependency
from app.logger import get_logger
from app.schemas.admin import DecisionRequest, PendingReservation

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


async def require_admin(x_admin_key: str = Header(default="")) -> None:
    if x_admin_key != settings.admin_api_key.get_secret_value():
        logger.warning("Rejected an admin request with an invalid X-Admin-Key")
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.get(
    "/reservations",
    response_model=list[PendingReservation],
    dependencies=[Depends(require_admin)],
)
async def list_pending(
    request: Request,
    session: SQLSessionDependency,
) -> list[PendingReservation]:
    pipeline: ApprovalPipeline = request.app.state.approval_pipeline

    rows = await session.scalars(
        sa.select(Reservation).where(Reservation.status == "pending").order_by(Reservation.created_at)
    )

    return [
        PendingReservation(
            id=reservation.id,
            parking_id=reservation.parking_id,
            space_type=reservation.space_type,
            first_name=reservation.first_name,
            surname=reservation.surname,
            car_number=reservation.car_number,
            starts_at=reservation.starts_at,
            ends_at=reservation.ends_at,
            created_at=reservation.created_at,
            message=await pipeline.pending_message(reservation.id),
        )
        for reservation in rows.all()
    ]


@router.post(
    "/reservations/{reservation_id}/decision",
    dependencies=[Depends(require_admin)],
)
async def decide(
    reservation_id: UUID,
    payload: DecisionRequest,
    request: Request,
    session: SQLSessionDependency,
) -> dict:
    reservations = ReservationRepository(session)
    reservation = await reservations.get_by_id(reservation_id)

    if reservation is None or reservation.status != "pending":
        raise HTTPException(status_code=404, detail="No pending reservation with that id")

    logger.info("Admin decision received for reservation %s: %s", reservation_id, payload.decision)

    pipeline: ApprovalPipeline = request.app.state.approval_pipeline
    await pipeline.resume(reservation_id, payload.decision, payload.note)

    return {"status": "ok"}
