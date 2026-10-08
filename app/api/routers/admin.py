"""Stage 2: REST API for the administrator to review and decide on pending
reservations. ui/admin_app.py (Streamlit) is the console that calls this.

A submitted reservation's conversation waits at ParkingGraph's
admin_decision interrupt; a decision resumes that conversation's thread.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException

from app.ai.tools import ChatContext
from app.config import settings
from app.dependencies import ParkingGraphDependency, ParkingServiceDependency
from app.logger import get_logger
from app.schemas.admin import DecisionRequest, DecisionResponse, PendingReservation

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
    service: ParkingServiceDependency,
    graph: ParkingGraphDependency,
) -> list[PendingReservation]:
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
            message=(await graph.pending_message(reservation.conversation_id) if reservation.conversation_id else None),
        )
        for reservation in await service.list_pending()
    ]


@router.post(
    "/reservations/{reservation_id}/decision",
    response_model=DecisionResponse,
    dependencies=[Depends(require_admin)],
)
async def decide(
    reservation_id: UUID,
    payload: DecisionRequest,
    service: ParkingServiceDependency,
    graph: ParkingGraphDependency,
) -> DecisionResponse:
    reservation = await service.get_pending(reservation_id)

    if reservation is None or reservation.conversation_id is None:
        raise HTTPException(status_code=404, detail="No pending reservation with that id")

    logger.info("Admin decision received for reservation %s: %s", reservation_id, payload.decision)

    try:
        status = await graph.resume(
            reservation.conversation_id,
            reservation_id,
            payload.decision,
            payload.note,
            ChatContext(service=service),
        )
    except LookupError as exc:
        raise HTTPException(status_code=409, detail="That reservation's conversation is not awaiting approval") from exc

    return DecisionResponse(status=status)
