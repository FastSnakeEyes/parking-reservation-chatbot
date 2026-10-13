"""Stage 2: REST API for the administrator to review and decide on pending
reservations. ui/admin_app.py (Streamlit) is the console that calls this.

A submitted reservation's conversation waits for the human-in-the-loop admin
agent's review (app/ai/graph.py); a decision resumes that conversation's thread.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.ai.tools import ChatContext
from app.dependencies import ParkingGraphDependency, ParkingServiceDependency, require_admin
from app.enums import ApprovalDecision
from app.logger import get_logger
from app.schemas.admin import DecisionRequest, DecisionResponse, PendingReservation

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/reservations",
    response_model=list[PendingReservation],
    dependencies=[Depends(require_admin)],
)
async def list_pending(
    parking_service: ParkingServiceDependency,
    parking_graph: ParkingGraphDependency,
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
            message=(
                await parking_graph.pending_message(reservation.conversation_id)
                if reservation.conversation_id
                else None
            ),
        )
        for reservation in await parking_service.list_pending()
    ]


@router.post(
    "/reservations/{reservation_id}/decision",
    response_model=DecisionResponse,
    dependencies=[Depends(require_admin)],
)
async def decide(
    reservation_id: UUID,
    payload: DecisionRequest,
    parking_service: ParkingServiceDependency,
    parking_graph: ParkingGraphDependency,
) -> DecisionResponse:
    reservation = await parking_service.get_pending(reservation_id)

    if reservation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No pending reservation with that id")

    logger.info("Admin decision received for reservation %s: %s", reservation_id, payload.decision)

    if reservation.conversation_id is not None:
        try:
            reservation_status = await parking_graph.resume(
                reservation.conversation_id,
                reservation_id,
                payload.decision,
                payload.note,
                ChatContext(parking_service=parking_service),
            )
            return DecisionResponse(status=reservation_status)
        except LookupError:
            pass

    # No conversation waiting on it (e.g. submitted before reservations had a
    # conversation_id): decide directly, so a pending reservation can always be
    # closed. The customer gets no chat notice - there's no conversation to send it to.
    logger.warning("Reservation %s has no conversation awaiting approval; deciding directly", reservation_id)
    reservation_status = await parking_service.record_decision(
        reservation_id,
        approved=payload.decision == ApprovalDecision.APPROVED,
        note=payload.note,
    )
    return DecisionResponse(status=reservation_status)
