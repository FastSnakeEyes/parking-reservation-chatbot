from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.enums import ApprovalDecision, ReservationStatus


class PendingReservation(BaseModel):
    id: UUID
    parking_id: int
    space_type: str
    first_name: str
    surname: str
    car_number: str
    starts_at: datetime
    ends_at: datetime
    created_at: datetime
    message: str | None = None


class DecisionRequest(BaseModel):
    decision: ApprovalDecision
    note: str | None = None


class DecisionResponse(BaseModel):
    status: ReservationStatus  # RECORDED or REJECTED
