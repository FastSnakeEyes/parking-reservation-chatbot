from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

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


class AdminReplyDecision(BaseModel):
    """parse_admin_reply's structured output (app/ai/parsers.py)."""

    approved: bool = Field(description="True only if the administrator clearly approved the reservation")
    reason: str = Field(description="The administrator's reason or comment; empty string if none")


class DecisionResponse(BaseModel):
    status: ReservationStatus  # APPROVED or REJECTED
