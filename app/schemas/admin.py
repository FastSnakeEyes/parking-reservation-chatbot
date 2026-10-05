from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


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
    decision: Literal["approved", "rejected"]
    note: str | None = None
