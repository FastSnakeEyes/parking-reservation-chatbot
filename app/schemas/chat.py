from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class SourceResponse(BaseModel):
    label: int
    source: str
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    # "waiting_for_admin" while the conversation is paused for the
    # administrator's approval of a submitted reservation.
    status: Literal["complete", "waiting_for_admin"] = "complete"
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)


class BookingInput(BaseModel):
    parking_id: int = Field(gt=0)
    space_type: Literal["standard", "ev"]

    first_name: str = Field(min_length=1, max_length=100)
    surname: str = Field(min_length=1, max_length=100)
    car_number: str = Field(min_length=1, max_length=20)

    starts_at: datetime
    ends_at: datetime

    @field_validator(
        "first_name",
        "surname",
        "car_number",
        mode="before",
    )
    @classmethod
    def clean_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("starts_at", "ends_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("A timezone offset is required")

        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_period(self):
        if self.ends_at <= self.starts_at:
            raise ValueError("End must be after start")

        return self
