from datetime import UTC, datetime

import pytest
from app.schemas.chat import BookingInput
from pydantic import ValidationError

VALID = {
    "parking_id": 1,
    "space_type": "standard",
    "first_name": "Jane",
    "surname": "Doe",
    "car_number": "ABC123",
    "starts_at": datetime(2030, 1, 1, 10, tzinfo=UTC),
    "ends_at": datetime(2030, 1, 1, 12, tzinfo=UTC),
}


def test_valid_booking_passes():
    booking = BookingInput.model_validate(VALID)

    assert booking.space_type == "standard"
    assert booking.first_name == "Jane"


def test_end_before_start_rejected():
    payload = {**VALID, "ends_at": VALID["starts_at"]}

    with pytest.raises(ValidationError):
        BookingInput.model_validate(payload)


def test_naive_datetime_rejected():
    payload = {**VALID, "starts_at": datetime(2030, 1, 1, 10)}

    with pytest.raises(ValidationError):
        BookingInput.model_validate(payload)


def test_blank_name_rejected():
    payload = {**VALID, "first_name": ""}

    with pytest.raises(ValidationError):
        BookingInput.model_validate(payload)


def test_names_are_stripped():
    payload = {**VALID, "first_name": "  Jane  "}

    booking = BookingInput.model_validate(payload)

    assert booking.first_name == "Jane"
