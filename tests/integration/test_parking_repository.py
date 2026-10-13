"""ParkingRepository's public parking data (app/db/repositories/parking.py).
sample_parking: 10 standard spaces (5 bookable), open 00:00-23:59 daily,
1 EUR/hour and a 10 EUR daily maximum on weekdays and weekends."""

from datetime import date

import pytest
from app.db.repositories.parking import ParkingRepository

A_MONDAY = date(2026, 10, 19)
A_SATURDAY = date(2026, 10, 24)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_public_returns_id_name_and_district(db_session, sample_parking):
    parkings = await ParkingRepository(db_session).list_public()

    assert {"id": sample_parking.id, "name": "Test Parking", "district": "Center"} in parkings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_info_has_hours_prices_and_capacity(db_session, sample_parking):
    info = await ParkingRepository(db_session).get_public_info(sample_parking.id, A_MONDAY)

    assert info["name"] == "Test Parking"
    assert info["hours"]["is_closed"] is False
    assert float(info["prices"]["standard"]["price_per_hour"]) == 1
    assert info["space_types"] == [{"space_type": "standard", "total_spaces": 10, "bookable_spaces": 5}]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_public_info_for_an_unknown_parking_is_none(db_session):
    assert await ParkingRepository(db_session).get_public_info(999999, A_MONDAY) is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("on_date", [A_MONDAY, A_SATURDAY], ids=["weekday", "weekend"])
async def test_get_price_picks_the_day_type(db_session, sample_parking, on_date):
    price = await ParkingRepository(db_session).get_price(sample_parking.id, on_date, "standard")

    assert price["day_type"] == ("weekend" if on_date == A_SATURDAY else "weekday")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_price_rejects_an_unknown_space_type(db_session, sample_parking):
    with pytest.raises(ValueError, match="space_type"):
        await ParkingRepository(db_session).get_price(sample_parking.id, A_MONDAY, "motorbike")
