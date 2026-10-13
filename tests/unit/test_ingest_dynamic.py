"""The dynamic-data builders (scripts/ingest_dynamic.py): the parking data
they produce must be internally consistent, or the bot would quote missing
prices, wrong hours or impossible capacities."""

from datetime import UTC, date, datetime, time

import pytest
from scripts.ingest_dynamic import (
    build_capacity_overrides,
    build_parkings,
    build_prices,
    build_space_types,
    build_special_hours,
    build_working_hours,
)

PARKINGS = build_parkings()
PARKING_IDS = {parking["slug"]: index for index, parking in enumerate(PARKINGS, start=1)}
SPACE_TYPES = build_space_types(PARKINGS, PARKING_IDS)
A_DATE_WITH_DATA = date(2026, 10, 19)  # inside the price data's validity ranges


def test_parkings_have_unique_slugs_and_valid_capacity():
    assert len({parking["slug"] for parking in PARKINGS}) == len(PARKINGS) == 5
    assert all(0 < parking["bookable_spaces"] <= parking["total_spaces"] for parking in PARKINGS)


def test_space_types_split_each_parkings_capacity_exactly():
    for parking in PARKINGS:
        rows = [row for row in SPACE_TYPES if row["parking_id"] == PARKING_IDS[parking["slug"]]]

        assert {row["space_type"] for row in rows} == {"standard", "ev"}
        assert sum(row["total_spaces"] for row in rows) == parking["total_spaces"]
        assert sum(row["bookable_spaces"] for row in rows) == parking["bookable_spaces"]
        assert all(0 <= row["bookable_spaces"] <= row["total_spaces"] for row in rows)


def test_space_types_reject_more_ev_spaces_than_bookable():
    too_small = [{**PARKINGS[0], "bookable_spaces": 1}]  # old-town has 6 EV spaces

    with pytest.raises(ValueError, match="Invalid capacity"):
        build_space_types(too_small, PARKING_IDS)


def test_working_hours_cover_every_day_and_closed_days_have_no_times():
    rows = build_working_hours(PARKING_IDS)

    for parking_id in PARKING_IDS.values():
        days = sorted(row["day_of_week"] for row in rows if row["parking_id"] == parking_id)
        assert days == [1, 2, 3, 4, 5, 6, 7]

    for row in rows:
        assert row["is_closed"] == (row["open_time"] is None)
        assert (row["open_time"] is None) == (row["close_time"] is None)

    business_center_sunday = next(
        row for row in rows if row["parking_id"] == PARKING_IDS["business-center"] and row["day_of_week"] == 7
    )
    assert business_center_sunday["is_closed"] is True


def test_special_hours_are_consistent():
    rows = build_special_hours(PARKING_IDS)

    assert rows
    for row in rows:
        assert row["is_closed"] == (row["open_time"] is None)
        assert row["note"]
    keys = [(row["parking_id"], row["date"]) for row in rows]
    assert len(keys) == len(set(keys))  # one special schedule per parking and date


def _price_valid_on(row: dict, on_date: date) -> bool:
    return row["valid_from"] <= on_date and (row["valid_to"] is None or on_date <= row["valid_to"])


def test_every_space_type_has_exactly_one_price_per_day_type():
    """No 'No applicable tariff' answers, and no ambiguous double prices."""
    prices = build_prices(PARKING_IDS)

    for space_type in SPACE_TYPES:
        for day_type in ("weekday", "weekend"):
            matching = [
                row
                for row in prices
                if row["parking_id"] == space_type["parking_id"]
                and row["space_type"] == space_type["space_type"]
                and row["day_type"] == day_type
                and _price_valid_on(row, A_DATE_WITH_DATA)
            ]
            assert len(matching) == 1, (space_type, day_type, matching)
            assert 0 < matching[0]["price_per_hour"] <= matching[0]["daily_max"]


def test_capacity_overrides_are_converted_to_utc_and_fit_the_space_type():
    rows = build_capacity_overrides(PARKING_IDS)
    bookable = {(row["parking_id"], row["space_type"]): row["bookable_spaces"] for row in SPACE_TYPES}

    for row in rows:
        # 06:00-14:00 Vilnius time on 2026-10-20 (summer time, UTC+3) is 03:00-11:00 UTC.
        assert row["starts_at"] == datetime.combine(date(2026, 10, 20), time(3, 0), tzinfo=UTC)
        assert row["ends_at"] == datetime.combine(date(2026, 10, 20), time(11, 0), tzinfo=UTC)
        assert 0 <= row["bookable_capacity"] <= bookable[(row["parking_id"], row["space_type"])]


def test_capacity_override_ids_are_stable_across_runs():
    """Deterministic ids are what make re-running the import an update, not a duplicate."""
    assert [row["id"] for row in build_capacity_overrides(PARKING_IDS)] == [
        row["id"] for row in build_capacity_overrides(PARKING_IDS)
    ]
