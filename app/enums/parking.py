from enum import StrEnum


class SpaceType(StrEnum):
    STANDARD = "standard"
    EV = "ev"


class DayType(StrEnum):
    """Which tariff applies on a date (prices.day_type)."""

    WEEKDAY = "weekday"
    WEEKEND = "weekend"
