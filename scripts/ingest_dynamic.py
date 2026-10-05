import asyncio
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings
from app.db.models import (
    CapacityOverride,
    Parking,
    ParkingSpaceType,
    Price,
    SpecialHours,
    WorkingHours,
)
from app.db.repositories.postgres import PostgresRepository


def build_parkings() -> list[dict]:
    values = [
        (
            "old-town", "VilniusPark Old Town",
            "Senamiestis", "54.6840", "25.2860", 180, 60,
        ),
        (
            "station", "VilniusPark Station",
            "Naujamiestis", "54.6705", "25.2835", 120, 40,
        ),
        (
            "business-center", "VilniusPark Business Center",
            "Šnipiškės", "54.6960", "25.2770", 350, 120,
        ),
        (
            "airport", "VilniusPark Airport Long-Stay",
            "Kirtimai", "54.6385", "25.2850", 400, 200,
        ),
        (
            "uzupis", "VilniusPark Užupis",
            "Užupis", "54.6805", "25.2960", 40, 15,
        ),
    ]

    return [
        {
            "slug": slug,
            "name": name,
            "district": district,
            "latitude": Decimal(latitude),
            "longitude": Decimal(longitude),
            "total_spaces": total,
            "bookable_spaces": bookable,
        }
        for (
            slug,
            name,
            district,
            latitude,
            longitude,
            total,
            bookable,
        ) in values
    ]


async def upsert_parkings(
    session: AsyncSession,
    rows: list[dict],
) -> dict[str, int]:
    """Match by unique slug and preserve existing numeric IDs."""

    table = Parking.__table__
    parking_ids: dict[str, int] = {}

    for row in rows:
        statement = insert(table).values(**row)

        statement = statement.on_conflict_do_update(
            index_elements=[table.c.slug],
            set_={
                field: statement.excluded[field]
                for field in row
                if field != "slug"
            },
        ).returning(table.c.id)

        result = await session.execute(statement)
        parking_ids[row["slug"]] = result.scalar_one()

    return parking_ids


def build_space_types(
    parkings: list[dict],
    parking_ids: dict[str, int],
) -> list[dict]:
    # Fictional inventory: all EV spaces are bookable.
    ev_counts = {
        "old-town": 6,
        "station": 4,
        "business-center": 12,
        "airport": 20,
        "uzupis": 2,
    }

    rows = []

    for parking in parkings:
        slug = parking["slug"]
        total = parking["total_spaces"]
        bookable = parking["bookable_spaces"]
        ev = ev_counts[slug]

        if not 0 <= ev <= bookable <= total:
            raise ValueError(f"Invalid capacity for {slug}")

        rows.extend([
            {
                "parking_id": parking_ids[slug],
                "space_type": "standard",
                "total_spaces": total - ev,
                "bookable_spaces": bookable - ev,
            },
            {
                "parking_id": parking_ids[slug],
                "space_type": "ev",
                "total_spaces": ev,
                "bookable_spaces": ev,
            },
        ])

    return rows


def build_working_hours(
    parking_ids: dict[str, int],
) -> list[dict]:
    schedules = {
        "old-town": [("06:00", "24:00")] * 7,
        "station": [("00:00", "24:00")] * 7,
        "business-center": (
            [("06:00", "22:00")] * 5
            + [("08:00", "20:00"), (None, None)]
        ),
        "airport": [("00:00", "24:00")] * 7,
        "uzupis": (
            [("07:00", "21:00")] * 5
            + [("08:00", "21:00"), (None, None)]
        ),
    }

    return [
        {
            "parking_id": parking_ids[slug],
            "day_of_week": day,
            "open_time": opening,
            "close_time": closing,
            "is_closed": opening is None,
        }
        for slug, intervals in schedules.items()
        for day, (opening, closing) in enumerate(intervals, start=1)
    ]


def build_special_hours(
    parking_ids: dict[str, int],
) -> list[dict]:
    values = [
        (
            "business-center", "2026-11-01",
            None, None, "All Saints' Day",
        ),
        (
            "uzupis", "2026-11-01",
            None, None, "All Saints' Day",
        ),
        (
            "old-town", "2026-11-01",
            "08:00", "20:00", "All Saints' Day – reduced hours",
        ),
        (
            "old-town", "2026-12-24",
            "06:00", "18:00", "Christmas Eve – early closing",
        ),
        (
            "business-center", "2026-12-24",
            None, None, "Christmas Eve",
        ),
        (
            "business-center", "2026-12-25",
            None, None, "Christmas Day",
        ),
        (
            "business-center", "2026-12-26",
            None, None, "Second day of Christmas",
        ),
        (
            "uzupis", "2026-12-24",
            "08:00", "15:00", "Christmas Eve – early closing",
        ),
        (
            "uzupis", "2026-12-25",
            None, None, "Christmas Day",
        ),
        (
            "uzupis", "2026-12-26",
            None, None, "Second day of Christmas",
        ),
        (
            "business-center", "2027-01-01",
            None, None, "New Year's Day",
        ),
        (
            "uzupis", "2027-01-01",
            None, None, "New Year's Day",
        ),
    ]

    return [
        {
            "parking_id": parking_ids[slug],
            "date": date.fromisoformat(day),
            "open_time": opening,
            "close_time": closing,
            "is_closed": opening is None,
            "note": note,
        }
        for slug, day, opening, closing, note in values
    ]


def build_prices(
    parking_ids: dict[str, int],
) -> list[dict]:
    # EV tariffs are fictional examples; electricity is not included.
    # slug, type, day_type, hourly, daily, valid_from, valid_to
    values = [
        (
            "old-town", "standard", "weekday",
            "2.80", "24.00", "2026-01-01", "2026-06-30",
        ),
        (
            "old-town", "standard", "weekday",
            "3.00", "25.00", "2026-07-01", None,
        ),
        (
            "old-town", "standard", "weekend",
            "2.50", "20.00", "2026-01-01", None,
        ),
        (
            "old-town", "ev", "weekday",
            "3.80", "30.00", "2026-01-01", "2026-06-30",
        ),
        (
            "old-town", "ev", "weekday",
            "4.00", "32.00", "2026-07-01", None,
        ),
        (
            "old-town", "ev", "weekend",
            "3.50", "27.00", "2026-01-01", None,
        ),
        (
            "station", "standard", "weekday",
            "2.00", "15.00", "2026-01-01", None,
        ),
        (
            "station", "standard", "weekend",
            "2.00", "15.00", "2026-01-01", None,
        ),
        (
            "station", "ev", "weekday",
            "3.00", "22.00", "2026-01-01", None,
        ),
        (
            "station", "ev", "weekend",
            "3.00", "22.00", "2026-01-01", None,
        ),
        (
            "business-center", "standard", "weekday",
            "2.50", "18.00", "2026-01-01", None,
        ),
        (
            "business-center", "standard", "weekend",
            "1.50", "10.00", "2026-01-01", None,
        ),
        (
            "business-center", "ev", "weekday",
            "3.50", "25.00", "2026-01-01", None,
        ),
        (
            "business-center", "ev", "weekend",
            "2.50", "17.00", "2026-01-01", None,
        ),
        (
            "airport", "standard", "weekday",
            "1.50", "12.00", "2026-01-01", None,
        ),
        (
            "airport", "standard", "weekend",
            "1.50", "12.00", "2026-01-01", None,
        ),
        (
            "airport", "ev", "weekday",
            "2.50", "19.00", "2026-01-01", None,
        ),
        (
            "airport", "ev", "weekend",
            "2.50", "19.00", "2026-01-01", None,
        ),
        (
            "uzupis", "standard", "weekday",
            "1.80", "12.00", "2026-01-01", None,
        ),
        (
            "uzupis", "standard", "weekend",
            "1.50", "10.00", "2026-01-01", None,
        ),
        (
            "uzupis", "ev", "weekday",
            "2.80", "19.00", "2026-01-01", None,
        ),
        (
            "uzupis", "ev", "weekend",
            "2.50", "17.00", "2026-01-01", None,
        ),
    ]

    return [
        {
            "parking_id": parking_ids[slug],
            "space_type": space_type,
            "day_type": day_type,
            "price_per_hour": Decimal(hourly),
            "daily_max": Decimal(daily),
            "currency": "EUR",
            "valid_from": date.fromisoformat(start),
            "valid_to": date.fromisoformat(end) if end else None,
        }
        for (
            slug,
            space_type,
            day_type,
            hourly,
            daily,
            start,
            end,
        ) in values
    ]


def build_capacity_overrides(
    parking_ids: dict[str, int],
) -> list[dict]:
    vilnius = ZoneInfo("Europe/Vilnius")

    starts_at = datetime(
        2026, 10, 20, 6, 0,
        tzinfo=vilnius,
    ).astimezone(timezone.utc)

    ends_at = datetime(
        2026, 10, 20, 14, 0,
        tzinfo=vilnius,
    ).astimezone(timezone.utc)

    # Sample maintenance restriction: only one level remains usable.
    capacities = {
        "standard": 18,
        "ev": 2,
    }

    return [
        {
            "id": uuid5(
                NAMESPACE_URL,
                f"vilniuspark:old-town:maintenance:2026-10-20:{space_type}",
            ),
            "parking_id": parking_ids["old-town"],
            "space_type": space_type,
            "starts_at": starts_at,
            "ends_at": ends_at,
            "bookable_capacity": capacity,
            "note": "Demo maintenance: only one level is usable",
        }
        for space_type, capacity in capacities.items()
    ]


async def remove_old_maintenance_hours(
    session: AsyncSession,
    parking_ids: dict[str, int],
) -> None:
    """Remove the previous demo schedule replaced by capacity overrides."""

    statement = sa.delete(SpecialHours).where(
        SpecialHours.parking_id == parking_ids["old-town"],
        SpecialHours.date == date(2026, 10, 20),
        SpecialHours.note == "Level -3 floor maintenance",
    )

    await session.execute(statement)


async def main() -> None:
    engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )

    try:
        async with AsyncSession(
            engine,
            expire_on_commit=False,
        ) as session:
            async with session.begin():
                parkings = build_parkings()
                parking_ids = await upsert_parkings(session, parkings)

                await remove_old_maintenance_hours(
                    session,
                    parking_ids,
                )

                # Capacity types must exist before prices and overrides.
                datasets = [
                    (
                        ParkingSpaceType,
                        build_space_types(parkings, parking_ids),
                    ),
                    (
                        WorkingHours,
                        build_working_hours(parking_ids),
                    ),
                    (
                        SpecialHours,
                        build_special_hours(parking_ids),
                    ),
                    (
                        Price,
                        build_prices(parking_ids),
                    ),
                    (
                        CapacityOverride,
                        build_capacity_overrides(parking_ids),
                    ),
                ]

                for model, rows in datasets:
                    repository = PostgresRepository(
                        session=session,
                        model=model,
                    )

                    await repository.upsert_many(rows)

            print(f"Processed {len(parkings)} parkings")

            for model, rows in datasets:
                print(f"Processed {len(rows)} {model.__tablename__}")

            print(f"Parking IDs: {parking_ids}")
            print("Transaction committed.")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())