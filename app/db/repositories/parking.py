from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.enums import DayType, SpaceType


class ParkingRepository(PostgresRepository[Parking]):
    def __init__(self, session: AsyncSession):
        super().__init__(session=session, model=Parking)
        self.timezone = ZoneInfo(settings.timezone)

    async def list_public(self) -> list[dict[str, Any]]:
        """Every parking's public id, name and district."""
        result = await self.session.execute(sa.select(Parking.id, Parking.name, Parking.district).order_by(Parking.id))
        return [dict(row) for row in result.mappings().all()]

    async def get_working_hours(
        self,
        parking_id: int,
        on_date: date,
    ) -> dict[str, Any] | None:
        # SQL string casts preserve PostgreSQL's 24:00.
        statement = sa.select(
            sa.cast(
                SpecialHours.open_time,
                sa.String,
            ).label("open_time"),
            sa.cast(
                SpecialHours.close_time,
                sa.String,
            ).label("close_time"),
            SpecialHours.is_closed,
            SpecialHours.note,
        ).where(
            SpecialHours.parking_id == parking_id,
            SpecialHours.date == on_date,
        )

        result = await self.session.execute(statement)
        special = result.mappings().one_or_none()

        if special is not None:
            return {
                **dict(special),
                "source": "special_hours",
            }

        statement = sa.select(
            sa.cast(
                WorkingHours.open_time,
                sa.String,
            ).label("open_time"),
            sa.cast(
                WorkingHours.close_time,
                sa.String,
            ).label("close_time"),
            WorkingHours.is_closed,
        ).where(
            WorkingHours.parking_id == parking_id,
            WorkingHours.day_of_week == on_date.isoweekday(),
        )

        result = await self.session.execute(statement)
        weekly = result.mappings().one_or_none()

        if weekly is None:
            return None

        return {
            **dict(weekly),
            "note": None,
            "source": "working_hours",
        }

    async def get_price(
        self,
        parking_id: int,
        on_date: date,
        space_type: str,
    ) -> dict[str, Any] | None:
        if space_type not in set(SpaceType):
            raise ValueError(f"space_type must be one of {', '.join(SpaceType)}")

        day_type = DayType.WEEKEND if on_date.isoweekday() >= 6 else DayType.WEEKDAY

        statement = sa.select(
            Price.space_type,
            Price.day_type,
            Price.price_per_hour,
            Price.daily_max,
            Price.currency,
            Price.valid_from,
            Price.valid_to,
        ).where(
            Price.parking_id == parking_id,
            Price.space_type == space_type,
            Price.day_type == day_type,
            Price.valid_from <= on_date,
            sa.or_(
                Price.valid_to.is_(None),
                Price.valid_to >= on_date,
            ),
        )

        result = await self.session.execute(statement)

        # Overlapping active tariffs raise an error.
        price = result.mappings().one_or_none()

        return dict(price) if price is not None else None

    async def get_space_types(
        self,
        parking_id: int,
    ) -> list[dict[str, Any]]:
        statement = (
            sa.select(
                ParkingSpaceType.space_type,
                ParkingSpaceType.total_spaces,
                ParkingSpaceType.bookable_spaces,
            )
            .where(
                ParkingSpaceType.parking_id == parking_id,
            )
            .order_by(ParkingSpaceType.space_type)
        )

        result = await self.session.execute(statement)

        return [dict(row) for row in result.mappings().all()]

    async def get_capacity_overrides(
        self,
        parking_id: int,
        on_date: date,
    ) -> list[dict[str, Any]]:
        # Calculate each boundary in local time to handle DST.
        day_start = datetime.combine(
            on_date,
            time.min,
            tzinfo=self.timezone,
        ).astimezone(UTC)

        day_end = datetime.combine(
            on_date + timedelta(days=1),
            time.min,
            tzinfo=self.timezone,
        ).astimezone(UTC)

        statement = (
            sa.select(
                CapacityOverride.id,
                CapacityOverride.space_type,
                CapacityOverride.starts_at,
                CapacityOverride.ends_at,
                CapacityOverride.bookable_capacity,
                CapacityOverride.note,
            )
            .where(
                CapacityOverride.parking_id == parking_id,
                CapacityOverride.starts_at < day_end,
                CapacityOverride.ends_at > day_start,
            )
            .order_by(
                CapacityOverride.starts_at,
                CapacityOverride.space_type,
                CapacityOverride.id,
            )
        )

        result = await self.session.execute(statement)

        return [dict(row) for row in result.mappings().all()]

    async def get_public_info(
        self,
        parking_id: int,
        on_date: date,
    ) -> dict[str, Any] | None:
        parking = await self.get_by_id(parking_id)

        if parking is None:
            return None

        # Queries sharing one AsyncSession run sequentially.
        hours = await self.get_working_hours(
            parking_id,
            on_date,
        )
        space_types = await self.get_space_types(
            parking_id,
        )
        overrides = await self.get_capacity_overrides(
            parking_id,
            on_date,
        )

        prices: dict[str, dict[str, Any] | None] = {}

        for capacity in space_types:
            space_type = capacity["space_type"]

            prices[space_type] = await self.get_price(
                parking_id=parking_id,
                on_date=on_date,
                space_type=space_type,
            )

        return {
            "id": parking.id,
            "slug": parking.slug,
            "name": parking.name,
            "district": parking.district,
            "latitude": parking.latitude,
            "longitude": parking.longitude,
            "total_spaces": parking.total_spaces,
            "bookable_spaces": parking.bookable_spaces,
            "date": on_date.isoformat(),
            "timezone": settings.timezone,
            "hours": hours,
            "prices": prices,
            "space_types": space_types,
            "capacity_overrides": overrides,
        }
