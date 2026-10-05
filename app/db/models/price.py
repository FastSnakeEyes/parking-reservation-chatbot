from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.parking_space_type import ParkingSpaceType


class Price(Base):
    __tablename__ = "prices"

    parking_id: Mapped[int] = mapped_column(
        sa.Integer,
        primary_key=True,
    )
    space_type: Mapped[str] = mapped_column(
        sa.String(20),
        primary_key=True,
    )
    day_type: Mapped[str] = mapped_column(
        sa.String(10),
        primary_key=True,
    )
    valid_from: Mapped[date] = mapped_column(
        sa.Date,
        primary_key=True,
    )
    valid_to: Mapped[date | None] = mapped_column(sa.Date)

    price_per_hour: Mapped[Decimal] = mapped_column(sa.Numeric(10, 2))
    daily_max: Mapped[Decimal] = mapped_column(sa.Numeric(10, 2))
    currency: Mapped[str] = mapped_column(
        sa.String(3),
        server_default=sa.text("'EUR'"),
    )

    capacity_type: Mapped[ParkingSpaceType] = relationship(
        back_populates="prices",
        lazy="raise",
    )

    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["parking_id", "space_type"],
            [
                "parking_space_types.parking_id",
                "parking_space_types.space_type",
            ],
            name="fk_prices_space_type",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "day_type IN ('weekday', 'weekend')",
            name="ck_prices_day_type",
        ),
        sa.CheckConstraint(
            "price_per_hour >= 0 AND daily_max >= 0",
            name="ck_prices_amounts",
        ),
        sa.CheckConstraint(
            "valid_to IS NULL OR valid_to >= valid_from",
            name="ck_prices_validity",
        ),
    )