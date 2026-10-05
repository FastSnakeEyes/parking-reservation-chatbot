from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.capacity_override import CapacityOverride
    from app.db.models.parking_space_type import ParkingSpaceType
    from app.db.models.price import Price
    from app.db.models.reservation import Reservation
    from app.db.models.special_hours import SpecialHours
    from app.db.models.working_hours import WorkingHours


class Parking(Base):
    __tablename__ = "parkings"

    id: Mapped[int] = mapped_column(
        sa.Integer,
        sa.Identity(),
        primary_key=True,
    )
    slug: Mapped[str] = mapped_column(sa.String(64), unique=True)
    name: Mapped[str] = mapped_column(sa.String(200))
    district: Mapped[str] = mapped_column(sa.String(100))

    latitude: Mapped[Decimal] = mapped_column(sa.Numeric(9, 6))
    longitude: Mapped[Decimal] = mapped_column(sa.Numeric(9, 6))

    total_spaces: Mapped[int] = mapped_column(sa.Integer)
    bookable_spaces: Mapped[int] = mapped_column(sa.Integer)

    space_types: Mapped[list[ParkingSpaceType]] = relationship(
        back_populates="parking",
        lazy="raise",
    )
    working_hours: Mapped[list[WorkingHours]] = relationship(
        back_populates="parking",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    special_hours: Mapped[list[SpecialHours]] = relationship(
        back_populates="parking",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )
    prices: Mapped[list[Price]] = relationship(
        "Price",
        primaryjoin="Parking.id == foreign(Price.parking_id)",
        viewonly=True,
        lazy="raise",
    )

    # Navigation only: the actual foreign key is to ParkingSpaceType.
    reservations: Mapped[list[Reservation]] = relationship(
        "Reservation",
        primaryjoin="Parking.id == foreign(Reservation.parking_id)",
        back_populates="parking",
        viewonly=True,
        lazy="raise",
    )
    capacity_overrides: Mapped[list[CapacityOverride]] = relationship(
        "CapacityOverride",
        primaryjoin=(
            "Parking.id == foreign(CapacityOverride.parking_id)"
        ),
        back_populates="parking",
        viewonly=True,
        lazy="raise",
    )

    __table_args__ = (
        sa.CheckConstraint(
            "latitude BETWEEN -90 AND 90",
            name="ck_parkings_latitude",
        ),
        sa.CheckConstraint(
            "longitude BETWEEN -180 AND 180",
            name="ck_parkings_longitude",
        ),
        sa.CheckConstraint(
            "total_spaces >= 0",
            name="ck_parkings_total_spaces",
        ),
        sa.CheckConstraint(
            "bookable_spaces >= 0 AND bookable_spaces <= total_spaces",
            name="ck_parkings_bookable_spaces",
        ),
    )