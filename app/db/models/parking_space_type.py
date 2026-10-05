from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.capacity_override import CapacityOverride
    from app.db.models.parking import Parking
    from app.db.models.reservation import Reservation
    from app.db.models.price import Price


class ParkingSpaceType(Base):
    """Aggregated capacity; EV spaces are equipped for charging."""

    __tablename__ = "parking_space_types"

    parking_id: Mapped[int] = mapped_column(
        sa.ForeignKey("parkings.id"),
        primary_key=True,
    )
    space_type: Mapped[str] = mapped_column(
        sa.String(20),
        primary_key=True,
    )

    total_spaces: Mapped[int] = mapped_column(sa.Integer)
    bookable_spaces: Mapped[int] = mapped_column(sa.Integer)

    parking: Mapped[Parking] = relationship(
        back_populates="space_types",
        lazy="raise",
    )
    reservations: Mapped[list[Reservation]] = relationship(
        back_populates="capacity_type",
        lazy="raise",
    )
    capacity_overrides: Mapped[list[CapacityOverride]] = relationship(
        back_populates="capacity_type",
        lazy="raise",
    )

    prices: Mapped[list[Price]] = relationship(
        back_populates="capacity_type",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        sa.CheckConstraint(
            "space_type IN ('standard', 'ev')",
            name="ck_space_types_type",
        ),
        sa.CheckConstraint(
            "total_spaces >= 0",
            name="ck_space_types_total_spaces",
        ),
        sa.CheckConstraint(
            "bookable_spaces >= 0 AND bookable_spaces <= total_spaces",
            name="ck_space_types_bookable_spaces",
        ),
    )