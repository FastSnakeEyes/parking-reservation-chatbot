from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.parking import Parking
    from app.db.models.parking_space_type import ParkingSpaceType


class CapacityOverride(Base):
    """Temporary usable booking capacity before subtracting reservations."""

    __tablename__ = "capacity_overrides"

    id: Mapped[UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid4,
    )
    parking_id: Mapped[int] = mapped_column(sa.Integer)
    space_type: Mapped[str] = mapped_column(sa.String(20))

    starts_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
    )
    ends_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
    )

    bookable_capacity: Mapped[int] = mapped_column(sa.Integer)
    note: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )

    capacity_type: Mapped[ParkingSpaceType] = relationship(
        back_populates="capacity_overrides",
        lazy="raise",
    )
    parking: Mapped[Parking] = relationship(
        "Parking",
        primaryjoin=(
            "foreign(CapacityOverride.parking_id) == Parking.id"
        ),
        back_populates="capacity_overrides",
        viewonly=True,
        lazy="raise",
    )

    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["parking_id", "space_type"],
            [
                "parking_space_types.parking_id",
                "parking_space_types.space_type",
            ],
            name="fk_capacity_overrides_space_type",
        ),
        sa.CheckConstraint(
            "ends_at > starts_at",
            name="ck_capacity_overrides_period",
        ),
        sa.CheckConstraint(
            "bookable_capacity >= 0",
            name="ck_capacity_overrides_capacity",
        ),
        sa.Index(
            "ix_capacity_overrides_lookup",
            "parking_id",
            "space_type",
            "starts_at",
            "ends_at",
        ),
    )