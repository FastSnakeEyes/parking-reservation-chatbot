from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.reservation import Reservation


class CapacityAllocation(Base):
    """One row per confirmed reservation's claim on bookable capacity.

    This is the ledger CapacityAllocationRepository counts against (locked
    on the parking_space_types row) to decide whether a new reservation
    fits - it is how "no more than bookable_capacity overlapping
    reservations" is enforced at the database level, as distinct from
    CapacityOverride (an admin-set temporary ceiling on capacity itself).
    """

    __tablename__ = "capacity_allocations"

    id: Mapped[UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid4,
    )
    parking_id: Mapped[int] = mapped_column(sa.Integer)
    space_type: Mapped[str] = mapped_column(sa.String(20))
    reservation_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("reservations.id"),
        unique=True,
    )

    starts_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )

    reservation: Mapped[Reservation] = relationship(lazy="raise")

    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["parking_id", "space_type"],
            [
                "parking_space_types.parking_id",
                "parking_space_types.space_type",
            ],
            name="fk_capacity_allocations_space_type",
        ),
        sa.CheckConstraint(
            "ends_at > starts_at",
            name="ck_capacity_allocations_period",
        ),
        sa.Index(
            "ix_capacity_allocations_lookup",
            "parking_id",
            "space_type",
            "starts_at",
            "ends_at",
        ),
    )
