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


class Reservation(Base):
    """Private booking data; never ingest it into the vector database."""

    __tablename__ = "reservations"

    id: Mapped[UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid4,
    )
    parking_id: Mapped[int] = mapped_column(sa.Integer)
    space_type: Mapped[str] = mapped_column(sa.String(20))

    first_name: Mapped[str] = mapped_column(sa.String(100))
    surname: Mapped[str] = mapped_column(sa.String(100))
    car_number: Mapped[str] = mapped_column(sa.String(20))

    starts_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
    )
    ends_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
    )

    status: Mapped[str] = mapped_column(
        sa.String(20),
        server_default=sa.text("'pending'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    approved_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
    )
    # The ParkingGraph thread (chat:{session_id}) paused waiting for this
    # reservation's approval - the admin API resumes it by this id.
    conversation_id: Mapped[str | None] = mapped_column(sa.String(100))

    capacity_type: Mapped[ParkingSpaceType] = relationship(
        back_populates="reservations",
        lazy="raise",
    )
    parking: Mapped[Parking] = relationship(
        "Parking",
        primaryjoin="foreign(Reservation.parking_id) == Parking.id",
        back_populates="reservations",
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
            name="fk_reservations_space_type",
        ),
        sa.CheckConstraint(
            "ends_at > starts_at",
            name="ck_reservations_period",
        ),
        sa.CheckConstraint(
            """
            status IN (
                'pending', 'approved', 'rejected', 'cancelled', 'recorded'
            )
            """,
            name="ck_reservations_status",
        ),
        sa.CheckConstraint(
            """
            status NOT IN ('approved', 'recorded')
            OR approved_at IS NOT NULL
            """,
            name="ck_reservations_approval",
        ),
        sa.Index(
            "ix_reservations_availability",
            "parking_id",
            "space_type",
            "status",
            "starts_at",
            "ends_at",
        ),
    )
