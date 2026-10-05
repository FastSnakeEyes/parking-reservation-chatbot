from __future__ import annotations

from datetime import time
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.parking import Parking


class WorkingHours(Base):
    """Weekly schedule: Monday=1, Sunday=7."""

    __tablename__ = "working_hours"

    parking_id: Mapped[int] = mapped_column(
        sa.ForeignKey("parkings.id", ondelete="CASCADE"),
        primary_key=True,
    )
    day_of_week: Mapped[int] = mapped_column(
        sa.SmallInteger,
        primary_key=True,
    )

    open_time: Mapped[time | None] = mapped_column(sa.Time)
    close_time: Mapped[time | None] = mapped_column(sa.Time)
    is_closed: Mapped[bool] = mapped_column(
        sa.Boolean,
        server_default=sa.false(),
    )

    parking: Mapped[Parking] = relationship(
        back_populates="working_hours",
        lazy="raise",
    )

    __table_args__ = (
        sa.CheckConstraint(
            "day_of_week BETWEEN 1 AND 7",
            name="ck_working_hours_day",
        ),
        sa.CheckConstraint(
            """
            (is_closed AND open_time IS NULL AND close_time IS NULL)
            OR
            (
                NOT is_closed
                AND open_time IS NOT NULL
                AND close_time IS NOT NULL
                AND open_time < close_time
            )
            """,
            name="ck_working_hours_interval",
        ),
    )