from __future__ import annotations

from datetime import date as DateValue
from datetime import time
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.parking import Parking


class SpecialHours(Base):
    """Date-specific hours replacing the weekly schedule."""

    __tablename__ = "special_hours"

    parking_id: Mapped[int] = mapped_column(
        sa.ForeignKey("parkings.id", ondelete="CASCADE"),
        primary_key=True,
    )
    date: Mapped[DateValue] = mapped_column(
        sa.Date,
        primary_key=True,
    )

    open_time: Mapped[time | None] = mapped_column(sa.Time)
    close_time: Mapped[time | None] = mapped_column(sa.Time)
    is_closed: Mapped[bool] = mapped_column(
        sa.Boolean,
        server_default=sa.false(),
    )
    note: Mapped[str | None] = mapped_column(sa.Text)

    parking: Mapped[Parking] = relationship(
        back_populates="special_hours",
        lazy="raise",
    )

    __table_args__ = (
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
            name="ck_special_hours_interval",
        ),
    )