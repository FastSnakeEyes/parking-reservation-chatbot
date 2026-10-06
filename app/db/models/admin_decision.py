from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models.base import Base

if TYPE_CHECKING:
    from app.db.models.reservation import Reservation


class AdminDecision(Base):
    """The administrator's decision on a reservation, recorded once.

    `UNIQUE(reservation_id)` enforces "only one decision per reservation" at
    the database level: a second decide() on the same reservation raises an
    integrity error the repository translates into a typed conflict, rather
    than silently overwriting the first decision.
    """

    __tablename__ = "admin_decisions"

    id: Mapped[UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid4,
    )
    reservation_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("reservations.id"),
        unique=True,
    )
    # The reservations.version this decision was made against - lets the
    # caller detect a decision made on a since-changed reservation.
    reservation_version: Mapped[int] = mapped_column(sa.Integer)

    decision: Mapped[str] = mapped_column(sa.String(20))
    note: Mapped[str | None] = mapped_column(sa.Text)
    decided_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    idempotency_key: Mapped[str] = mapped_column(sa.String(200))

    reservation: Mapped[Reservation] = relationship(lazy="raise")

    __table_args__ = (
        sa.CheckConstraint(
            "decision IN ('approved', 'rejected')",
            name="ck_admin_decisions_decision",
        ),
    )
