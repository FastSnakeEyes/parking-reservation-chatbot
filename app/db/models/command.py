from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base


class Command(Base):
    """A durable, worker-executed unit of work (WorkflowService's outbox).

    GraphRunner enqueues one of these instead of running the graph inline in
    the request; the in-process worker claims and executes it. The
    idempotency_key makes a retried enqueue a no-op instead of a duplicate.
    """

    __tablename__ = "commands"

    id: Mapped[UUID] = mapped_column(
        sa.Uuid,
        primary_key=True,
        default=uuid4,
    )
    idempotency_key: Mapped[str] = mapped_column(
        sa.String(200),
        unique=True,
    )
    command_type: Mapped[str] = mapped_column(sa.String(50))
    thread_id: Mapped[str] = mapped_column(sa.String(100))
    payload: Mapped[dict] = mapped_column(JSONB)

    status: Mapped[str] = mapped_column(
        sa.String(20),
        server_default=sa.text("'pending'"),
    )
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(sa.Text)

    claimed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
    )
    completed_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True))

    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('pending', 'claimed', 'succeeded', 'failed')",
            name="ck_commands_status",
        ),
        sa.Index("ix_commands_pending", "status", "created_at"),
    )
