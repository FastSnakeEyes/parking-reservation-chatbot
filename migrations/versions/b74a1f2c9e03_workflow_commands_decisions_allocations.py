"""workflow commands, admin decisions, capacity allocations

Revision ID: b74a1f2c9e03
Revises: a23259397b3e
Create Date: 2026-10-07 09:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "b74a1f2c9e03"
down_revision: str | Sequence[str] | None = "a23259397b3e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "reservations",
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
    )
    op.add_column(
        "reservations",
        sa.Column("export_status", sa.String(length=20), server_default=sa.text("'pending'"), nullable=False),
    )
    op.add_column(
        "reservations",
        sa.Column("exported_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_reservations_export_status",
        "reservations",
        "export_status IN ('pending', 'exported', 'failed')",
    )

    op.create_table(
        "commands",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("command_type", sa.String(length=50), nullable=False),
        sa.Column("thread_id", sa.String(length=100), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("status", sa.String(length=20), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'claimed', 'succeeded', 'failed')",
            name="ck_commands_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index("ix_commands_pending", "commands", ["status", "created_at"], unique=False)

    op.create_table(
        "admin_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("reservation_id", sa.Uuid(), nullable=False),
        sa.Column("reservation_version", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=20), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.CheckConstraint(
            "decision IN ('approved', 'rejected')",
            name="ck_admin_decisions_decision",
        ),
        sa.ForeignKeyConstraint(["reservation_id"], ["reservations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("reservation_id"),
    )

    op.create_table(
        "capacity_allocations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("parking_id", sa.Integer(), nullable=False),
        sa.Column("space_type", sa.String(length=20), nullable=False),
        sa.Column("reservation_id", sa.Uuid(), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("ends_at > starts_at", name="ck_capacity_allocations_period"),
        sa.ForeignKeyConstraint(["reservation_id"], ["reservations.id"]),
        sa.ForeignKeyConstraint(
            ["parking_id", "space_type"],
            ["parking_space_types.parking_id", "parking_space_types.space_type"],
            name="fk_capacity_allocations_space_type",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("reservation_id"),
    )
    op.create_index(
        "ix_capacity_allocations_lookup",
        "capacity_allocations",
        ["parking_id", "space_type", "starts_at", "ends_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_capacity_allocations_lookup", table_name="capacity_allocations")
    op.drop_table("capacity_allocations")

    op.drop_table("admin_decisions")

    op.drop_index("ix_commands_pending", table_name="commands")
    op.drop_table("commands")

    op.drop_constraint("ck_reservations_export_status", "reservations", type_="check")
    op.drop_column("reservations", "exported_at")
    op.drop_column("reservations", "export_status")
    op.drop_column("reservations", "version")
