"""approved replaces the recorded reservation status

An approved reservation is now stored as status 'approved' (approved by the
administrator and written through the MCP server); 'recorded' is gone.

Revision ID: c3d4e5f6a7b8
Revises: b7c1d2e3f4a5
Create Date: 2026-10-13 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b7c1d2e3f4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("ck_reservations_status", "reservations", type_="check")
    op.drop_constraint("ck_reservations_approval", "reservations", type_="check")

    # Every 'recorded' row was approved (approved_at is set) and written via MCP.
    op.execute("UPDATE reservations SET status = 'approved' WHERE status = 'recorded'")

    op.create_check_constraint(
        "ck_reservations_status",
        "reservations",
        "status IN ('pending', 'approved', 'rejected', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_reservations_approval",
        "reservations",
        "status <> 'approved' OR approved_at IS NOT NULL",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_reservations_status", "reservations", type_="check")
    op.drop_constraint("ck_reservations_approval", "reservations", type_="check")

    # Before this revision, every approval was stored as 'recorded'.
    op.execute("UPDATE reservations SET status = 'recorded' WHERE status = 'approved'")

    op.create_check_constraint(
        "ck_reservations_status",
        "reservations",
        "status IN ('pending', 'approved', 'rejected', 'cancelled', 'recorded')",
    )
    op.create_check_constraint(
        "ck_reservations_approval",
        "reservations",
        "status NOT IN ('approved', 'recorded') OR approved_at IS NOT NULL",
    )
