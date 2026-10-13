"""delete the old approval pipeline's checkpoint threads

Before ParkingGraph, a separate approval graph checkpointed one thread per
reservation, keyed by the bare reservation id. Nothing reads those threads
any more (the current app only creates `chat:` threads), and the idle-thread
cleanup only sweeps `chat:` threads, so they would stay forever.

The checkpoint tables are created by the app's checkpointer.setup() on its
first start, not by Alembic - on a database where that hasn't happened yet,
there is nothing to delete.

Revision ID: d5e6f7a8b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-10-13 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5e6f7a8b9c0"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DELETE_LEGACY_THREADS = {
    "checkpoint_writes": "DELETE FROM checkpoint_writes WHERE thread_id NOT LIKE 'chat:%'",
    "checkpoint_blobs": "DELETE FROM checkpoint_blobs WHERE thread_id NOT LIKE 'chat:%'",
    "checkpoints": "DELETE FROM checkpoints WHERE thread_id NOT LIKE 'chat:%'",
}


def upgrade() -> None:
    """Upgrade schema."""
    existing = set(sa.inspect(op.get_bind()).get_table_names())

    for table, statement in DELETE_LEGACY_THREADS.items():
        if table in existing:
            op.execute(sa.text(statement))


def downgrade() -> None:
    """Downgrade schema."""
    # Deleted checkpoints can't be restored; the old approval graph is gone anyway.
