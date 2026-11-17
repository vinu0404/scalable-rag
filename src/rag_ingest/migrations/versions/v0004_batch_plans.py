"""Store each scheduled batch's plan so embedding jobs skip the inventory download.

Additive: one nullable column. Batches scheduled before this revision keep
NULL and their jobs fall back to reading the full sealed inventory.
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None
STATEMENTS = ["ALTER TABLE scheduled_batches ADD COLUMN plan JSONB"]


def upgrade():
    """Apply the additive column."""
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
