"""Track erasure of files left by uploads that were never accepted.

A refused, failed or expired upload may still have bytes (or unfinished
multipart parts) under its own unique key. files_removed_at records that the
scheduler erased them; the partial index finds the ones still to do. Adding a
nullable column is instant; the index builds CONCURRENTLY.
"""

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None
STATEMENTS = [
    "ALTER TABLE uploads ADD COLUMN IF NOT EXISTS files_removed_at TIMESTAMP WITH TIME ZONE",
]
CONCURRENT = [
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_uploads_abandoned"
    " ON uploads (tenant_id, expires_at)"
    " WHERE status IN ('failed', 'expired') AND files_removed_at IS NULL",
]


def upgrade():
    """Add the column, then build the index outside a transaction block."""
    for statement in STATEMENTS:
        op.execute(statement)
    with op.get_context().autocommit_block():
        for statement in CONCURRENT:
            op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
