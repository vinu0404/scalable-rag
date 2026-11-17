"""Index source versions by content and generations by source.

Duplicate detection looks up live documents by SHA-256: content artifact ->
source versions -> generations -> document. Built CONCURRENTLY so existing
tables keep accepting writes while the indexes build.
"""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None
STATEMENTS = [
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_sources_artifact"
    " ON source_versions (tenant_id, artifact_id)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_generations_source"
    " ON generations (tenant_id, source_id)",
]


def upgrade():
    """CONCURRENTLY cannot run inside a transaction block."""
    with op.get_context().autocommit_block():
        for statement in STATEMENTS:
            op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
