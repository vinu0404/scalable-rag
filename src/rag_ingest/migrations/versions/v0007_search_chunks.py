"""Search foundation: one copy of chunk text, and document labels for filters.

- search_chunks holds each chunk's text once; vector and keyword indexes keep
  only IDs. Builds indexed before this revision have no chunk rows: reprocess
  them (POST /v1/documents/{id}/reprocess). Scheduled batches must have plans.
- documents gains tags, metadata and created_at columns. created_at is filled
  from each document's saved record here; labels start empty.

Column additions with constant defaults are instant; indexes build CONCURRENTLY.
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None
STATEMENTS = [
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{}' NOT NULL",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}' NOT NULL",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE",
    # Every scheduled batch must carry its plan; batches without one cannot run.
    "ALTER TABLE scheduled_batches ALTER COLUMN plan SET NOT NULL",
    "UPDATE documents SET created_at = (payload->'fields'->'created_at'->>'$time')::timestamptz"
    " WHERE created_at IS NULL",
    "\nCREATE TABLE IF NOT EXISTS search_chunks (\n\ttenant_id UUID NOT NULL, \n\tvector_id TEXT NOT NULL, \n\tdocument_id UUID NOT NULL, \n\tgeneration_id UUID NOT NULL, \n\tordinal INTEGER NOT NULL, \n\tcontent TEXT NOT NULL, \n\tspans JSONB NOT NULL, \n\ttext_sha256 VARCHAR(64) NOT NULL, \n\tPRIMARY KEY (tenant_id, vector_id), \n\tFOREIGN KEY(tenant_id, generation_id) REFERENCES generations (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
]
CONCURRENT = [
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_search_chunks_generation"
    " ON search_chunks (tenant_id, generation_id)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_tags ON documents USING gin (tags)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_metadata"
    " ON documents USING gin (metadata jsonb_path_ops)",
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documents_created"
    " ON documents (tenant_id, created_at)",
]


def upgrade():
    """Add columns and the table, then build indexes outside a transaction block."""
    for statement in STATEMENTS:
        op.execute(statement)
    with op.get_context().autocommit_block():
        for statement in CONCURRENT:
            op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
