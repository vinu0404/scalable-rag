"""Built-in keyword index: a tsvector per chunk row and a per-tenant GIN index.

Columns stay empty until a keyword index is enabled (INGEST_KEYWORD_INDEX);
`rag-ingest reindex-keyword` fills them for already published builds.
btree_gin lets one GIN index cover (tenant_id, search_tsv). Column additions
are instant; the index builds CONCURRENTLY.
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None
STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS btree_gin",
    "ALTER TABLE search_chunks ADD COLUMN IF NOT EXISTS search_tsv TSVECTOR",
    "ALTER TABLE search_chunks ADD COLUMN IF NOT EXISTS keyword_target TEXT",
]
CONCURRENT = [
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_search_chunks_keyword"
    " ON search_chunks USING gin (tenant_id, search_tsv)",
]


def upgrade():
    """Add the extension and columns, then build the index outside a transaction."""
    for statement in STATEMENTS:
        op.execute(statement)
    with op.get_context().autocommit_block():
        for statement in CONCURRENT:
            op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
