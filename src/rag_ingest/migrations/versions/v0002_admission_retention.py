"""Add generation admission, outbox job lookup, and retention timestamps.

Existing generations are marked admitted so in-flight work continues with
its saved deadlines. Index builds briefly block writes to their tables; on a
large existing outbox, run this during low traffic.
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None
STATEMENTS = [
    "ALTER TABLE generations ADD COLUMN admitted_at TIMESTAMP WITH TIME ZONE",
    "UPDATE generations SET admitted_at = now()",
    "ALTER TABLE operations ADD COLUMN created_at TIMESTAMP WITH TIME ZONE"
    " DEFAULT now() NOT NULL",
    "ALTER TABLE quarantine ADD COLUMN created_at TIMESTAMP WITH TIME ZONE"
    " DEFAULT now() NOT NULL",
    "CREATE INDEX ix_generations_waiting ON generations (tenant_id)"
    " WHERE admitted_at IS NULL",
    "CREATE INDEX ix_outbox_job ON outbox (tenant_id, job_id)",
    "CREATE INDEX ix_outbox_published ON outbox (published_at)"
    " WHERE published_at IS NOT NULL",
    "CREATE INDEX ix_operations_created ON operations (created_at)",
    "CREATE INDEX ix_quarantine_created ON quarantine (created_at)",
]


def upgrade():
    """Apply additive columns, backfill, and indexes in one transaction."""
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
