"""Versioned PostgreSQL tables and indexes for durable ingestion.

Payloads preserve validated domain snapshots. Relational columns support
constraints, ownership checks, and indexed scheduling without scanning payloads.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR

metadata = sa.MetaData()
U = sa.Uuid
T = sa.DateTime(timezone=True)


def table(name, *columns, constraints=()):
    """Give each record a composite tenant-scoped primary key."""
    return sa.Table(
        name,
        metadata,
        sa.Column("tenant_id", U, primary_key=True),
        sa.Column("id", U, primary_key=True),
        sa.Column("payload", JSONB, nullable=False),
        *columns,
        *constraints,
    )


def fk(column, target):
    """Prevent a record from referencing another tenant's parent."""
    return sa.ForeignKeyConstraint(
        ["tenant_id", column],
        [target + ".tenant_id", target + ".id"],
        deferrable=True,
        initially="DEFERRED",
    )


documents = table(
    "documents",
    sa.Column("active_id", U),
    sa.Column("desired_id", U),
    sa.Column("deleted_at", T),
    # Revision 0007: labels and creation time projected for search filters.
    sa.Column("tags", ARRAY(sa.Text), nullable=False, server_default="{}"),
    sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    sa.Column("created_at", T),
)
artifacts = table(
    "content_artifacts",
    sa.Column("sha256", sa.String(64), nullable=False),
    sa.Column("size_bytes", sa.BigInteger, nullable=False),
    sa.Column("state", sa.Text, nullable=False, server_default="active"),
    constraints=(
        sa.UniqueConstraint("tenant_id", "sha256"),
        sa.CheckConstraint("size_bytes >= 0"),
    ),
)
uploads = table(
    "uploads",
    sa.Column("document_id", U, nullable=False),
    sa.Column("request_key", sa.Text, nullable=False),
    sa.Column("request_fingerprint", sa.String(64), nullable=False),
    sa.Column("baseline", U),
    sa.Column("pipeline", JSONB, nullable=False),
    sa.Column("verification_ref", JSONB),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("expires_at", T, nullable=False),
    sa.Column("failure_code", sa.Text),
    # Revision 0006: when the stored file of a never-accepted upload was erased.
    sa.Column("files_removed_at", T),
    constraints=(
        fk("document_id", "documents"),
        sa.UniqueConstraint("tenant_id", "request_key"),
    ),
)
sources = table(
    "source_versions",
    sa.Column("document_id", U, nullable=False),
    sa.Column("artifact_id", U, nullable=False),
    constraints=(
        fk("document_id", "documents"),
        fk("artifact_id", "content_artifacts"),
    ),
)
generations = table(
    "generations",
    sa.Column("document_id", U, nullable=False),
    sa.Column("source_id", U, nullable=False),
    sa.Column("pipeline", JSONB, nullable=False),
    sa.Column("deadline_at", T, nullable=False),
    sa.Column("inventory", JSONB),
    sa.Column("inventory_ref", JSONB),
    sa.Column("chunking_job_id", U),
    sa.Column("scheduling_done", sa.Boolean, nullable=False, server_default=sa.false()),
    sa.Column("status", sa.Text, nullable=False, server_default="processing"),
    sa.Column("admitted_at", T),
    constraints=(fk("document_id", "documents"), fk("source_id", "source_versions")),
)
jobs = table(
    "jobs",
    sa.Column("document_id", U, nullable=False),
    sa.Column("generation_id", U, nullable=False),
    sa.Column("stage", sa.Text, nullable=False),
    sa.Column("shard", sa.Text, nullable=False),
    sa.Column("status", sa.Text, nullable=False),
    sa.Column("due_at", T),
    sa.Column("lease_id", U),
    sa.Column("lease_expires_at", T),
    sa.Column("deadline_at", T, nullable=False),
    sa.Column("reconciled_at", T),
    sa.Column("retry_id", U),
    sa.Column("failure_code", sa.Text),
    constraints=(
        fk("document_id", "documents"),
        fk("generation_id", "generations"),
        sa.UniqueConstraint("tenant_id", "generation_id", "stage", "shard"),
        sa.CheckConstraint(
            "status IN ('queued','running','retry_wait','waiting','succeeded','failed','cancelled')"
        ),
        sa.CheckConstraint(
            "(status = 'running') = (lease_id IS NOT NULL AND lease_expires_at IS NOT NULL)"
        ),
    ),
)
outbox = table(
    "outbox",
    sa.Column("job_id", U, nullable=False),
    sa.Column("available_at", T, nullable=False),
    sa.Column("published_at", T),
    sa.Column("claim_id", U),
    sa.Column("claim_expires_at", T),
    constraints=(fk("job_id", "jobs"),),
)
checkpoints = table(
    "checkpoints",
    sa.Column("job_id", U, nullable=False),
    constraints=(fk("job_id", "jobs"), sa.UniqueConstraint("tenant_id", "job_id")),
)
plans = table(
    "multipart_plans",
    sa.Column("upload_id", U, nullable=False),
    constraints=(fk("upload_id", "uploads"),),
)
parts = sa.Table(
    "upload_parts",
    metadata,
    sa.Column("tenant_id", U, primary_key=True),
    sa.Column("upload_id", U, primary_key=True),
    sa.Column("provider_id", sa.Text, primary_key=True),
    sa.Column("part_number", sa.Integer, primary_key=True),
    sa.Column("payload", JSONB, nullable=False),
    fk("upload_id", "uploads"),
)
operations = sa.Table(
    "operations",
    metadata,
    sa.Column("tenant_id", U, primary_key=True),
    sa.Column("id", U, primary_key=True),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("binding", sa.String(64), nullable=False),
    sa.Column("result", JSONB, nullable=False),
    sa.Column("created_at", T, nullable=False, server_default=sa.func.now()),
)
cleanup = table(
    "cleanup",
    sa.Column("document_id", U, nullable=False),
    sa.Column("generation_id", U),
    sa.Column("status", sa.Text, nullable=False, server_default="pending"),
    constraints=(fk("document_id", "documents"),),
)
scheduled = sa.Table(
    "scheduled_batches",
    metadata,
    sa.Column("tenant_id", U, primary_key=True),
    sa.Column("generation_id", U, primary_key=True),
    sa.Column("batch_key", sa.Text, primary_key=True),
    sa.Column("job_id", U, nullable=False),
    # Revision 0004: the batch's BatchPlan, so its embedding job reads one row
    # instead of the whole inventory (required since 0007).
    sa.Column("plan", JSONB, nullable=False),
    fk("generation_id", "generations"),
)
sa.Index("ix_jobs_due", jobs.c.tenant_id, jobs.c.status, jobs.c.due_at)
sa.Index("ix_jobs_expired", jobs.c.tenant_id, jobs.c.status, jobs.c.lease_expires_at)
sa.Index(
    "ix_outbox_ready",
    outbox.c.tenant_id,
    outbox.c.available_at,
    postgresql_where=outbox.c.published_at.is_(None),
)
sa.Index(
    "ix_jobs_generation",
    jobs.c.tenant_id,
    jobs.c.generation_id,
    jobs.c.stage,
    jobs.c.status,
)
sa.Index(
    "ix_uploads_expiry", uploads.c.tenant_id, uploads.c.status, uploads.c.expires_at
)

quarantine = table(
    "quarantine",
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("created_at", T, nullable=False, server_default=sa.func.now()),
)

sa.Index(
    "ix_jobs_recovery", jobs.c.tenant_id, jobs.c.status, jobs.c.reconciled_at, jobs.c.id
)

sa.Index(
    "ix_generations_fanout",
    generations.c.tenant_id,
    generations.c.status,
    generations.c.scheduling_done,
)

# Revision 0002: admission, recovery lookups and retention pruning.
sa.Index(
    "ix_generations_waiting",
    generations.c.tenant_id,
    postgresql_where=generations.c.admitted_at.is_(None),
)
sa.Index("ix_outbox_job", outbox.c.tenant_id, outbox.c.job_id)
sa.Index(
    "ix_outbox_published",
    outbox.c.published_at,
    postgresql_where=outbox.c.published_at.is_not(None),
)
sa.Index("ix_operations_created", operations.c.created_at)
sa.Index("ix_quarantine_created", quarantine.c.created_at)

# Revision 0003: external recognition jobs, shared rate limits, usage budgets.
recognition_jobs = table(
    "recognition_jobs",
    sa.Column("job_id", U, nullable=False),
    sa.Column("generation_id", U, nullable=False),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("next_poll_at", T),
    sa.Column("polls", sa.Integer, nullable=False, server_default="0"),
    sa.Column("submitted_at", T, nullable=False),
    sa.Column("finished_at", T),
    sa.Column("failure_code", sa.Text),
    sa.Column("failures", JSONB),
    constraints=(
        fk("job_id", "jobs"),
        sa.CheckConstraint(
            "state IN ('pending','done','failed','expired','abandoned')"
        ),
    ),
)
sa.Index(
    "ix_recognition_due",
    recognition_jobs.c.tenant_id,
    recognition_jobs.c.next_poll_at,
    postgresql_where=recognition_jobs.c.state == "pending",
)
sa.Index("ix_recognition_job", recognition_jobs.c.tenant_id, recognition_jobs.c.job_id)
rate_limits = sa.Table(
    "rate_limits",
    metadata,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("tokens", sa.Float, nullable=False),
    sa.Column("updated_at", T, nullable=False),
)
recognition_usage = sa.Table(
    "recognition_usage",
    metadata,
    sa.Column("tenant_id", U, primary_key=True),
    sa.Column("day", sa.Date, primary_key=True),
    sa.Column("request_id", sa.Text, primary_key=True),
    sa.Column("pages", sa.Integer, nullable=False),
)

# Revision 0005: find live documents by content (duplicate detection).
sa.Index("ix_sources_artifact", sources.c.tenant_id, sources.c.artifact_id)
sa.Index("ix_generations_source", generations.c.tenant_id, generations.c.source_id)

# Revision 0006: uploads that ended failed/expired and still have stored files.
sa.Index(
    "ix_uploads_abandoned",
    uploads.c.tenant_id,
    uploads.c.expires_at,
    postgresql_where=sa.and_(
        uploads.c.status.in_(["failed", "expired"]), uploads.c.files_removed_at.is_(None)
    ),
)

# Revision 0007: one copy of each chunk's text, for search results and filters.
# Vector and keyword indexes hold only IDs; results are read from here after
# checking the document is live, so no index can return deleted or foreign text.
search_chunks = sa.Table(
    "search_chunks",
    metadata,
    sa.Column("tenant_id", U, primary_key=True),
    sa.Column("vector_id", sa.Text, primary_key=True),
    sa.Column("document_id", U, nullable=False),
    sa.Column("generation_id", U, nullable=False),
    sa.Column("ordinal", sa.Integer, nullable=False),
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("spans", JSONB, nullable=False),
    sa.Column("text_sha256", sa.String(64), nullable=False),
    # Revision 0008: the built-in keyword index (PostgresKeywordIndex).
    sa.Column("search_tsv", TSVECTOR),
    sa.Column("keyword_target", sa.Text),
    fk("generation_id", "generations"),
)
sa.Index(
    "ix_search_chunks_generation",
    search_chunks.c.tenant_id,
    search_chunks.c.generation_id,
)
sa.Index("ix_documents_tags", documents.c.tags, postgresql_using="gin")
sa.Index(
    "ix_documents_metadata",
    documents.c.metadata,
    postgresql_using="gin",
    postgresql_ops={"metadata": "jsonb_path_ops"},
)
sa.Index("ix_documents_created", documents.c.tenant_id, documents.c.created_at)

# Revision 0008: one tenant's keyword matches are found inside one GIN index.
sa.Index(
    "ix_search_chunks_keyword",
    search_chunks.c.tenant_id,
    search_chunks.c.search_tsv,
    postgresql_using="gin",
)
# A GIN index over a uuid column needs btree_gin (also when tests create tables).
sa.event.listen(
    metadata, "before_create", sa.DDL("CREATE EXTENSION IF NOT EXISTS btree_gin")
)
