"""Create initial durable metadata schema. This revision is immutable after release."""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None
STATEMENTS = [
    "\nCREATE TABLE content_artifacts (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tsha256 VARCHAR(64) NOT NULL, \n\tsize_bytes BIGINT NOT NULL, \n\tstate TEXT DEFAULT 'active' NOT NULL, \n\tPRIMARY KEY (tenant_id, id), \n\tUNIQUE (tenant_id, sha256), \n\tCHECK (size_bytes >= 0)\n)\n\n",
    "\nCREATE TABLE documents (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tactive_id UUID, \n\tdesired_id UUID, \n\tdeleted_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (tenant_id, id)\n)\n\n",
    "\nCREATE TABLE operations (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tkind TEXT NOT NULL, \n\tbinding VARCHAR(64) NOT NULL, \n\tresult JSONB NOT NULL, \n\tPRIMARY KEY (tenant_id, id)\n)\n\n",
    "\nCREATE TABLE quarantine (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\treason TEXT NOT NULL, \n\tPRIMARY KEY (tenant_id, id)\n)\n\n",
    "\nCREATE TABLE cleanup (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tdocument_id UUID NOT NULL, \n\tgeneration_id UUID, \n\tstatus TEXT DEFAULT 'pending' NOT NULL, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, document_id) REFERENCES documents (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "\nCREATE TABLE source_versions (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tdocument_id UUID NOT NULL, \n\tartifact_id UUID NOT NULL, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, document_id) REFERENCES documents (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tFOREIGN KEY(tenant_id, artifact_id) REFERENCES content_artifacts (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "\nCREATE TABLE uploads (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tdocument_id UUID NOT NULL, \n\trequest_key TEXT NOT NULL, \n\trequest_fingerprint VARCHAR(64) NOT NULL, \n\tbaseline UUID, \n\tpipeline JSONB NOT NULL, \n\tverification_ref JSONB, \n\tstatus TEXT NOT NULL, \n\texpires_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tfailure_code TEXT, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, document_id) REFERENCES documents (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tUNIQUE (tenant_id, request_key)\n)\n\n",
    "\nCREATE TABLE generations (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tdocument_id UUID NOT NULL, \n\tsource_id UUID NOT NULL, \n\tpipeline JSONB NOT NULL, \n\tdeadline_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tinventory JSONB, \n\tinventory_ref JSONB, \n\tchunking_job_id UUID, \n\tscheduling_done BOOLEAN DEFAULT false NOT NULL, \n\tstatus TEXT DEFAULT 'processing' NOT NULL, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, document_id) REFERENCES documents (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tFOREIGN KEY(tenant_id, source_id) REFERENCES source_versions (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "\nCREATE TABLE multipart_plans (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tupload_id UUID NOT NULL, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, upload_id) REFERENCES uploads (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "\nCREATE TABLE upload_parts (\n\ttenant_id UUID NOT NULL, \n\tupload_id UUID NOT NULL, \n\tprovider_id TEXT NOT NULL, \n\tpart_number INTEGER NOT NULL, \n\tpayload JSONB NOT NULL, \n\tPRIMARY KEY (tenant_id, upload_id, provider_id, part_number), \n\tFOREIGN KEY(tenant_id, upload_id) REFERENCES uploads (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "\nCREATE TABLE jobs (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tdocument_id UUID NOT NULL, \n\tgeneration_id UUID NOT NULL, \n\tstage TEXT NOT NULL, \n\tshard TEXT NOT NULL, \n\tstatus TEXT NOT NULL, \n\tdue_at TIMESTAMP WITH TIME ZONE, \n\tlease_id UUID, \n\tlease_expires_at TIMESTAMP WITH TIME ZONE, \n\tdeadline_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\treconciled_at TIMESTAMP WITH TIME ZONE, \n\tretry_id UUID, \n\tfailure_code TEXT, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, document_id) REFERENCES documents (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tFOREIGN KEY(tenant_id, generation_id) REFERENCES generations (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tUNIQUE (tenant_id, generation_id, stage, shard), \n\tCHECK (status IN ('queued','running','retry_wait','succeeded','failed','cancelled')), \n\tCHECK ((status = 'running') = (lease_id IS NOT NULL AND lease_expires_at IS NOT NULL))\n)\n\n",
    "\nCREATE TABLE scheduled_batches (\n\ttenant_id UUID NOT NULL, \n\tgeneration_id UUID NOT NULL, \n\tbatch_key TEXT NOT NULL, \n\tjob_id UUID NOT NULL, \n\tPRIMARY KEY (tenant_id, generation_id, batch_key), \n\tFOREIGN KEY(tenant_id, generation_id) REFERENCES generations (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "\nCREATE TABLE checkpoints (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tjob_id UUID NOT NULL, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, job_id) REFERENCES jobs (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tUNIQUE (tenant_id, job_id)\n)\n\n",
    "\nCREATE TABLE outbox (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tjob_id UUID NOT NULL, \n\tavailable_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tpublished_at TIMESTAMP WITH TIME ZONE, \n\tclaim_id UUID, \n\tclaim_expires_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, job_id) REFERENCES jobs (tenant_id, id) DEFERRABLE INITIALLY DEFERRED\n)\n\n",
    "CREATE INDEX ix_uploads_expiry ON uploads (tenant_id, status, expires_at)",
    "CREATE INDEX ix_generations_fanout ON generations (tenant_id, status, scheduling_done)",
    "CREATE INDEX ix_jobs_due ON jobs (tenant_id, status, due_at)",
    "CREATE INDEX ix_jobs_expired ON jobs (tenant_id, status, lease_expires_at)",
    "CREATE INDEX ix_jobs_generation ON jobs (tenant_id, generation_id, stage, status)",
    "CREATE INDEX ix_jobs_recovery ON jobs (tenant_id, status, reconciled_at, id)",
    "CREATE INDEX ix_outbox_ready ON outbox (tenant_id, available_at) WHERE published_at IS NULL",
]


def upgrade():
    """Create tables and indexes in foreign-key dependency order."""
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
