"""Add the waiting job status, external recognition jobs, rate limits and usage.

All changes are additive. The jobs status constraint is replaced in the same
transaction, so no row is ever unchecked.
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None
STATEMENTS = [
    "ALTER TABLE jobs DROP CONSTRAINT jobs_status_check",
    "ALTER TABLE jobs ADD CONSTRAINT jobs_status_check CHECK (status IN"
    " ('queued','running','retry_wait','waiting','succeeded','failed','cancelled'))",
    "\nCREATE TABLE recognition_jobs (\n\ttenant_id UUID NOT NULL, \n\tid UUID NOT NULL, \n\tpayload JSONB NOT NULL, \n\tjob_id UUID NOT NULL, \n\tgeneration_id UUID NOT NULL, \n\tstate TEXT NOT NULL, \n\tnext_poll_at TIMESTAMP WITH TIME ZONE, \n\tpolls INTEGER DEFAULT '0' NOT NULL, \n\tsubmitted_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tfinished_at TIMESTAMP WITH TIME ZONE, \n\tfailure_code TEXT, \n\tfailures JSONB, \n\tPRIMARY KEY (tenant_id, id), \n\tFOREIGN KEY(tenant_id, job_id) REFERENCES jobs (tenant_id, id) DEFERRABLE INITIALLY DEFERRED, \n\tCHECK (state IN ('pending','done','failed','expired','abandoned'))\n)\n\n",
    "CREATE INDEX ix_recognition_due ON recognition_jobs (tenant_id, next_poll_at)"
    " WHERE state = 'pending'",
    "CREATE INDEX ix_recognition_job ON recognition_jobs (tenant_id, job_id)",
    "\nCREATE TABLE rate_limits (\n\tkey TEXT NOT NULL, \n\ttokens FLOAT NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (key)\n)\n\n",
    "\nCREATE TABLE recognition_usage (\n\ttenant_id UUID NOT NULL, \n\tday DATE NOT NULL, \n\trequest_id TEXT NOT NULL, \n\tpages INTEGER NOT NULL, \n\tPRIMARY KEY (tenant_id, day, request_id)\n)\n\n",
]


def upgrade():
    """Apply additive recognition schema in one transaction."""
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    """Refuse automated destructive rollback of accepted ingestion history."""
    raise RuntimeError("Restore from a reviewed backup/migration plan instead")
