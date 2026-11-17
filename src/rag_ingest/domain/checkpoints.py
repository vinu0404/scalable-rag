"""Describe completed work whose saved output can survive a worker restart.

Write output to durable storage first, then register its checkpoint using the
current job lease. A Python object alone does not make that output durable.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_ingest.domain.enums import ProcessingStage


@dataclass(frozen=True, kw_only=True)
class StageCheckpoint:
    """Record the completed output of one stage job or batch.

    Example: embedding batch 0 finishes and saves a manifest listing its chunk
    IDs and vector outputs. After a crash, a worker can validate that manifest
    instead of repeating the finished batch. An unfinished batch has no complete
    checkpoint and must be recovered through safe, repeatable operations.

    source_sha256 identifies the original file. input_sha256 identifies this
    stage's exact input or input manifest. pipeline_fingerprint identifies the
    processing settings. Reuse must match all of these and the work identity.

    The manifest describes output locations, checksums, and format versions.
    It and its referenced outputs must use pinned versions or write-once keys.
    Register only after verifying they are complete. Do not overwrite outputs
    from another attempt; use immutable attempt-specific storage locations.

    producer_lease_id records which claim committed this checkpoint. A new
    worker may reuse it under its own current lease after checking validity.
    This record does not renew ownership or make a generation searchable.
    """

    checkpoint_id: UUID
    tenant_id: UUID
    document_id: UUID
    generation_id: UUID
    job_id: UUID
    producer_lease_id: UUID
    stage: ProcessingStage
    shard_key: str
    source_sha256: str
    input_sha256: str
    pipeline_fingerprint: str
    manifest_bucket: str
    manifest_key: str
    manifest_sha256: str
    manifest_size_bytes: int
    created_at: datetime
    manifest_version_id: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed checkpoint fields before they are used.

        This checks IDs, hashes, locations, size, and time. It does not read
        storage or prove outputs exist. Publication must be a conditional
        database operation, so it cannot be represented by this checkpoint.
        """
        for name in (
            "checkpoint_id",
            "tenant_id",
            "document_id",
            "generation_id",
            "job_id",
            "producer_lease_id",
        ):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        if (
            not isinstance(self.stage, ProcessingStage)
            or self.stage is ProcessingStage.PUBLISH
        ):
            raise ValueError("stage must be a processing stage before publication")
        for name in ("shard_key", "manifest_bucket", "manifest_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")
        for name in (
            "source_sha256",
            "input_sha256",
            "pipeline_fingerprint",
            "manifest_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if type(self.manifest_size_bytes) is not int or self.manifest_size_bytes <= 0:
            raise ValueError("manifest_size_bytes must be a positive integer")
        if self.manifest_version_id is not None:
            if (
                not isinstance(self.manifest_version_id, str)
                or not self.manifest_version_id.strip()
            ):
                raise ValueError("manifest_version_id must not be blank when provided")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("created_at must be a datetime with a timezone")
