"""Stable vocabulary shared by ingestion services and adapters."""

from enum import StrEnum, unique


@unique
class JobStatus(StrEnum):
    """Execution state of one job, independent of its queue message."""

    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    # Parked after submitting work to an external tool; no lease, no attempt
    # spent while waiting. The scheduler re-queues it once results are in.
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@unique
class ProcessingStage(StrEnum):
    """Processing stages after a completed upload has been verified."""

    EXTRACT = "extract"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    PUBLISH = "publish"
