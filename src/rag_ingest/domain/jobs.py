"""Describe scheduled work and its current worker lease.

These records hold data. Database operations will claim work, renew leases,
and save results atomically so competing workers cannot both own a job.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_ingest.domain.enums import JobStatus, ProcessingStage
from rag_ingest.domain.policy import ExecutionBudget


@dataclass(frozen=True, kw_only=True)
class JobLease:
    """Identify one successful claim of a job by a worker.

    worker_id identifies the worker; lease_id identifies this particular claim.
    The database must assign a fresh lease_id whenever ownership is acquired.
    Workers must supply it when renewing ownership or saving progress.
    An expired lease can still appear in a snapshot so recovery can inspect it.

    acquired_at and attempt_deadline_at are fixed for this acquisition. A
    heartbeat may move expires_at forward, but never beyond the attempt or job
    deadline. A fresh claim gets a new attempt window within the same job budget.
    """

    worker_id: str
    lease_id: UUID
    acquired_at: datetime
    expires_at: datetime
    attempt_deadline_at: datetime

    def __post_init__(self) -> None:
        """Check worker identity and timezone-aware ownership/attempt boundaries.

        This does not grant ownership or check the database's current time.
        """
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError("worker_id must not be blank")
        if not isinstance(self.lease_id, UUID):
            raise ValueError("lease_id must be a UUID")
        for name in ("acquired_at", "expires_at", "attempt_deadline_at"):
            value = getattr(self, name)
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must be a datetime with a timezone")
        if not self.acquired_at < self.expires_at <= self.attempt_deadline_at:
            raise ValueError("Require acquired_at < expires_at <= attempt_deadline_at")


@dataclass(frozen=True, kw_only=True)
class Job:
    """Describe one stage of work for a processing generation.

    For example, a job may extract text or embed one batch of chunks.
    shard_key distinguishes batches within the same stage; use '0' for unsplit
    work. attempt counts successful ownership claims, starting at zero.
    next_attempt_at records when scheduled retry work becomes eligible.
    A queue message points to this record; it is not the record itself.
    budget stores the policy and fixed deadlines; preserve it across retries.
    Its starts_at equals created_at so queue waiting counts against the budget.
    """

    tenant_id: UUID
    document_id: UUID
    generation_id: UUID
    job_id: UUID
    stage: ProcessingStage
    status: JobStatus
    created_at: datetime
    budget: ExecutionBudget
    shard_key: str = "0"
    attempt: int = 0
    lease: JobLease | None = None
    next_attempt_at: datetime | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent job data before another component uses it.

        Running work needs a lease and at least one claim. Other statuses must
        release the lease. Retry-wait work needs a scheduled retry time.
        These checks do not enforce transitions or prevent concurrent writes;
        database constraints and conditional updates must enforce those rules.
        """
        if not isinstance(self.status, JobStatus):
            raise ValueError("status must be a JobStatus")
        if not isinstance(self.stage, ProcessingStage):
            raise ValueError("stage must be a ProcessingStage")
        if not self.shard_key.strip():
            raise ValueError("shard_key must not be blank")
        if type(self.attempt) is not int or self.attempt < 0:
            raise ValueError("attempt must be a nonnegative integer")
        if not isinstance(self.budget, ExecutionBudget):
            raise ValueError("budget must be an ExecutionBudget")
        for name, value in (
            ("created_at", self.created_at),
            ("next_attempt_at", self.next_attempt_at),
        ):
            if value is not None and (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must include a timezone")
        # A budget starts at creation, or later when a waiting generation is
        # admitted; it can never start before the job existed.
        if self.budget.starts_at < self.created_at:
            raise ValueError("Job budget cannot start before created_at")
        if self.attempt > self.budget.policy.retry.max_attempts:
            raise ValueError("attempt exceeds the saved retry policy")
        if self.lease is not None and not isinstance(self.lease, JobLease):
            raise ValueError("lease must be a JobLease")
        if self.status is JobStatus.RUNNING:
            if self.lease is None or self.attempt == 0:
                raise ValueError(
                    "Running jobs need a lease and a positive attempt count"
                )
            if self.lease.acquired_at < self.budget.starts_at:
                raise ValueError("Lease acquisition cannot precede the job budget")
            if self.lease.attempt_deadline_at > self.budget.deadline_at:
                raise ValueError("Attempt deadline cannot exceed the job deadline")
            duration = (
                self.lease.attempt_deadline_at - self.lease.acquired_at
            ).total_seconds()
            if duration > self.budget.policy.max_attempt_seconds:
                raise ValueError("Attempt window exceeds its saved policy limit")
        elif self.lease is not None:
            raise ValueError("Only running jobs can hold a lease")
        if self.status is JobStatus.WAITING:
            if self.attempt < 1 or self.next_attempt_at is not None:
                raise ValueError(
                    "Waiting jobs follow a claimed attempt and have no retry time"
                )
        if self.status is JobStatus.RETRY_WAIT:
            if self.next_attempt_at is None:
                raise ValueError("Retry-wait jobs need next_attempt_at")
            if not 1 <= self.attempt < self.budget.policy.retry.max_attempts:
                raise ValueError(
                    "Retry-wait jobs need a prior attempt and remaining budget"
                )
            if (
                not self.budget.starts_at
                <= self.next_attempt_at
                < self.budget.deadline_at
            ):
                raise ValueError("Retry time must fit within the fixed job budget")
        elif self.next_attempt_at is not None:
            raise ValueError("Only retry-wait jobs can have next_attempt_at")
