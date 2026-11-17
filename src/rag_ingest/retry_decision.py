"""Decide whether a known temporary failure permits another durable job attempt.

This is pure logic: no sleeping, random generation, database calls, or queue
operations. The database must recheck current ownership and budgets when the
service applies a decision because this job snapshot can become stale.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, unique
from uuid import UUID

from rag_ingest.contracts.errors import RetryableError
from rag_ingest.domain.enums import JobStatus
from rag_ingest.domain.jobs import Job


@unique
class RetryAction(StrEnum):
    """Choose scheduling, terminal failure, or stopping without a state write."""

    RETRY = "retry"
    FAIL = "fail"
    STOP = "stop"


@dataclass(frozen=True, kw_only=True)
class RetryDecision:
    """Return a proposed action with a safe reason code and optional delay.

    STOP means this attempt must not change job state using lost authority.
    FAIL still requires a current lease-guarded database write. RETRY does not
    schedule anything until schedule_job_retry commits its durable transition.
    """

    action: RetryAction
    reason_code: str
    delay_seconds: float | None = None

    def __post_init__(self) -> None:
        """Require a recognized action and a finite delay only for retry decisions."""
        if not isinstance(self.action, RetryAction):
            raise ValueError("action must be a RetryAction")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must not be blank")
        if self.action is RetryAction.RETRY:
            _check_nonnegative("delay_seconds", self.delay_seconds)
        elif self.delay_seconds is not None:
            raise ValueError("Only retry decisions can carry a delay")


def _check_nonnegative(name: str, value: object) -> None:
    """Reject invalid delay inputs without silently shortening provider guidance."""
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and nonnegative")


def decide_job_retry(
    job: Job,
    error: RetryableError,
    *,
    worker_id: str,
    lease_id: UUID,
    now: datetime,
    jitter_fraction: float,
) -> RetryDecision:
    """Apply saved attempt/deadline policy to one explicitly safe-to-repeat failure.

    Supply a fresh random fraction between zero and one for full jitter, or a
    fixed fraction in tests. On failed attempt n, the backoff window is
    min(cap, base * 2**(n-1)). Provider minimum delays can exceed that window
    but cannot move the fixed job/generation deadline.

    now must be timezone-aware and derived conservatively from authoritative
    timing. Comparing it to a snapshot does not establish current DB ownership.
    Uncertain outcomes, credentials, permanent input errors, and programming
    errors need separate handling; they must not be passed as RetryableError.
    """
    if not isinstance(job, Job) or not isinstance(error, RetryableError):
        raise ValueError("A Job and an explicitly retryable error are required")
    if (
        not isinstance(worker_id, str)
        or not worker_id.strip()
        or not isinstance(lease_id, UUID)
    ):
        raise ValueError("A worker identity and lease UUID are required")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be a datetime with a timezone")
    _check_nonnegative("jitter_fraction", jitter_fraction)
    if jitter_fraction > 1:
        raise ValueError("jitter_fraction must not exceed one")
    if job.status is not JobStatus.RUNNING or job.lease is None:
        return RetryDecision(action=RetryAction.STOP, reason_code="job_not_running")
    lease = job.lease
    if lease.worker_id != worker_id or lease.lease_id != lease_id:
        return RetryDecision(action=RetryAction.STOP, reason_code="lease_replaced")
    if now < lease.acquired_at:
        raise ValueError("Decision time cannot precede lease acquisition")
    if now >= lease.expires_at:
        return RetryDecision(action=RetryAction.STOP, reason_code="lease_expired")

    retry = job.budget.policy.retry
    if job.attempt >= retry.max_attempts:
        return RetryDecision(
            action=RetryAction.FAIL, reason_code="attempt_budget_exhausted"
        )
    # ldexp computes base * 2**exponent without constructing a huge integer.
    try:
        window = min(
            retry.max_delay_seconds,
            math.ldexp(retry.base_delay_seconds, job.attempt - 1),
        )
    except OverflowError:
        window = retry.max_delay_seconds
    provider_minimum = error.retry_after_seconds
    if provider_minimum is not None:
        _check_nonnegative("provider retry delay", provider_minimum)
    delay = max(window * jitter_fraction, provider_minimum or 0.0)
    remaining = (job.budget.deadline_at - now).total_seconds()
    if delay >= remaining:
        return RetryDecision(
            action=RetryAction.FAIL, reason_code="retry_cannot_fit_deadline"
        )
    return RetryDecision(
        action=RetryAction.RETRY, reason_code="temporary_failure", delay_seconds=delay
    )
