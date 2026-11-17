"""Describe execution limits that must survive restarts and retries.

Runtime configuration supplies initial values. Persist the chosen policy and
absolute deadlines with accepted work; never refresh them on every attempt.
These records do not run timers, sleep, renew leases, or schedule retries.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime


def _positive_seconds(name: str, value: float) -> None:
    """Require finite positive seconds while rejecting booleans and text."""
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class RetryPolicy:
    """Limit total ownership attempts and bound exponential-backoff delays.

    max_attempts includes the first claim: 5 means at most 5 executions, not
    1 execution plus 5 retries. Base/cap define the jitter window; provider
    minimum delays can be longer, subject to the fixed execution deadline.
    Values are illustrative defaults to tune against real workloads.
    """

    max_attempts: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0

    def __post_init__(self) -> None:
        """Reject invalid attempt counts or inconsistent backoff settings."""
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        _positive_seconds("base_delay_seconds", self.base_delay_seconds)
        _positive_seconds("max_delay_seconds", self.max_delay_seconds)
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base_delay_seconds must not exceed max_delay_seconds")


@dataclass(frozen=True, kw_only=True)
class ExecutionPolicy:
    """Keep retry, attempt, lease, heartbeat, and job time limits together.

    max_job_seconds includes queue waiting, processing, and retry waiting once
    the generation is admitted; waiting for admission does not count.
    max_attempt_seconds limits one ownership attempt even if heartbeats keep
    succeeding. Renewals must be clamped to the fixed attempt/job deadlines.
    heartbeat_seconds controls database ownership checks; SQS visibility has
    separate transport configuration and renewal operations.
    """

    retry: RetryPolicy = field(default_factory=RetryPolicy)
    max_job_seconds: float = 1800.0
    max_attempt_seconds: float = 300.0
    lease_seconds: int = 120
    heartbeat_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Require a valid retry policy and consistent positive timing limits."""
        if not isinstance(self.retry, RetryPolicy):
            raise ValueError("retry must be a RetryPolicy")
        for name in ("max_job_seconds", "max_attempt_seconds", "heartbeat_seconds"):
            _positive_seconds(name, getattr(self, name))
        if type(self.lease_seconds) is not int or self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        if (
            not self.heartbeat_seconds
            < self.lease_seconds
            <= self.max_attempt_seconds
            <= self.max_job_seconds
        ):
            raise ValueError("Require heartbeat < lease <= attempt limit <= job limit")


@dataclass(frozen=True, kw_only=True)
class ExecutionBudget:
    """Bind saved policy to fixed times for one job within a generation.

    starts_at is when this job's budget begins, including queue waiting.
    deadline_at cannot exceed starts_at + max_job_seconds or the generation's
    fixed overall deadline. Downstream jobs inherit the generation deadline;
    creating another stage or retry must not restart that overall clock.

    The database chooses authoritative times. Expired budgets remain readable
    for reconciliation; constructing a record does not compare it to wall time.
    """

    policy: ExecutionPolicy
    starts_at: datetime
    deadline_at: datetime
    generation_deadline_at: datetime

    def __post_init__(self) -> None:
        """Check timezone-aware ordering and enforce the configured maximum window."""
        if not isinstance(self.policy, ExecutionPolicy):
            raise ValueError("policy must be an ExecutionPolicy")
        for name in ("starts_at", "deadline_at", "generation_deadline_at"):
            value = getattr(self, name)
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must be a datetime with a timezone")
        if not self.starts_at < self.deadline_at <= self.generation_deadline_at:
            raise ValueError(
                "Require starts_at < deadline_at <= generation_deadline_at"
            )
        if (
            self.deadline_at - self.starts_at
        ).total_seconds() > self.policy.max_job_seconds:
            raise ValueError("Job deadline exceeds its saved policy window")
