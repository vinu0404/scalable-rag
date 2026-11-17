"""Check retry safety using fixed times, without a queue or database.

These tests verify decisions only. SQL concurrency and actual retry scheduling
need separate integration tests when the PostgreSQL adapter is implemented.
"""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RateLimitedError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.domain.enums import JobStatus, ProcessingStage
from rag_ingest.domain.jobs import Job, JobLease
from rag_ingest.domain.policy import ExecutionBudget, ExecutionPolicy
from rag_ingest.retry_decision import RetryAction, decide_job_retry


class RetryDecisionTests(unittest.TestCase):
    """Protect ownership and fixed-budget behavior against future regressions."""

    def setUp(self) -> None:
        """Build one running job with reproducible identity and timing."""
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.policy = ExecutionPolicy()
        self.budget = ExecutionBudget(
            policy=self.policy,
            starts_at=self.now,
            deadline_at=self.now + timedelta(minutes=30),
            generation_deadline_at=self.now + timedelta(hours=1),
        )
        self.lease = JobLease(
            worker_id="worker-A",
            lease_id=UUID(int=5),
            acquired_at=self.now,
            expires_at=self.now + timedelta(seconds=120),
            attempt_deadline_at=self.now + timedelta(seconds=300),
        )
        self.job = Job(
            tenant_id=UUID(int=1),
            document_id=UUID(int=2),
            generation_id=UUID(int=3),
            job_id=UUID(int=4),
            stage=ProcessingStage.EMBED,
            status=JobStatus.RUNNING,
            created_at=self.now,
            budget=self.budget,
            attempt=1,
            lease=self.lease,
        )

    def decide(self, *, job=None, error=None, **overrides):
        """Supply fixed defaults while allowing a test to change one scenario."""
        arguments = dict(
            worker_id="worker-A",
            lease_id=self.lease.lease_id,
            now=self.now,
            jitter_fraction=0.5,
        )
        arguments.update(overrides)
        return decide_job_retry(
            self.job if job is None else job,
            RetryableError("Temporary provider failure") if error is None else error,
            **arguments,
        )

    def test_jitter_uses_saved_backoff_policy(self):
        """Later attempts grow the window without changing the persisted budget."""
        for attempt, expected in ((1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0)):
            with self.subTest(attempt=attempt):
                decision = self.decide(job=replace(self.job, attempt=attempt))
                self.assertEqual(decision.action, RetryAction.RETRY)
                self.assertEqual(decision.delay_seconds, expected)
        self.assertEqual(self.job.attempt, 1)
        self.assertEqual(self.job.budget, self.budget)

    def test_final_allowed_attempt_is_terminal(self):
        """Five total attempts must not accidentally become six executions."""
        decision = self.decide(job=replace(self.job, attempt=5))
        self.assertEqual(decision.action, RetryAction.FAIL)
        self.assertEqual(decision.reason_code, "attempt_budget_exhausted")
        self.assertIsNone(decision.delay_seconds)

    def test_provider_minimum_can_exceed_backoff_cap(self):
        """A provider's longer wait must not be shortened to the jitter cap."""
        decision = self.decide(
            error=RateLimitedError("Throttled", retry_after_seconds=90)
        )
        self.assertEqual(decision.action, RetryAction.RETRY)
        self.assertEqual(decision.delay_seconds, 90)

    def test_retry_at_deadline_is_rejected(self):
        """Starting exactly at the fixed deadline leaves no execution time."""
        decision = self.decide(
            error=RateLimitedError("Throttled", retry_after_seconds=1800)
        )
        self.assertEqual(decision.action, RetryAction.FAIL)
        self.assertEqual(decision.reason_code, "retry_cannot_fit_deadline")

    def test_delay_just_before_deadline_is_not_silently_shortened(self):
        """A fitting delay is preserved; database timing still needs rechecking."""
        decision = self.decide(
            error=RateLimitedError("Throttled", retry_after_seconds=1799)
        )
        self.assertEqual(decision.action, RetryAction.RETRY)
        self.assertEqual(decision.delay_seconds, 1799)

    def test_replaced_worker_or_lease_must_stop(self):
        """An old attempt cannot borrow the current acquisition's authority."""
        for overrides in ({"worker_id": "worker-B"}, {"lease_id": UUID(int=99)}):
            with self.subTest(overrides=overrides):
                decision = self.decide(**overrides)
                self.assertEqual(decision.action, RetryAction.STOP)
                self.assertIsNone(decision.delay_seconds)

    def test_exact_lease_expiry_stops_even_when_attempt_budget_is_spent(self):
        """An expired worker must not propose a failure write for another owner."""
        decision = self.decide(
            job=replace(self.job, attempt=5), now=self.lease.expires_at
        )
        self.assertEqual(decision.action, RetryAction.STOP)
        self.assertEqual(decision.reason_code, "lease_expired")

    def test_completed_job_is_not_retried(self):
        """Duplicate delivery must not restart completed work through this helper."""
        completed = replace(self.job, status=JobStatus.SUCCEEDED, lease=None)
        self.assertEqual(self.decide(job=completed).action, RetryAction.STOP)

    def test_uncertain_and_nonretryable_errors_are_not_treated_as_transient(self):
        """Known unsafe categories must use their own recovery or failure paths."""
        for error in (
            UncertainOutcomeError("Unknown write outcome"),
            PermanentInputError("Unsupported input"),
            OperatorActionRequiredError("Dependency access denied"),
        ):
            with self.subTest(category=type(error).__name__):
                with self.assertRaises(ValueError):
                    self.decide(error=error)

    def test_large_attempt_exponent_saturates_without_huge_integer_work(self):
        """A large configured budget must not cause unbounded exponent allocation."""
        retry = replace(self.policy.retry, max_attempts=10**100)
        policy = replace(self.policy, retry=retry)
        job = replace(
            self.job, budget=replace(self.budget, policy=policy), attempt=10**90
        )
        self.assertEqual(self.decide(job=job, jitter_fraction=1).delay_seconds, 60)

    def test_invalid_clock_and_jitter_inputs_are_rejected(self):
        """Bad timing inputs must not yield a plausible scheduling decision."""
        cases = (
            {"now": self.now.replace(tzinfo=None)},
            {"now": self.now - timedelta(seconds=1)},
            {"jitter_fraction": float("nan")},
            {"jitter_fraction": 1.1},
            {"jitter_fraction": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.decide(**overrides)


if __name__ == "__main__":
    unittest.main()
