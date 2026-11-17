"""Exercise real Tenacity async control flow without contacting providers.

Only cooperative async operations are tested. SDK retry amplification and
remote write uncertainty still require adapter integration tests.
"""

import asyncio
import unittest
from dataclasses import replace
from unittest.mock import patch

from rag_ingest.contracts.errors import (
    LeaseLostError,
    OperatorActionRequiredError,
    PermanentInputError,
    RateLimitedError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.retry import CallRetryPolicy, run_with_retries


class AsyncRetryTests(unittest.IsolatedAsyncioTestCase):
    """Protect call limits, exception identity, deadline handling, and cancellation."""

    async def asyncSetUp(self):
        """Use a fresh event loop and short test policy for every scenario."""
        self.loop = asyncio.get_running_loop()
        self.policy = CallRetryPolicy(
            max_attempts=2,
            base_delay_seconds=0.001,
            max_delay_seconds=0.002,
            max_elapsed_seconds=1,
            attempt_timeout_seconds=0.2,
        )

    async def test_success_is_not_repeated(self):
        """A confirmed success returns unchanged after exactly one invocation."""
        calls = 0
        expected = object()

        async def operation():
            """Return a distinct object whose identity can be checked."""
            nonlocal calls
            calls += 1
            return expected

        result = await run_with_retries(
            operation, deadline=self.loop.time() + 2, policy=self.policy
        )
        self.assertIs(result, expected)
        self.assertEqual(calls, 1)

    async def test_safe_transient_failure_gets_a_fresh_call(self):
        """One retryable failure can recover on a newly created awaitable."""
        calls = 0

        async def operation():
            """Fail once and then produce a confirmed result."""
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RetryableError("Temporary dependency failure")
            return "done"

        result = await run_with_retries(
            operation, deadline=self.loop.time() + 2, policy=self.policy
        )
        self.assertEqual(result, "done")
        self.assertEqual(calls, 2)

    async def test_attempt_limit_reraises_original_error(self):
        """Exhaustion must not hide the classified failure inside RetryError."""
        calls = 0
        error = RetryableError("Still unavailable")

        async def operation():
            """Count calls while reporting the same known transient failure."""
            nonlocal calls
            calls += 1
            raise error

        with self.assertRaises(RetryableError) as caught:
            await run_with_retries(
                operation, deadline=self.loop.time() + 2, policy=self.policy
            )
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, 2)

    async def test_nonretryable_categories_propagate_once(self):
        """Lease loss and unsafe/unknown failures must not trigger automatic calls."""
        for error in (
            LeaseLostError("Ownership lost"),
            UncertainOutcomeError("Write outcome unknown"),
            PermanentInputError("Unsupported input"),
            OperatorActionRequiredError("Dependency access denied"),
            ValueError("Programming failure"),
            TimeoutError("Unclassified transport timeout"),
        ):
            with self.subTest(category=type(error).__name__):
                calls = 0

                async def operation():
                    """Raise the selected category without provider-side effects."""
                    nonlocal calls
                    calls += 1
                    raise error

                with self.assertRaises(type(error)) as caught:
                    await run_with_retries(
                        operation, deadline=self.loop.time() + 2, policy=self.policy
                    )
                self.assertIs(caught.exception, error)
                self.assertEqual(calls, 1)

    async def test_provider_minimum_delay_is_observed(self):
        """A provider's wait must survive a smaller random backoff result."""
        call_times = []
        minimum = 0.02

        async def operation():
            """Throttle once, recording when each actual invocation starts."""
            call_times.append(self.loop.time())
            if len(call_times) == 1:
                raise RateLimitedError("Throttled", retry_after_seconds=minimum)
            return "done"

        with patch(
            "rag_ingest.retry.wait_random_exponential", return_value=lambda state: 0
        ):
            await run_with_retries(
                operation, deadline=self.loop.time() + 2, policy=self.policy
            )
        self.assertEqual(len(call_times), 2)
        self.assertGreaterEqual(call_times[1] - call_times[0], minimum)

    async def test_long_provider_delay_returns_to_durable_retry_path(self):
        """Do not shorten a required delay or occupy the worker waiting too long."""
        calls = 0
        error = RateLimitedError("Wait longer", retry_after_seconds=30)

        async def operation():
            """Report a delay that cannot fit the short-call policy."""
            nonlocal calls
            calls += 1
            raise error

        with self.assertRaises(RateLimitedError) as caught:
            await asyncio.wait_for(
                run_with_retries(
                    operation, deadline=self.loop.time() + 2, policy=self.policy
                ),
                timeout=0.5,
            )
        self.assertIs(caught.exception, error)
        self.assertEqual(calls, 1)

    async def test_expired_deadline_does_not_start_operation(self):
        """No request may start after its known deadline has already passed."""
        calls = 0

        async def operation():
            """Detect any accidental invocation after expiry."""
            nonlocal calls
            calls += 1

        with self.assertRaises(TimeoutError):
            await run_with_retries(
                operation, deadline=self.loop.time() - 1, policy=self.policy
            )
        self.assertEqual(calls, 0)

    async def assert_cooperative_timeout(self, *, deadline_seconds, policy):
        """Check that a pending operation is cancelled and never automatically retried."""
        calls = 0
        cleaned_up = asyncio.Event()

        async def operation():
            """Wait indefinitely until the helper cancels this awaitable."""
            nonlocal calls
            calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                cleaned_up.set()

        with self.assertRaises(TimeoutError):
            await run_with_retries(
                operation, deadline=self.loop.time() + deadline_seconds, policy=policy
            )
        self.assertEqual(calls, 1)
        self.assertTrue(cleaned_up.is_set())

    async def test_per_call_timeout_stops_pending_work(self):
        """The per-attempt timeout applies even when the outer budget is larger."""
        await self.assert_cooperative_timeout(
            deadline_seconds=2,
            policy=replace(self.policy, attempt_timeout_seconds=0.03),
        )

    async def test_caller_deadline_overrides_larger_policy(self):
        """The earlier caller deadline must clamp the current call."""
        await self.assert_cooperative_timeout(deadline_seconds=0.03, policy=self.policy)

    async def test_total_policy_budget_overrides_later_caller_deadline(self):
        """A long caller deadline must not bypass the helper's elapsed-time cap."""
        await self.assert_cooperative_timeout(
            deadline_seconds=2,
            policy=replace(
                self.policy, max_elapsed_seconds=0.03, attempt_timeout_seconds=0.03
            ),
        )

    async def test_external_cancellation_reaches_caller(self):
        """Worker shutdown must not become another retry attempt."""
        entered = asyncio.Event()
        cleaned_up = asyncio.Event()
        calls = 0

        async def operation():
            """Signal startup and wait for the test to cancel the owning task."""
            nonlocal calls
            calls += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned_up.set()

        task = asyncio.create_task(
            run_with_retries(
                operation,
                deadline=self.loop.time() + 2,
                policy=self.policy,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=0.5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(cleaned_up.is_set())
            self.assertEqual(calls, 1)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
