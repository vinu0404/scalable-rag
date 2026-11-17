"""Use Tenacity for short async call retries, not durable job scheduling.

Retry only explicitly classified safe failures. Adapters still need network
timeouts and correct handling of uncertain writes. Job leases, checkpoints,
and PostgreSQL retry scheduling remain the responsibility of the core.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    wait_random_exponential,
)

from rag_ingest.contracts.errors import RetryableError

T = TypeVar("T")


@dataclass(frozen=True, kw_only=True)
class CallRetryPolicy:
    """Limit retries inside one adapter call, separately from job attempts.

    max_attempts includes the first call. Total elapsed time includes calls
    and waits. Provider minimum delays are honored only if they fit the call
    deadline; otherwise the last error is returned to the durable retry path.
    These short defaults are illustrative and must account for SDK retries.
    """

    max_attempts: int = 2
    base_delay_seconds: float = 0.1
    max_delay_seconds: float = 1.0
    max_elapsed_seconds: float = 5.0
    attempt_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid counts, nonfinite durations, and inconsistent limits."""
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        for name in (
            "base_delay_seconds",
            "max_delay_seconds",
            "max_elapsed_seconds",
            "attempt_timeout_seconds",
        ):
            value = getattr(self, name)
            try:
                valid = (
                    type(value) in (int, float) and math.isfinite(value) and value > 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{name} must be finite and positive")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base_delay_seconds must not exceed max_delay_seconds")
        if self.attempt_timeout_seconds > self.max_elapsed_seconds:
            raise ValueError(
                "attempt_timeout_seconds must not exceed max_elapsed_seconds"
            )


async def run_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    deadline: float,
    policy: CallRetryPolicy | None = None,
) -> T:
    """Run a repeatable async operation within a fixed monotonic deadline.

    deadline uses asyncio.get_running_loop().time(), not wall-clock time.
    The runner must derive a conservative local deadline from its saved job,
    attempt, and lease limits. This helper cannot establish database ownership.
    Each call to operation must create a fresh awaitable and fresh/replayable
    request data; do not reuse an exhausted upload stream or coroutine object.

    Retry only RetryableError, including RateLimitedError. Uncertain outcomes,
    permanent input errors, bad credentials, lease loss, unknown errors, and
    cancellation propagate immediately. Deadline/attempt expiry raises a raw
    TimeoutError; the adapter must classify remote-write uncertainty before
    deciding on any further retry. Cancellation cannot undo remote effects.

    Async timeouts require cooperative cancellation. Do not pass blocking or
    cancellation-suppressing code and assume it will be stopped; use bounded
    SDK transport calls and isolated runtimes where needed. No job status,
    lease, queue visibility, or durable retry record is changed here.
    """
    selected = CallRetryPolicy() if policy is None else policy
    if not isinstance(selected, CallRetryPolicy):
        raise ValueError("policy must be a CallRetryPolicy")
    try:
        valid_deadline = type(deadline) in (int, float) and math.isfinite(deadline)
    except OverflowError:
        valid_deadline = False
    if not valid_deadline:
        raise ValueError("deadline must be a finite monotonic timestamp")
    loop = asyncio.get_running_loop()
    effective_deadline = min(deadline, loop.time() + selected.max_elapsed_seconds)
    if loop.time() >= effective_deadline:
        raise TimeoutError("Adapter-call deadline has already expired")
    jitter = wait_random_exponential(
        multiplier=selected.base_delay_seconds,
        max=selected.max_delay_seconds,
    )

    def wait_seconds(state: RetryCallState) -> float:
        """Use jittered backoff without shortening a provider's minimum delay."""
        error = state.outcome.exception() if state.outcome is not None else None
        minimum = (
            error.retry_after_seconds if isinstance(error, RetryableError) else None
        )
        return max(float(jitter(state)), minimum or 0.0)

    def stop(state: RetryCallState) -> bool:
        """Stop before another wait would consume the remaining call budget."""
        return (
            state.attempt_number >= selected.max_attempts
            or loop.time() + state.upcoming_sleep >= effective_deadline
        )

    async with asyncio.timeout_at(effective_deadline):
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(RetryableError),
            wait=wait_seconds,
            stop=stop,
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                if loop.time() >= effective_deadline:
                    raise TimeoutError(
                        "Adapter-call deadline expired before another call"
                    )
                attempt_deadline = min(
                    effective_deadline,
                    loop.time() + selected.attempt_timeout_seconds,
                )
                async with asyncio.timeout_at(attempt_deadline):
                    return await operation()
    raise RuntimeError("Retry loop ended without a result or error")
