"""Define how the publisher and worker interact with a queue adapter.

Assume notifications can be delivered more than once. PostgreSQL owns job
eligibility, leases, retry scheduling, and completion. Queue delivery alone
must never grant permission to process or publish a generation.
"""

from dataclasses import dataclass, field
from typing import Protocol

from rag_ingest.domain.events import JobAvailable


@dataclass(frozen=True, kw_only=True)
class QueueDelivery:
    """Hold one received message and the receipt needed to manage that delivery.

    body is raw text so the runner can inspect malformed notifications and
    apply its failure policy. receipt_handle belongs to this delivery, not the
    database job. A redelivery must use its newly returned receipt handle.
    receive_count is optional transport metadata, not the job attempt count.
    """

    message_id: str
    body: str = field(repr=False)
    receipt_handle: str = field(repr=False)
    receive_count: int | None = None

    def __post_init__(self) -> None:
        """Check transport fields without assuming the body is a valid event.

        Do not log raw bodies or receipt handles. Hiding them from repr reduces
        accidental exposure but does not redact direct access or exceptions.
        """
        for name in ("message_id", "receipt_handle"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")
        if not isinstance(self.body, str):
            raise ValueError("body must be a string")
        if self.receive_count is not None:
            if type(self.receive_count) is not int or self.receive_count < 1:
                raise ValueError("receive_count must be a positive integer")


class Queue(Protocol):
    """Specify the operations every supported queue adapter must implement.

    This is an interface, not a queue implementation. Implementations must
    bound network calls and map known failures to contracts.errors categories.
    An uncertain send or acknowledgment must not be treated as confirmed
    failure. Adapters must validate their supported parameter ranges and must
    not block the async event loop with synchronous network calls.
    """

    async def publish(self, event: JobAvailable) -> None:
        """Send this event and return only after queue acceptance is confirmed.

        Serialize with event.to_json(). Retain the event ID on retries. Raise
        UncertainOutcomeError if acceptance cannot be determined. Returning
        does not mean a worker has received or processed the event.
        """
        ...

    async def receive(
        self,
        *,
        max_messages: int,
        wait_seconds: int,
        visibility_seconds: int,
    ) -> tuple[QueueDelivery, ...]:
        """Receive at most max_messages with temporary delivery visibility.

        Return an empty tuple when no messages are returned; this does not
        prove the queue has no backlog. The runner must bound concurrency and
        claim each eligible job in PostgreSQL before starting its work.
        """
        ...

    async def acknowledge(self, delivery: QueueDelivery) -> None:
        """Request deletion using this delivery's current receipt handle.

        Call only after a durable outcome or continuation permits it. Queue
        acknowledgment does not update job status, and duplicate delivery must
        remain safe even after a successful call. Surface failures for recovery.
        """
        ...

    async def extend_visibility(self, delivery: QueueDelivery, *, seconds: int) -> None:
        """Ask the queue to keep this delivery hidden for another interval.

        This does not renew the PostgreSQL job lease or stop an old worker.
        The runner must maintain both separately and enforce total deadlines.
        Surface receipt or network failures rather than reporting false success.
        """
        ...

    async def aclose(self) -> None:
        """Release resources owned by this adapter during application shutdown.

        Repeated close calls must be safe. Do not acknowledge pending messages
        as part of cleanup; their disposition depends on durable job state.
        """
        ...


class BatchQueue(Queue, Protocol):
    """Optional: send several events in one call (SQS: up to 10).

    The outbox publisher uses this when present and otherwise calls publish
    once per event. Each event gets its own outcome so one refused message
    never blocks or duplicates the others' bookkeeping.
    """

    async def publish_batch(
        self, events: tuple[JobAvailable, ...]
    ) -> tuple[Exception | None, ...]:
        """One outcome per event, in order: None when the queue accepted it,
        else the classified error (UncertainOutcomeError when unknown)."""
        ...
