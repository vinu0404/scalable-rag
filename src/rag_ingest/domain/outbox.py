"""Track notifications that must be sent to the queue.

The database must save a job and its outbox record in one transaction.
These classes describe the records; they do not save or send anything.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_ingest.domain.events import JobAvailable


def _check_time(name: str, value: datetime) -> None:
    """Require a real datetime with a timezone for scheduling comparisons."""
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be a datetime with a timezone")


@dataclass(frozen=True, kw_only=True)
class PublisherClaim:
    """Describe one publisher's temporary right to send an outbox record.

    Use a fresh claim_id on each successful database claim. Database updates
    must check this ID and expiry before renewing or recording a send outcome.
    This is separate from the job lease held by a processing worker.
    """

    publisher_id: str
    claim_id: UUID
    expires_at: datetime

    def __post_init__(self) -> None:
        """Reject an empty publisher name, invalid claim ID, or invalid time.

        Expired claims remain readable so another publisher can recover them.
        """
        if not self.publisher_id.strip():
            raise ValueError("publisher_id must not be blank")
        if not isinstance(self.claim_id, UUID):
            raise ValueError("claim_id must be a UUID")
        _check_time("expires_at", self.expires_at)


@dataclass(frozen=True, kw_only=True)
class OutboxRecord:
    """Remember a notification until successful queue delivery is recorded.

    event.event_id is this record's identity; retain it across send retries.
    available_at is the earliest time a publisher may claim unpublished work.
    publish_attempts counts successful publisher claims, not confirmed sends.
    published_at means the queue accepted a send and we saved that outcome.
    It does not mean the worker has received or completed the job.

    A crash after sending but before saving published_at may cause another
    send. The worker must still handle duplicate notifications safely.
    """

    event: JobAvailable
    created_at: datetime
    available_at: datetime
    publish_attempts: int = 0
    claim: PublisherClaim | None = None
    published_at: datetime | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent record fields before a component uses them.

        Claimed or published records need at least one publisher attempt.
        Publishing releases the claim. Atomic database operations must enforce
        ownership and transitions; creating this object does not enforce them.
        """
        if not isinstance(self.event, JobAvailable):
            raise ValueError("event must be a JobAvailable notification")
        if type(self.publish_attempts) is not int or self.publish_attempts < 0:
            raise ValueError("publish_attempts must be a nonnegative integer")
        _check_time("created_at", self.created_at)
        _check_time("available_at", self.available_at)
        if self.claim is not None:
            if not isinstance(self.claim, PublisherClaim):
                raise ValueError("claim must be a PublisherClaim")
            if self.publish_attempts == 0:
                raise ValueError(
                    "Claimed records need a positive publish_attempts count"
                )
        if self.published_at is not None:
            _check_time("published_at", self.published_at)
            if self.claim is not None:
                raise ValueError("Published records must release their publisher claim")
            if self.publish_attempts == 0:
                raise ValueError(
                    "Published records need a positive publish_attempts count"
                )
