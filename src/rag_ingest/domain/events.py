"""Define small job notifications shared by the outbox and queue.

Send IDs rather than file contents or credentials. A notification asks a worker
to check a job; it does not grant permission to run or publish that job.
"""

import json
from dataclasses import dataclass
from typing import ClassVar
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class JobAvailable:
    """Tell a worker that a job may be ready for execution.

    Create event_id once when saving the outbox record and reuse it on send
    retries. job_id identifies the work; several notifications may reference
    the same job. Workers must check its tenant, status, retry time, lease,
    and document state in PostgreSQL before doing work.
    """

    event_id: UUID
    tenant_id: UUID
    job_id: UUID

    EVENT_TYPE: ClassVar[str] = "job_available"
    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Require UUID objects so malformed IDs fail before sending.

        This checks format only; it does not verify that the records exist.
        """
        for name in ("event_id", "tenant_id", "job_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")

    def to_json(self) -> str:
        """Convert this notification into JSON for storage or queue delivery.

        Include a type and format version so receivers can reject messages
        they do not understand. UUIDs are encoded as strings.
        """
        return json.dumps(
            {
                "event_type": self.EVENT_TYPE,
                "schema_version": self.SCHEMA_VERSION,
                "event_id": str(self.event_id),
                "tenant_id": str(self.tenant_id),
                "job_id": str(self.job_id),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, body: str) -> "JobAvailable":
        """Read a version-1 notification and reject invalid message formats.

        Raise ValueError without repeating message contents in the error.
        The future worker runner must decide how to handle invalid messages.
        """
        try:
            payload = json.loads(body)
        except (TypeError, ValueError, RecursionError):
            raise ValueError("Job notification must contain valid JSON") from None

        expected = {"event_type", "schema_version", "event_id", "tenant_id", "job_id"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("Job notification has missing or unexpected fields")
        if payload["event_type"] != cls.EVENT_TYPE:
            raise ValueError("Unsupported job notification type")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != cls.SCHEMA_VERSION
        ):
            raise ValueError("Unsupported job notification schema version")

        identifiers = {}
        for name in ("event_id", "tenant_id", "job_id"):
            value = payload[name]
            if not isinstance(value, str):
                raise ValueError(f"Notification {name} must be a UUID string")
            try:
                identifiers[name] = UUID(value)
            except ValueError:
                raise ValueError(f"Notification {name} must be a UUID string") from None
        return cls(**identifiers)
