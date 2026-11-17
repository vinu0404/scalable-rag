"""Describe verified file bytes that documents in one tenant may share.

Matching bytes can reuse storage without merging document names, access rules,
or history. The database will enforce uniqueness on (tenant_id, sha256).
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class ContentArtifact:
    """Identify one verified byte sequence and its retained storage location.

    Calculate sha256 from the actual completed file before creating this
    record. Separate source versions may reference the same artifact, while
    preserving each upload's original location and document-specific history.

    This class does not calculate hashes, find duplicates, or delete objects.
    The metadata adapter must handle concurrent registration with a unique
    constraint. Matching a hash must not grant access to another document.

    Pin object_version_id or enforce a write-once key in storage. Keep this
    object while any source version needs it. Cleanup must coordinate with
    new references so a concurrent upload cannot attach to deleted bytes.
    """

    tenant_id: UUID
    artifact_id: UUID
    sha256: str
    size_bytes: int
    bucket: str
    object_key: str
    created_at: datetime
    object_version_id: str | None = None

    def __post_init__(self) -> None:
        """Reject invalid IDs, hash format, size, storage references, or time.

        These checks validate the record only. They do not confirm that the
        object exists, its bytes match, or no duplicate row already exists.
        """
        for name in ("tenant_id", "artifact_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        for name in ("bucket", "object_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")
        if self.object_version_id is not None:
            if (
                not isinstance(self.object_version_id, str)
                or not self.object_version_id.strip()
            ):
                raise ValueError("object_version_id must not be blank when provided")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("created_at must be a datetime with a timezone")
