"""Track uploads before their files become verified source versions.

An upload session helps clients retry requests and lets the system recover
unfinished uploads. These records do not upload bytes or verify storage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique
from uuid import UUID


@unique
class UploadStatus(StrEnum):
    """Name the steps of accepting a file, separately from ingestion job status.

    VERIFYING means completion was reported and storage still needs checking.
    ACCEPTED means verification and durable ingestion registration succeeded;
    it does not mean the document has finished indexing.
    """

    UPLOADING = "uploading"
    VERIFYING = "verifying"
    ACCEPTED = "accepted"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True, kw_only=True)
class UploadSession:
    """Remember one intended file upload and its assigned storage location.

    request_key lets a client retry the same initiation request. The database
    must enforce tenant-scoped uniqueness and reject changed request details
    under the same key. Keep upload_id distinct from a provider's multipart ID.

    expected_size_bytes and expected_sha256 describe the client's expectation;
    neither is trusted proof of the stored file. Verification must inspect the
    exact completed object before creating a SourceVersion and ingestion job.

    expires_at is the deadline for finishing this intake session. Expiry or
    cancellation must be saved in the database; this object runs no timers.
    """

    tenant_id: UUID
    document_id: UUID
    upload_id: UUID
    request_key: str = field(repr=False)
    original_name: str
    bucket: str
    object_key: str
    expected_size_bytes: int
    created_at: datetime
    expires_at: datetime
    status: UploadStatus = UploadStatus.UPLOADING
    expected_sha256: str | None = None
    multipart_upload_id: str | None = field(default=None, repr=False)
    source_version_id: UUID | None = None
    # Keep this file even when a live document already has identical bytes.
    allow_duplicate: bool = False
    # Set when the upload was refused because that document has its bytes.
    duplicate_of: UUID | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent upload details before other components use them.

        Accepted sessions must point to a source version. Other states cannot
        claim acceptance. These checks do not verify bytes, authorize a user,
        enforce uniqueness, or make database transitions atomic.
        """
        for name in ("tenant_id", "document_id", "upload_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        for name in ("request_key", "original_name", "bucket", "object_key"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if type(self.expected_size_bytes) is not int or self.expected_size_bytes < 0:
            raise ValueError("expected_size_bytes must be a nonnegative integer")
        for name in ("created_at", "expires_at"):
            value = getattr(self, name)
            if (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must be a datetime with a timezone")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        if not isinstance(self.status, UploadStatus):
            raise ValueError("status must be an UploadStatus")
        if self.expected_sha256 is not None:
            value = self.expected_sha256
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(
                    "expected_sha256 must be a lowercase SHA-256 hex digest"
                )
        if (
            self.multipart_upload_id is not None
            and not self.multipart_upload_id.strip()
        ):
            raise ValueError("multipart_upload_id must not be blank when provided")
        if self.source_version_id is not None and not isinstance(
            self.source_version_id, UUID
        ):
            raise ValueError("source_version_id must be a UUID")
        if type(self.allow_duplicate) is not bool:
            raise ValueError("allow_duplicate must be a boolean")
        if self.duplicate_of is not None and (
            not isinstance(self.duplicate_of, UUID)
            or self.status is not UploadStatus.FAILED
        ):
            raise ValueError("Only a failed upload can name the document it duplicates")
        if self.status is UploadStatus.ACCEPTED:
            if self.source_version_id is None:
                raise ValueError("Accepted uploads need source_version_id")
        elif self.source_version_id is not None:
            raise ValueError(
                "Only accepted uploads can reference a registered source version"
            )


def request_fingerprint(descriptor):
    """Hash a client upload request for idempotency-key binding.

    descriptor holds only client-chosen fields; the server recipe is not part
    of it, so a replay after a configuration change still finds its upload.
    """
    import hashlib
    import json

    return hashlib.sha256(json.dumps(dict(descriptor), sort_keys=True).encode()).hexdigest()
