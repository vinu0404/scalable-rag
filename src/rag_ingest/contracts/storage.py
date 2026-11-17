"""Define storage operations without depending on a provider SDK.

Use immutable object references for source files and processing outputs.
Tenant authorization and reference-safe cleanup belong in the service layer.
Presigned uploads and multipart transfers have a separate contract.
"""

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit


@dataclass(frozen=True, kw_only=True)
class ObjectRef:
    """Identify an object by bucket, key, and optional provider version ID.

    A version ID pins the intended bytes. Without one, the deployment must
    enforce a write-once key for accepted inputs and saved outputs. The record
    itself does not enforce immutability or authorize access to that location.
    """

    bucket: str
    key: str
    version_id: str | None = None

    def __post_init__(self) -> None:
        """Reject empty storage addresses before making a provider request."""
        for name in ("bucket", "key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")
        if self.version_id is not None:
            if not isinstance(self.version_id, str) or not self.version_id.strip():
                raise ValueError("version_id must not be blank when provided")


@dataclass(frozen=True, kw_only=True)
class ObjectInfo:
    """Describe an inspected object's exact reference and reported byte count.

    Size alone does not verify content. The verification service must compute
    or establish a trusted checksum for these exact bytes. Do not treat an
    opaque provider identifier such as an ETag as the file's SHA-256.
    """

    ref: ObjectRef
    size_bytes: int

    def __post_init__(self) -> None:
        """Require an object reference and a nonnegative byte count."""
        if not isinstance(self.ref, ObjectRef):
            raise ValueError("ref must be an ObjectRef")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class PresignedRead:
    """Return a temporary GET URL for viewing or downloading one exact object.

    Keep the URL out of logs because possession grants temporary access.
    expires_at is the latest intended expiry; actual availability can end
    earlier if credentials expire or the object becomes unavailable.
    """

    url: str = field(repr=False)
    expires_at: datetime

    def __post_init__(self) -> None:
        """Check URL format and expiry timezone without opening the URL.

        This does not verify its signature, reachability, or caller permission.
        """
        try:
            parsed = urlsplit(self.url)
            valid = (
                isinstance(self.url, str)
                and parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and (parsed.port is None or 1 <= parsed.port <= 65535)
            )
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("url must be a valid HTTP or HTTPS URL")
        if (
            not isinstance(self.expires_at, datetime)
            or self.expires_at.tzinfo is None
            or self.expires_at.utcoffset() is None
        ):
            raise ValueError("expires_at must be a datetime with a timezone")


class ObjectStore(Protocol):
    """Specify bounded, async storage operations for adapters to implement.

    Map known provider failures to contracts.errors categories. A timeout
    after sending a write or delete may have an uncertain outcome. Validate
    parameters and reject unsupported correctness requirements explicitly.
    Never weaken immutable writes or silently read a different object version.
    Synchronous SDK network calls must not block the async event loop.
    """

    async def stat(self, ref: ObjectRef) -> ObjectInfo | None:
        """Inspect this object, returning None only for confirmed absence.

        Resolve the exact version when supplied; include the resolved version
        in ObjectInfo when available. Access denial and network failure must
        raise classified errors rather than being mistaken for a missing file.
        """
        ...

    def open_reader(
        self, ref: ObjectRef, *, block_size: int
    ) -> AbstractAsyncContextManager[AsyncIterator[bytes]]:
        """Open a bounded stream for use with async with and async for.

        block_size must be positive. Yield blocks no larger than that size
        without buffering the whole file. Release the connection on normal
        exit, cancellation, or an early stop. Stream failures must surface;
        they must not look like successful end-of-file.
        """
        ...

    async def put_immutable(
        self,
        *,
        bucket: str,
        key: str,
        blocks: AsyncIterable[bytes],
        expected_size_bytes: int,
        expected_sha256: str,
    ) -> ObjectInfo:
        """Save verified output without replacing different existing content.

        Consume blocks with bounded memory and verify the expected size and
        hash. Return only after a complete object is stored. An existing key
        may be reused only after verifying identical bytes; reject a mismatch.
        Creation must handle concurrent writers safely, not check-then-write.
        Use pinned versions or enforce write-once keys for subsequent reads.

        If the response is lost, verify the intended object before deciding
        what happened. Retrying requires a fresh iterable of the same bytes;
        an already consumed stream cannot be assumed to replay itself.
        """
        ...

    async def presign_read(
        self,
        ref: ObjectRef,
        *,
        expires_seconds: int,
        download: bool = False,
        filename: str | None = None,
    ) -> PresignedRead:
        """Create a temporary GET URL for an already-authorized file request.

        The service must first check tenant access, document permissions, and
        deletion state, then resolve the permitted exact object reference.
        Do not accept unrestricted bucket/key values directly from the client.

        download=False requests inline viewing; True requests an attachment.
        Viewing also depends on browser support and a safe, validated content
        type. Validate and encode any display filename as response metadata.
        Enforce positive duration and deployment limits, and sign an address
        reachable by the client rather than an internal worker-only hostname.

        Treat issued URLs as temporary grants. Checking permission at issuance
        does not recheck it on every later read; immediate access-removal needs
        a separately designed delivery path. Signing does not prove existence.
        """
        ...

    async def delete(self, ref: ObjectRef) -> None:
        """Delete only the referenced object or version; confirmed absence is safe.

        The caller must first check retention and remaining references and
        coordinate cleanup with new attachments. Never delete other versions
        or a whole prefix implicitly. Surface uncertain outcomes for recovery.
        """
        ...

    async def aclose(self) -> None:
        """Release resources owned by this adapter; repeated calls must be safe.

        Closing must not silently delete objects or complete abandoned uploads.
        """
        ...


@runtime_checkable
class CleanupObjectStore(Protocol):
    """Erase authorized outputs; required by the janitor for physical deletion.

    Called only after a durable, reference-safe cleanup plan is frozen. An
    adapter whose retention is managed elsewhere must still implement these
    deliberately (for example as documented no-ops), never omit them.
    """

    async def delete_prefix_versions(self, bucket: str, prefix: str) -> None:
        """Delete every version under one tenants/<t>/generations/<g>/ prefix."""
        ...

    async def delete_all_key_versions(self, ref: ObjectRef) -> None:
        """Delete every version of exactly this key, not keys sharing its prefix."""
        ...


class StorageVerifiedObjectStore(ObjectStore, Protocol):
    """Optional: skip re-reading an upload whose SHA-256 storage already verified.

    When storage checked the client's declared full-object SHA-256 at write
    time (and refuses mismatched bytes), intake verification can use that
    digest and read only the file-type prefix, instead of downloading the
    whole file a second time. Stores without these methods keep the full read.
    """

    async def verified_sha256(self, ref: ObjectRef) -> tuple[int, str] | None:
        """(size, lowercase sha256 hex) verified by storage for this pinned
        version, or None when storage has no full-object SHA-256 for it."""
        ...

    async def read_prefix(self, ref: ObjectRef, length: int) -> bytes:
        """The first `length` bytes (fewer if the object is shorter)."""
        ...
