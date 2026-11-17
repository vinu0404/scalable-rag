"""Define direct and multipart upload operations for storage adapters.

The intake service authorizes each request and supplies its assigned object
location. Upload completion is followed by verification and registration;
it must not automatically enqueue unverified content.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from urllib.parse import urlsplit

from rag_ingest.contracts.storage import ObjectInfo, ObjectRef


@dataclass(frozen=True, kw_only=True)
class PresignedUpload:
    """Give a client a temporary PUT URL and the headers it must send.

    Use the URL and headers exactly as returned. Do not log them. Expiry is
    the latest intended deadline; temporary credentials can expire earlier.
    Uploading successfully does not prove the file passed intake checks.
    """

    url: str = field(repr=False)
    expires_at: datetime
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        """Check address, expiry, and immutable header pairs without signing anything."""
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
        if not isinstance(self.headers, tuple):
            raise ValueError("headers must be a tuple of string pairs")
        seen = set()
        for pair in self.headers:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(not isinstance(value, str) for value in pair)
                or not pair[0].strip()
                or any("\r" in value or "\n" in value for value in pair)
                or pair[0].lower() in seen
            ):
                raise ValueError(
                    "headers must contain distinct names and single-line string values"
                )
            seen.add(pair[0].lower())


@dataclass(frozen=True, kw_only=True)
class MultipartTransfer:
    """Identify the provider operation used to upload one object in parts.

    Persist provider_upload_id with our UploadSession after initiation. It is
    not our upload_id. The target must be a new assigned key without a version.
    """

    target: ObjectRef
    provider_upload_id: str = field(repr=False)

    def __post_init__(self) -> None:
        """Require an unversioned target and a nonblank provider transfer ID."""
        if not isinstance(self.target, ObjectRef) or self.target.version_id is not None:
            raise ValueError("target must be an ObjectRef without a version_id")
        if (
            not isinstance(self.provider_upload_id, str)
            or not self.provider_upload_id.strip()
        ):
            raise ValueError("provider_upload_id must not be blank")


@dataclass(frozen=True, kw_only=True)
class UploadedPart:
    """Describe a part reported by storage or supplied for completion.

    part_number starts at one. etag is the opaque provider completion token,
    not a SHA-256 digest. Client-supplied details must be checked against storage.
    """

    part_number: int
    etag: str
    size_bytes: int

    def __post_init__(self) -> None:
        """Check basic fields; adapters must also enforce provider part limits."""
        if type(self.part_number) is not int or self.part_number < 1:
            raise ValueError("part_number must be a positive integer")
        if not isinstance(self.etag, str) or not self.etag.strip():
            raise ValueError("etag must not be blank")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class PartPage:
    """Return a bounded page of parts so interrupted uploads can resume.

    next_after is the last returned part number when another page is needed;
    None means this listing is finished. Parts must be ordered and distinct.
    """

    parts: tuple[UploadedPart, ...]
    next_after: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid ordering or a continuation that cannot advance."""
        if not isinstance(self.parts, tuple) or any(
            not isinstance(p, UploadedPart) for p in self.parts
        ):
            raise ValueError("parts must be a tuple of UploadedPart records")
        numbers = [part.part_number for part in self.parts]
        if numbers != sorted(set(numbers)):
            raise ValueError("parts must have distinct, ascending part numbers")
        if self.next_after is not None:
            if (
                type(self.next_after) is not int
                or not numbers
                or self.next_after != numbers[-1]
            ):
                raise ValueError("next_after must equal the last returned part number")


class UploadStore(Protocol):
    """Specify upload operations without depending on an S3 provider SDK.

    Validate provider limits, durations, expected sizes, and checksum format.
    Enforce safe upload isolation: a still-valid upload URL must not change
    accepted source bytes. Pin the verified version or use an enforced
    write-once/finalization strategy; random keys alone are not protection.
    Map known failures to contracts.errors and bound all network operations.
    """

    async def presign_put(
        self,
        target: ObjectRef,
        *,
        expected_size_bytes: int,
        expected_sha256: str | None,
        expires_seconds: int,
    ) -> PresignedUpload:
        """Sign one direct PUT to an authorized session's assigned new key.

        Return required headers and a client-reachable address. Apply supported
        request constraints and declare limitations explicitly. The service
        must verify the completed object before accepting it, regardless of
        which constraints the provider can enforce during transfer.
        """
        ...

    async def begin_multipart(self, target: ObjectRef) -> MultipartTransfer:
        """Start a transfer, then let the service persist its provider ID.

        If creation succeeds but its response or database save is lost, the
        transfer may be orphaned. Reconciliation or storage lifecycle cleanup
        must handle it. Do not assume initiation is an idempotent operation.
        """
        ...

    async def presign_part(
        self,
        transfer: MultipartTransfer,
        *,
        part_number: int,
        expected_size_bytes: int,
        expected_sha256: str | None,
        expires_seconds: int,
    ) -> PresignedUpload:
        """Sign one part; refresh an expired URL for the same intended part.

        Retry with the same bytes and part number. The service must authorize
        the session and enforce bounded part count and total expected size.
        """
        ...

    async def list_parts(
        self,
        transfer: MultipartTransfer,
        *,
        after_part_number: int | None,
        limit: int,
    ) -> PartPage:
        """Inspect a bounded page of uploaded parts to recover transfer progress.

        Query the provider rather than trusting a browser's cached progress.
        This listing is not a transaction with concurrent part uploads.
        """
        ...

    async def complete_multipart(
        self,
        transfer: MultipartTransfer,
        *,
        parts: tuple[UploadedPart, ...],
    ) -> ObjectInfo:
        """Complete the checked, ordered part list and inspect the resulting object.

        Verify a complete provider response, not merely transport success.
        Resolve the exact resulting object/version. On an uncertain outcome,
        raise UncertainOutcomeError; the service must reconcile the intended
        object before completing again. Full-file verification still follows.
        """
        ...

    async def abort_multipart(self, transfer: MultipartTransfer) -> None:
        """Abort an unfinished transfer without deleting a completed source file.

        Coordinate with in-flight operations and verify cleanup as required.
        An absent transfer alone does not establish whether it completed or
        was aborted; reconcile session and object state before deciding.
        """
        ...

    async def aclose(self) -> None:
        """Release owned resources safely without completing or aborting transfers."""
        ...
