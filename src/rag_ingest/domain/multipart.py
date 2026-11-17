"""Describe a fixed multipart upload plan and observed completed parts.

Persist these records so a restart keeps the same part numbers and byte ranges.
Storage listings remain necessary to recover parts whose progress was not saved.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class MultipartPlan:
    """Remember how one provider transfer divides a file into byte ranges.

    The upload service chooses part_size_bytes from configuration and tested
    provider limits, then persists the plan before issuing part URLs. Keep it
    unchanged when resuming. A fresh provider transfer needs a new plan and
    must not reuse old transfer progress as if it belonged to the new one.

    Example: 10 GiB with 64 MiB parts has 160 parts. This class checks the math,
    not provider limits. Zero-byte files must use the direct-upload path.
    """

    tenant_id: UUID
    upload_id: UUID
    provider_upload_id: str = field(repr=False)
    file_size_bytes: int
    part_size_bytes: int
    total_parts: int

    def __post_init__(self) -> None:
        """Check identity, positive sizes, and the exact expected part count."""
        for name in ("tenant_id", "upload_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        if (
            not isinstance(self.provider_upload_id, str)
            or not self.provider_upload_id.strip()
        ):
            raise ValueError("provider_upload_id must not be blank")
        for name in ("file_size_bytes", "part_size_bytes", "total_parts"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        expected = (
            self.file_size_bytes + self.part_size_bytes - 1
        ) // self.part_size_bytes
        if self.total_parts != expected:
            raise ValueError("total_parts does not match the file and part sizes")

    def part_bounds(self, part_number: int) -> tuple[int, int]:
        """Return the start-inclusive, end-exclusive byte range for one part.

        A client can use these bounds to slice the original file without
        loading it all into memory. The final range can be shorter.
        """
        if type(part_number) is not int or not 1 <= part_number <= self.total_parts:
            raise ValueError("part_number is outside this upload plan")
        start = (part_number - 1) * self.part_size_bytes
        return start, min(start + self.part_size_bytes, self.file_size_bytes)

    def expected_part_size(self, part_number: int) -> int:
        """Calculate the byte count a storage-reported part should have."""
        start, end = self.part_bounds(part_number)
        return end - start


@dataclass(frozen=True, kw_only=True)
class ObservedUploadPart:
    """Remember a part reported by storage for a particular provider transfer.

    This records an observation, not permanent proof: a later part upload can
    replace that number's contents. Reconcile with storage before completion.
    The database must isolate progress by tenant, upload, provider transfer,
    and part number, and prevent stale observations overwriting newer decisions.
    """

    tenant_id: UUID
    upload_id: UUID
    provider_upload_id: str = field(repr=False)
    part_number: int
    etag: str
    size_bytes: int
    observed_at: datetime

    def __post_init__(self) -> None:
        """Check record fields without claiming the stored bytes are verified.

        ETag is an opaque completion token, not the file's SHA-256 digest.
        Size and ETag must come from a provider observation, not just a browser.
        """
        for name in ("tenant_id", "upload_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        for name in ("provider_upload_id", "etag"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")
        if type(self.part_number) is not int or self.part_number < 1:
            raise ValueError("part_number must be a positive integer")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        if (
            not isinstance(self.observed_at, datetime)
            or self.observed_at.tzinfo is None
            or self.observed_at.utcoffset() is None
        ):
            raise ValueError("observed_at must be a datetime with a timezone")

    def validate_against(self, plan: MultipartPlan) -> None:
        """Reject a part from another plan or with the wrong expected size.

        Call before saving or using this observation. This compares records;
        it does not read storage or check current upload-session state.
        """
        if not isinstance(plan, MultipartPlan):
            raise ValueError("plan must be a MultipartPlan")
        if (
            self.tenant_id != plan.tenant_id
            or self.upload_id != plan.upload_id
            or self.provider_upload_id != plan.provider_upload_id
        ):
            raise ValueError("Observed part belongs to a different upload plan")
        if self.size_bytes != plan.expected_part_size(self.part_number):
            raise ValueError("Observed part size does not match the upload plan")
