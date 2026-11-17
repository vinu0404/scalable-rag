"""Keep a document, its uploaded files, and its processing runs separate.

This lets us replace a file or rebuild its search data without losing history.
These classes hold data; they do not upload files or write to the database.
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_ingest.domain.limits import (
    MAX_METADATA_KEY_CHARS,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_CHARS,
    MAX_TAG_CHARS,
    MAX_TAGS,
)

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.:/-]*$")


def normalize_tags(tags) -> tuple[str, ...]:
    """Validated, de-duplicated, sorted tags (exact, case-sensitive matching)."""
    if isinstance(tags, (str, bytes)) or not isinstance(tags, (list, tuple)):
        raise ValueError("tags must be a list of strings")
    result = set()
    for tag in tags:
        if (
            not isinstance(tag, str)
            or not 1 <= len(tag) <= MAX_TAG_CHARS
            or not _LABEL.match(tag)
        ):
            raise ValueError(
                f"Each tag must be 1..{MAX_TAG_CHARS} characters: letters, digits,"
                " space, _ . : / -, starting with a letter or digit"
            )
        result.add(tag)
    if len(result) > MAX_TAGS:
        raise ValueError(f"At most {MAX_TAGS} tags")
    return tuple(sorted(result))


def normalize_metadata(metadata) -> tuple[tuple[str, object], ...]:
    """Validated metadata as sorted (key, value) pairs of flat scalar values.

    Values may be strings, integers, finite numbers or booleans; nesting is
    refused so filters stay simple exact matches.
    """
    if isinstance(metadata, tuple):
        metadata = dict(metadata)
    if not isinstance(metadata, dict) or len(metadata) > MAX_METADATA_KEYS:
        raise ValueError(f"metadata must be an object with at most {MAX_METADATA_KEYS} keys")
    pairs = []
    for key, value in metadata.items():
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= MAX_METADATA_KEY_CHARS
            or not _LABEL.match(key)
        ):
            raise ValueError(f"Invalid metadata key: 1..{MAX_METADATA_KEY_CHARS} label characters")
        if isinstance(value, str):
            if len(value) > MAX_METADATA_VALUE_CHARS:
                raise ValueError(f"Metadata values are at most {MAX_METADATA_VALUE_CHARS} characters")
        elif isinstance(value, bool) or isinstance(value, int):
            pass
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Metadata numbers must be finite")
        else:
            raise ValueError("Metadata values must be strings, numbers or booleans")
        pairs.append((key, value))
    return tuple(sorted(pairs))


def _require_aware(name: str, value: datetime) -> None:
    """Reject a time without a timezone so times can be compared reliably.

    Raise ValueError using the field name if its timezone is missing.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


def _require_sha256(name: str, value: str) -> None:
    """Check that a hash has 64 lowercase hexadecimal characters.

    Raise ValueError for an invalid format. This does not calculate the hash
    or check whether it matches a file or processing configuration.
    """
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, kw_only=True)
class Document:
    """Describe the document the user sees, such as 'Employee handbook'.

    Keep the same document ID when a revised file is uploaded. Store its name,
    access-policy reference, current search generation, and deletion time here.
    This is a copy of state; creating an object does not update the database.
    """

    tenant_id: UUID
    document_id: UUID
    display_name: str
    acl_ref: str
    created_at: datetime
    active_generation_id: UUID | None = None
    desired_generation_id: UUID | None = None
    deleted_at: datetime | None = None
    # Labels for search filters; set at upload, changed with PATCH.
    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, object], ...] = ()

    @property
    def metadata_dict(self) -> dict:
        return dict(self.metadata)

    def __post_init__(self) -> None:
        """Check names and times, and reject deleted documents still active in search.

        Runs after object creation and raises ValueError for invalid data.
        Database operations must also enforce these rules when saving changes.
        """
        if not self.display_name.strip() or not self.acl_ref.strip():
            raise ValueError("display_name and acl_ref must not be blank")
        _require_aware("created_at", self.created_at)
        if normalize_tags(self.tags) != self.tags:
            raise ValueError("tags must be normalized (normalize_tags)")
        if normalize_metadata(self.metadata) != self.metadata:
            raise ValueError("metadata must be normalized (normalize_metadata)")
        if self.deleted_at is not None:
            _require_aware("deleted_at", self.deleted_at)
            if self.active_generation_id is not None:
                raise ValueError("A deleted document cannot have an active generation")


@dataclass(frozen=True, kw_only=True)
class SourceVersion:
    """Describe one verified uploaded file belonging to a document.

    Record its storage location, byte count, and file hash. Uploading revised
    content creates another source version so workers can read the exact file.
    Storage must pin an object version or prevent overwriting the object key;
    this Python class cannot protect the remote file from being changed.

    artifact_id links this upload to reusable verified content. Identical
    uploads in the same tenant may share it while keeping separate version
    IDs and original storage references. Retention of originals follows policy.
    The registration service must check that the artifact belongs to this
    tenant and matches this file's verified hash and size before linking it.
    """

    tenant_id: UUID
    document_id: UUID
    source_version_id: UUID
    artifact_id: UUID
    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    created_at: datetime
    object_version_id: str | None = None

    def __post_init__(self) -> None:
        """Check the artifact ID, storage references, size, hash, and time.

        This checks the record only; it does not contact storage or read a file.
        A UUID value does not prove that the referenced artifact exists.
        """
        if not isinstance(self.artifact_id, UUID):
            raise ValueError("artifact_id must be a UUID")
        if not self.bucket.strip() or not self.object_key.strip():
            raise ValueError("bucket and object_key must not be blank")
        if self.object_version_id is not None and not self.object_version_id.strip():
            raise ValueError("object_version_id must not be blank when provided")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        _require_sha256("sha256", self.sha256)
        _require_aware("created_at", self.created_at)


@dataclass(frozen=True, kw_only=True)
class ProcessingGeneration:
    """Identify a build of search data from one source version and set of settings.

    Create a new generation when rebuilding, for example with another embedding
    model. Keep old search data available until the new build is safe to publish.
    pipeline_fingerprint is a hash of the versioned processing settings, not
    the file bytes. The database must also retain the full settings themselves.
    This record does not mean the build has finished or is searchable.
    """

    tenant_id: UUID
    document_id: UUID
    source_version_id: UUID
    generation_id: UUID
    pipeline_fingerprint: str
    created_at: datetime

    def __post_init__(self) -> None:
        """Check the settings-hash format and creation time.

        This does not run processing or verify that its outputs are complete.
        """
        _require_sha256("pipeline_fingerprint", self.pipeline_fingerprint)
        _require_aware("created_at", self.created_at)
