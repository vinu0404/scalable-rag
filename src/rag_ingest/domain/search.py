"""Search requests and results, independent of any vector or keyword engine.

Indexes return chunk IDs and scores only. Text, page spans and file names come
from the authoritative chunk table after the document is checked to be live
for the tenant, so no index can return deleted, superseded or foreign text.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_ingest.domain.documents import normalize_metadata, normalize_tags

MAX_FILTER_DOCUMENTS = 1000


@dataclass(frozen=True, kw_only=True)
class SearchFilters:
    """Restrict a search to documents matching every given condition.

    document_ids: only these documents. tags_any: at least one of these tags.
    tags_all: every one of these tags. metadata: each key has exactly this
    value. created_after / created_before: document creation time range
    (inclusive / exclusive). Empty fields do not restrict anything.
    """

    document_ids: tuple[UUID, ...] = ()
    tags_any: tuple[str, ...] = ()
    tags_all: tuple[str, ...] = ()
    metadata: tuple[tuple[str, object], ...] = ()
    created_after: datetime | None = None
    created_before: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_ids, tuple) or any(
            not isinstance(d, UUID) for d in self.document_ids
        ):
            raise ValueError("document_ids must be a tuple of UUIDs")
        if len(self.document_ids) > MAX_FILTER_DOCUMENTS:
            raise ValueError(f"At most {MAX_FILTER_DOCUMENTS} document_ids")
        for name in ("tags_any", "tags_all"):
            if normalize_tags(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be normalized tags")
        if normalize_metadata(self.metadata) != self.metadata:
            raise ValueError("metadata must be normalized")
        for name in ("created_after", "created_before"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, datetime) or value.utcoffset() is None
            ):
                raise ValueError(f"{name} must be a datetime with a timezone")
        if (
            self.created_after
            and self.created_before
            and self.created_after >= self.created_before
        ):
            raise ValueError("created_after must be earlier than created_before")

    @classmethod
    def build(cls, *, document_ids=(), tags_any=(), tags_all=(), metadata=None, **times):
        """Normalize caller input (lists, dicts) into a validated filter."""
        return cls(
            document_ids=tuple(document_ids),
            tags_any=normalize_tags(list(tags_any)),
            tags_all=normalize_tags(list(tags_all)),
            metadata=normalize_metadata(metadata or {}),
            **times,
        )

    @property
    def active(self) -> bool:
        return any(
            (
                self.document_ids,
                self.tags_any,
                self.tags_all,
                self.metadata,
                self.created_after,
                self.created_before,
            )
        )


@dataclass(frozen=True, kw_only=True)
class SearchResult:
    """One returned passage, with every other place the same text appears."""

    vector_id: str
    document_id: UUID
    generation_id: UUID
    filename: str
    ordinal: int
    text: str | None  # None only inside the pipeline, for a hit with no chunk row
    spans: tuple[dict, ...]
    score: float
    text_sha256: str
    # (document_id, filename, page numbers) of identical passages collapsed into this one
    also_in: tuple[tuple[UUID, str, tuple[int, ...]], ...] = ()
    # (retriever, rank) for each retriever that returned it, e.g. (("keyword", 1),)
    ranks: tuple[tuple[str, int], ...] = ()

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({s["page_number"] for s in self.spans if s.get("page_number")}))


def chunk_digest(pairs) -> str:
    """Digest of (vector_id, text) pairs, independent of order.

    One line per chunk sorted by ID: the ID and the SHA-256 of its UTF-8 text.
    The metadata store computes the same over stored rows, so equal digests
    mean every expected chunk is stored with exactly its text.
    """
    lines = sorted(
        f"{vector_id}\t{hashlib.sha256(text.encode('utf-8')).hexdigest()}\n"
        for vector_id, text in pairs
    )
    return hashlib.sha256("".join(lines).encode()).hexdigest()
