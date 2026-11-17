"""Describe text chunks and where their text came from.

Chunk IDs must stay stable across retries of the same generation. Source
references let retrieval show citations without losing document attribution.
"""

import hashlib
from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class SourceSpan:
    """Locate source text inside one saved extraction output.

    extraction_sha256 identifies the exact UTF-8 extracted text. start_char
    is inclusive and end_char is exclusive, using Python string positions,
    not PDF byte positions. page_number is optional and starts at one.
    A chunk spanning multiple pages can carry several spans.
    """

    extraction_sha256: str
    start_char: int
    end_char: int
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Check the hash, character range, and optional page number.

        The chunker must also check these positions against the actual saved
        extraction and its page map. This record does not read those files.
        """
        if (
            not isinstance(self.extraction_sha256, str)
            or len(self.extraction_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.extraction_sha256)
        ):
            raise ValueError("extraction_sha256 must be a lowercase SHA-256 hex digest")
        if (
            type(self.start_char) is not int
            or type(self.end_char) is not int
            or self.start_char < 0
            or self.end_char <= self.start_char
        ):
            raise ValueError("Source span must have 0 <= start_char < end_char")
        if self.page_number is not None:
            if type(self.page_number) is not int or self.page_number < 1:
                raise ValueError("page_number must be a positive integer")


@dataclass(frozen=True, kw_only=True)
class Chunk:
    """Describe one piece of text to embed and index for a generation.

    ordinal starts at zero and is assigned by deterministic chunking. Retries
    must reproduce the same text and order from the same input and settings;
    changing the recipe requires another generation.

    Keep document and source-version IDs even when content artifacts are
    shared. Matching text must never merge access rules or citation history.
    spans describe source locations; transformed chunk text may differ from
    the literal source slices, so adapters must preserve that mapping.
    """

    tenant_id: UUID
    document_id: UUID
    source_version_id: UUID
    generation_id: UUID
    ordinal: int
    text: str = field(repr=False)
    spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        """Require valid identities, nonblank text, and immutable source spans.

        This does not verify tenant relationships, page maps, or chunk order
        across a whole document. Services and the database must check those.
        """
        for name in ("tenant_id", "document_id", "source_version_id", "generation_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must not be blank")
        if (
            not isinstance(self.spans, tuple)
            or not self.spans
            or any(not isinstance(span, SourceSpan) for span in self.spans)
        ):
            raise ValueError("spans must be a nonempty tuple of SourceSpan records")

    @property
    def text_sha256(self) -> str:
        """Hash this chunk's exact UTF-8 text without changing whitespace.

        Text hashes can help check reuse, but embedding reuse must also check
        the complete embedding recipe. This is not the original file hash.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def vector_id(self) -> str:
        """Build the same logical vector ID every time this chunk is retried.

        A vector adapter can map this to its provider's supported ID format,
        but must preserve uniqueness and use the same mapping on every retry.
        Upserts and safe generation publication are still required.
        """
        return (
            f"{self.tenant_id}:{self.document_id}:{self.generation_id}:{self.ordinal}"
        )
