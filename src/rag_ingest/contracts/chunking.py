"""Define deterministic text chunking with preserved source references.

This contract consumes saved extraction, not the original PDF or upload parts.
The service owns artifact verification, job leases, and checkpoint commits.
"""

import math
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag_ingest.domain.chunks import Chunk
from rag_ingest.domain.documents import ProcessingGeneration, SourceVersion
from rag_ingest.domain.pipeline import ComponentSpec


@dataclass(frozen=True, kw_only=True)
class ChunkingInput:
    """Locate verified local copies of extracted text and its page map.

    extraction_sha256 hashes the exact UTF-8 text bytes. page_map_sha256 hashes
    the page-map file, whose versioned format records page boundaries in that
    text. The service must verify both files against their saved manifest and
    keep them unchanged while the chunker reads them. Paths are worker-owned,
    not arbitrary client-supplied filesystem addresses.
    """

    text_path: Path
    page_map_path: Path
    extraction_sha256: str
    page_map_sha256: str

    def __post_init__(self) -> None:
        """Check path and hash types without opening or verifying local files."""
        for name in ("text_path", "page_map_path"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a pathlib.Path")
        for name in ("extraction_sha256", "page_map_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, kw_only=True)
class ChunkingLimits:
    """Bound chunk count, each chunk's text length, and total chunking time.

    These are acceptance limits. Actual size, overlap, tokenizer, and splitting
    rules belong in the versioned recipe. Do not silently change those rules
    or truncate output to fit a runtime limit.
    """

    max_chunks: int
    max_chunk_characters: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Require positive integer limits and a finite positive time budget."""
        for name in ("max_chunks", "max_chunk_characters"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            valid = (
                type(self.timeout_seconds) in (int, float)
                and math.isfinite(self.timeout_seconds)
                and self.timeout_seconds > 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("timeout_seconds must be finite and positive")


class Chunker(Protocol):
    """Specify replaceable chunking without choosing a splitting algorithm.

    Implementations must validate driver, behavior version, and every option.
    The same verified input and recipe must produce the same ordered chunks.
    """

    def open_chunks(
        self,
        extraction: ChunkingInput,
        *,
        source: SourceVersion,
        generation: ProcessingGeneration,
        spec: ComponentSpec,
        limits: ChunkingLimits,
    ) -> AbstractAsyncContextManager[AsyncIterator[Chunk]]:
        """Yield bounded chunks with stable ordinals and accurate SourceSpan records.

        Validate matching tenant, document, and source-version relationships.
        The service must bind spec to the generation's saved pipeline recipe.
        Preserve exact text and page-map semantics; character offsets use
        decoded Python string positions, not UTF-8 byte offsets. Do not apply
        unrecorded newline conversion, normalization, or text transformations.

        Start ordinals at zero and emit them consecutively. Every chunk keeps
        its generation and document attribution. Source spans must reference
        extraction.extraction_sha256, valid character ranges, and the page map.
        If text is transformed, preserve its mapping to original source spans.
        Recipe settings define overlap and the treatment of empty inputs.

        Use bounded working memory instead of loading every chunk at once.
        Enforce limits without silently dropping content. Keep CPU work off
        the async event loop and enforce cancellation/timeouts in the runtime.
        Release owned resources on exit or early stop; do not delete caller
        input files. Partial iteration is not a completed chunking checkpoint.
        Map known failures to shared errors and never log document text.
        """
        ...

    async def aclose(self) -> None:
        """Release owned resources safely; repeated close calls must be harmless."""
        ...
