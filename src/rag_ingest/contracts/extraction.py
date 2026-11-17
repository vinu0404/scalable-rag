"""Define page-by-page text extraction without choosing a parser library.

The worker downloads and verifies the intended source into bounded temporary
storage first. The extractor reads that local file; it does not own database
leases, upload acceptance, checkpoint registration, or publication.
"""

import math
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rag_ingest.domain.pipeline import ComponentSpec


@dataclass(frozen=True, kw_only=True)
class ExtractedPage:
    """Return one page's text while preserving its original page number.

    Page numbers start at one. Empty text is allowed so a blank page does not
    shift later citations. The service saves page boundaries and the exact
    combined extraction used by SourceSpan character offsets. This class does
    not claim that OCR was performed or that an empty page was actually blank.
    """

    page_number: int
    text: str = field(repr=False)
    # How this page's text was obtained, recorded in the page map:
    # text (native), blank (no text, no visual content), unrecognized (visual
    # content but no usable text and no recognition), recognized (from the
    # recognizer). needs_recognition asks the service to recognize the page.
    method: str = "text"
    needs_recognition: bool = False

    METHODS = ("text", "blank", "unrecognized", "recognized")

    def __post_init__(self) -> None:
        """Require a positive page number and string text, including empty text."""
        if type(self.page_number) is not int or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.method not in self.METHODS:
            raise ValueError("Unknown page extraction method")
        if not isinstance(self.needs_recognition, bool):
            raise ValueError("needs_recognition must be a boolean")


@dataclass(frozen=True, kw_only=True)
class ExtractionLimits:
    """Bound the amount of parsing work and text one extraction may produce.

    These runtime limits decide whether work is allowed, not how text should
    be transformed. Truncating text to fit a limit is not successful extraction.
    OCR choices and other output-changing settings belong in ComponentSpec.
    Memory, CPU, and temporary-disk limits must also be enforced by the runtime.
    """

    max_pages: int
    max_page_characters: int
    max_total_characters: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Reject nonpositive limits and invalid extraction time budgets."""
        for name in ("max_pages", "max_page_characters", "max_total_characters"):
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


class Extractor(Protocol):
    """Specify an extractor that yields bounded pages through an async context.

    Validate the selected driver, implementation version, and options before
    processing. Do not silently substitute another parser or enable OCR with
    different behavior under the same configuration fingerprint.
    """

    def open_pages(
        self,
        source_path: Path,
        *,
        spec: ComponentSpec,
        limits: ExtractionLimits,
    ) -> AbstractAsyncContextManager[AsyncIterator[ExtractedPage]]:
        """Read an already verified local source and yield pages in source order.

        Use with async with and async for. Emit ascending, consecutive page
        numbers starting at one, including pages with no extracted text. For
        plain text, treat the source as one logical page unless the versioned
        recipe explicitly defines another mapping.

        Enforce page, text, and elapsed-time bounds. Report limits or unsupported
        input explicitly; never silently drop pages, truncate output, or treat
        parser failure as successful end-of-file. OCR/empty-text decisions must
        follow the saved recipe and distinguish unsupported work from success.

        Keep parsing off the async event loop. A noncooperative parser needs
        an isolated process/runtime that can actually be stopped; cancelling
        an await alone does not stop CPU work. Release owned resources on exit,
        failure, cancellation, or early iteration stop. Do not delete the
        caller's source file. No checkpoint is complete until the whole chosen
        extraction unit and its saved manifest are verified by the service.

        Map known input/provider failures to contracts.errors categories while
        keeping unexpected programming failures visible. Do not log page text.
        """
        ...

    async def aclose(self) -> None:
        """Release owned extractor resources; repeated close calls must be safe."""
        ...
