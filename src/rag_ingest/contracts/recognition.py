"""Contract for plugging in any text-recognition tool: an OCR model, a cloud
document API, a multimodal LLM, or a local binary.

The package never depends on a specific tool. An adapter declares what its
tool can do in RecognizerCapabilities; the package adapts to that. It decides
which pages need recognition, renders page images or builds page subsets,
splits work to fit the declared limits, maps results back to original page
numbers, caches finished pages, waits on job-style tools without holding a
worker, and enforces budgets and rate limits.

An adapter only calls its tool, converts the tool's output into
RecognizedPage text, and maps failures to contracts.errors categories:

- RateLimitedError: the tool throttled the call (retried with backoff).
- RetryableError: a temporary failure before or without any billed work.
- UncertainOutcomeError: a paid call whose outcome is unknown (bounded repeats).
- PermanentInputError: this input can never succeed unchanged.
- OperatorActionRequiredError: credentials, quota or configuration problems.

Never log or raise messages containing document text or credentials.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from rag_ingest.contracts.errors import IngestionError
from rag_ingest.contracts.storage import ObjectRef
from rag_ingest.domain.pipeline import ComponentSpec

INPUT_KINDS = frozenset({"path", "bytes", "storage_ref", "signed_url"})
IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})


@dataclass(frozen=True, kw_only=True)
class RecognizerCapabilities:
    """Declare what the tool accepts and returns; the package plans around it.

    unit: "page_image" tools receive one rendered page image per request;
    "document" tools receive a PDF (the original, or a subset of its pages).
    call_style: "sync" returns results from recognize(); "job" returns a job
    id from submit() and results later from poll().
    inputs: the forms a request may use, most private first is chosen:
    "path" (local file, in-process tools only), "bytes", "storage_ref" (the
    exact stored object version, for tools that read from the same storage),
    "signed_url" (a short-lived read link; used only if the recipe allows it).
    page_selection: a document tool can process a listed subset of pages of
    the full document itself, so the package need not build a page subset.
    output_format is fingerprinted: changing it produces a new build.
    """

    unit: Literal["page_image", "document"]
    call_style: Literal["sync", "job"]
    inputs: frozenset
    output_format: Literal["text", "markdown"] = "text"
    page_selection: bool = False
    max_pages_per_request: int | None = None
    max_bytes_per_request: int | None = None
    max_image_pixels: int | None = None
    image_media_types: frozenset = IMAGE_TYPES
    max_requests_per_second: float | None = None

    def __post_init__(self) -> None:
        if self.unit not in ("page_image", "document"):
            raise ValueError("unit must be 'page_image' or 'document'")
        if self.call_style not in ("sync", "job"):
            raise ValueError("call_style must be 'sync' or 'job'")
        if self.output_format not in ("text", "markdown"):
            raise ValueError("output_format must be 'text' or 'markdown'")
        if not isinstance(self.inputs, frozenset) or not self.inputs:
            raise ValueError("inputs must be a non-empty frozenset")
        if not self.inputs <= INPUT_KINDS:
            raise ValueError(
                "inputs may only contain " + ", ".join(sorted(INPUT_KINDS))
            )
        for name in (
            "max_pages_per_request",
            "max_bytes_per_request",
            "max_image_pixels",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        rate = self.max_requests_per_second
        if rate is not None and (
            type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
        ):
            raise ValueError("max_requests_per_second must be positive or None")
        if self.unit == "page_image":
            if self.page_selection:
                raise ValueError("page_image tools cannot declare page_selection")
            if not self.inputs & {"path", "bytes"}:
                raise ValueError("page_image tools must accept 'path' or 'bytes'")
            if self.max_pages_per_request not in (None, 1):
                raise ValueError("page_image tools receive exactly one page")
            if (
                not isinstance(self.image_media_types, frozenset)
                or not self.image_media_types
                or not self.image_media_types <= IMAGE_TYPES
            ):
                raise ValueError("image_media_types must be a subset of PNG/JPEG")


@dataclass(frozen=True, kw_only=True)
class DocumentInput:
    """Exactly one way to reach the content of a request."""

    kind: Literal["path", "bytes", "storage_ref", "signed_url"]
    path: Path | None = None
    data: bytes | None = field(default=None, repr=False)
    ref: ObjectRef | None = None
    url: str | None = field(default=None, repr=False)
    url_expires_at: datetime | None = None
    size_bytes: int = 0

    def __post_init__(self) -> None:
        present = {
            "path": self.path is not None,
            "bytes": self.data is not None,
            "storage_ref": self.ref is not None,
            "signed_url": self.url is not None,
        }
        if self.kind not in INPUT_KINDS or [k for k, v in present.items() if v] != [
            self.kind
        ]:
            raise ValueError("DocumentInput must carry exactly its declared kind")
        if self.kind == "storage_ref" and not self.ref.version_id:
            raise ValueError("storage_ref inputs must pin an object version")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")


@dataclass(frozen=True, kw_only=True)
class RecognitionRequest:
    """One unit of recognition work.

    page_numbers are the ORIGINAL 1-based page numbers this request covers, in
    order. For a page subset or a single page image, position i of the input
    corresponds to page_numbers[i]. For page selection, the input is the full
    document and the tool must process exactly these pages.

    request_id is deterministic for the same work; adapters may pass it to
    their tool as an idempotency key. tenant_id/document_id let adapters choose
    per-tenant credentials or regions; they never carry content.
    """

    tenant_id: UUID
    document_id: UUID
    generation_id: UUID
    request_id: str
    input: DocumentInput
    media_type: str
    page_numbers: tuple[int, ...]
    select_pages: bool = False
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.page_numbers or any(
            type(p) is not int or p < 1 for p in self.page_numbers
        ):
            raise ValueError("page_numbers must be positive page numbers")
        if len(set(self.page_numbers)) != len(self.page_numbers):
            raise ValueError("page_numbers must not repeat")
        if not isinstance(self.request_id, str) or not 8 <= len(self.request_id) <= 128:
            raise ValueError("request_id must be a bounded string")


@dataclass(frozen=True, kw_only=True)
class RecognizedPage:
    """Recognized text for one original page; confidence is 0..1 if known."""

    page_number: int
    text: str = field(repr=False)
    confidence: float | None = None

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        c = self.confidence
        if c is not None and (
            type(c) not in (int, float) or not math.isfinite(c) or not 0 <= c <= 1
        ):
            raise ValueError("confidence must be between 0 and 1, or None")


@dataclass(frozen=True, kw_only=True)
class PageFailure:
    """A page the tool could not recognize, with a contract error category."""

    page_number: int
    error: IngestionError

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        if not isinstance(self.error, IngestionError):
            raise ValueError("error must be a contracts.errors category")


@dataclass(frozen=True, kw_only=True)
class RecognitionResult:
    """Every requested page appears exactly once, as a page or a failure."""

    pages: tuple[RecognizedPage, ...]
    failures: tuple[PageFailure, ...] = ()

    def validate_for(self, request: RecognitionRequest, *, max_page_characters: int):
        """Reject missing, extra, duplicate or oversized page results."""
        if not isinstance(self.pages, tuple) or not isinstance(self.failures, tuple):
            raise ValueError("pages and failures must be tuples")
        seen = [p.page_number for p in self.pages] + [
            f.page_number for f in self.failures
        ]
        if sorted(seen) != sorted(request.page_numbers):
            raise ValueError(
                "Recognition result does not cover exactly the requested pages"
            )
        for page in self.pages:
            if len(page.text) > max_page_characters:
                raise ValueError("Recognized page exceeds the page text limit")


@dataclass(frozen=True, kw_only=True)
class SubmittedJob:
    """What poll() receives: the job id plus the request it was submitted for.

    Results are often positional (page 1..n of the submitted document), so a
    stateless adapter needs page_numbers to report ORIGINAL page numbers:
    position i of a whole document or page subset is page_numbers[i]; with
    select_pages the tool processed exactly these original pages.
    """

    job_id: str
    request_id: str
    page_numbers: tuple[int, ...]
    select_pages: bool = False


@dataclass(frozen=True, kw_only=True)
class JobPoll:
    """State of a job-style request.

    pending: not finished; retry_after_seconds suggests when to poll again.
    done: result holds the pages (and any per-page failures).
    failed: the whole job failed; error holds a contracts.errors category.
    expired: the tool no longer has this job; the package may resubmit it.
    """

    state: Literal["pending", "done", "failed", "expired"]
    result: RecognitionResult | None = None
    error: IngestionError | None = None
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.state not in ("pending", "done", "failed", "expired"):
            raise ValueError("Invalid job state")
        if (self.state == "done") != (self.result is not None):
            raise ValueError("Only done polls carry a result")
        if self.state == "failed" and not isinstance(self.error, IngestionError):
            raise ValueError("Failed polls carry a contracts.errors category")


class Recognizer(Protocol):
    """Implement this to use your own recognition tool.

    spec identifies the tool's behavior and is fingerprinted with every
    build: include the model version and every output-affecting option, never
    credentials. Pin model versions; a silently changing model cannot be
    detected by the package.

    Sync tools implement recognize(). Job tools implement submit() and poll();
    submit() must return only after the tool has accepted the job. Calls may
    run concurrently. Keep blocking work off the event loop and honor
    request.timeout_seconds. aclose() releases owned clients and is safe to
    call more than once.
    """

    @property
    def spec(self) -> ComponentSpec: ...

    @property
    def capabilities(self) -> RecognizerCapabilities: ...

    async def recognize(self, request: RecognitionRequest) -> RecognitionResult: ...

    async def submit(self, request: RecognitionRequest) -> str: ...

    async def poll(self, job: SubmittedJob) -> JobPoll: ...

    async def aclose(self) -> None: ...
