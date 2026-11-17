"""Template: plug your own recognition tool (OCR model, document API, LLM) in.

Copy this file, fill in the three TODOs, and run the conformance kit:

    class MyConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
        def make_recognizer(self):
            return MyRecognizer(...)

Then inject it:  build_engine(settings, recognizer=MyRecognizer(...))
and set INGEST_EXTRACTION_MODE (recognize_unreadable, recognize_with_images
or recognize_all). Run every worker AND the scheduler with the same
recognizer: job-style tools are polled by the scheduler.

The package does page selection, rendering, splitting, caching, budgets,
rate limiting and waiting. You only call your tool and map its errors.
"""

import asyncio
import json

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RateLimitedError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.contracts.recognition import (
    JobPoll,
    RecognitionResult,
    RecognizedPage,
    RecognizerCapabilities,
)
from rag_ingest.domain.pipeline import ComponentSpec


class MyRecognizer:
    """Describe your tool honestly; the package plans every request from this."""

    def __init__(self, *, client, model_version):
        self.client = client  # your SDK/HTTP client, created and owned here
        self._spec = ComponentSpec(
            driver="my_tool",  # stable name of this adapter
            implementation_version="1",  # bump when your adapter's output changes
            # Every output-affecting option, with a PINNED model version.
            # Never put credentials here: this is fingerprinted and stored.
            options_json=json.dumps({"model": model_version, "language": "en"}),
        )
        self._caps = RecognizerCapabilities(
            unit="document",  # or "page_image" if your tool reads one image
            call_style="sync",  # or "job" if it returns a job id to poll
            inputs=frozenset({"bytes"}),  # path | bytes | storage_ref | signed_url
            output_format="markdown",  # or "text"
            page_selection=False,  # True if the tool can take a page list itself
            max_pages_per_request=50,  # your tool's documented limits
            max_bytes_per_request=20 * 1024**2,
            max_requests_per_second=5,  # account-wide; shared across processes
        )

    @property
    def spec(self):
        return self._spec

    @property
    def capabilities(self):
        return self._caps

    async def recognize(self, request):
        """Sync tools: return one RecognizedPage (or PageFailure) per page."""
        try:
            async with asyncio.timeout(request.timeout_seconds):
                # TODO 1: call your tool with request.input.data (or .path,
                # .ref, .url) and, for page selection, request.page_numbers.
                response = await self.client.recognize(request.input.data)
        except TimeoutError:
            raise UncertainOutcomeError(
                "Tool response lost; it may be billed"
            ) from None
        except ConnectionError:
            raise RetryableError("Could not reach the tool") from None
        # TODO 2: map your tool's errors, e.g.
        #   throttled      -> RateLimitedError(..., retry_after_seconds=...)
        #   bad document   -> PermanentInputError(...)
        #   bad credentials-> OperatorActionRequiredError(...)
        _ = (RateLimitedError, PermanentInputError, OperatorActionRequiredError)
        # TODO 3: convert output. Position i of a page subset or single image
        # is request.page_numbers[i]. Strip tool-specific markup you do not
        # want indexed (for example image placeholders).
        pages = tuple(
            RecognizedPage(page_number=number, text=text)
            for number, text in zip(
                request.page_numbers, response.page_texts, strict=True
            )
        )
        return RecognitionResult(pages=pages)

    async def submit(self, request):
        """Job tools: start the job and return its id only once it is accepted."""
        raise NotImplementedError("Only for call_style='job'")

    async def poll(self, job):
        """Job tools: report pending / done(result) / failed(error) / expired.

        job.job_id is what submit() returned; job.page_numbers maps position i
        of the submitted document to its original page number.
        """
        return JobPoll(state="expired")

    async def aclose(self):
        """Release owned clients; safe to call twice."""
        close = getattr(self.client, "aclose", None)
        if close:
            await close()
