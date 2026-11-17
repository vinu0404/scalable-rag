"""A configurable, deterministic Recognizer for tests; it never calls a network.

It can play any capability shape (page_image/document, sync/job, any input
forms and limits) and records every request so tests can assert exactly what
the package sent. It checks that each request uses a declared input form and
that its content is the declared media type.
"""

import io
import json
from collections import Counter

from rag_ingest.contracts.recognition import (
    JobPoll,
    PageFailure,
    RecognitionResult,
    RecognizedPage,
    RecognizerCapabilities,
)
from rag_ingest.domain.pipeline import ComponentSpec


class FakeRecognizer:
    """Return text_for(page_number) for each requested page.

    failures: {page_number: error instance} reported as PageFailure.
    call_errors: errors raised by the next calls, in order (whole-call failure).
    job style: poll() returns pending pending_polls times per job, then done;
    expire_polls makes the first poll of that many jobs report expired.
    """

    def __init__(
        self,
        *,
        unit="page_image",
        call_style="sync",
        inputs=frozenset({"bytes"}),
        page_selection=False,
        max_pages_per_request=None,
        max_bytes_per_request=None,
        max_image_pixels=None,
        output_format="text",
        max_requests_per_second=None,
        model="fake-1",
        text_for=None,
        confidence=None,
        failures=None,
        call_errors=(),
        pending_polls=0,
        expire_polls=0,
        storage=None,
    ):
        self._spec = ComponentSpec(
            driver="fake",
            implementation_version="1",
            options_json=json.dumps({"model": model}),
        )
        self._caps = RecognizerCapabilities(
            unit=unit,
            call_style=call_style,
            inputs=frozenset(inputs),
            page_selection=page_selection,
            max_pages_per_request=max_pages_per_request,
            max_bytes_per_request=max_bytes_per_request,
            max_image_pixels=max_image_pixels,
            output_format=output_format,
            max_requests_per_second=max_requests_per_second,
        )
        self.text_for = text_for or (lambda page: f"recognized text of page {page}")
        self.confidence = confidence
        self.failures = dict(failures or {})
        self.call_errors = list(call_errors)
        self.pending_polls, self.expire_polls = pending_polls, expire_polls
        self.storage = storage  # optional ObjectStore to read storage_ref inputs
        self.requests, self.jobs, self.polls = [], {}, Counter()
        self.part_page_counts = {}
        self.closed = 0

    @property
    def spec(self):
        return self._spec

    @property
    def capabilities(self):
        return self._caps

    def _content(self, request):
        source = request.input
        if source.kind == "path":
            return source.path.read_bytes()
        if source.kind == "bytes":
            return source.data
        return None  # storage_ref / signed_url: not fetched unless storage given

    def _inspect(self, request):
        if request.input.kind not in self._caps.inputs:
            raise AssertionError("Request used an input form the tool did not declare")
        data = self._content(request)
        if data is None:
            return
        if self._caps.unit == "page_image":
            magic = {b"\x89PNG": "image/png", b"\xff\xd8\xff": "image/jpeg"}
            kind = next((v for k, v in magic.items() if data.startswith(k)), None)
            if kind != request.media_type:
                raise AssertionError("Page image content does not match its media type")
        else:
            if not data.startswith(b"%PDF-"):
                raise AssertionError("Document request content is not a PDF")
            from pypdf import PdfReader

            self.part_page_counts[request.request_id] = len(
                PdfReader(io.BytesIO(data)).pages
            )

    def _result(self, request):
        pages, failed = [], []
        for number in request.page_numbers:
            if number in self.failures:
                failed.append(
                    PageFailure(page_number=number, error=self.failures[number])
                )
            else:
                pages.append(
                    RecognizedPage(
                        page_number=number,
                        text=self.text_for(number),
                        confidence=self.confidence,
                    )
                )
        return RecognitionResult(pages=tuple(pages), failures=tuple(failed))

    async def recognize(self, request):
        self.requests.append(request)
        if self.call_errors:
            raise self.call_errors.pop(0)
        self._inspect(request)
        return self._result(request)

    async def submit(self, request):
        self.requests.append(request)
        if self.call_errors:
            raise self.call_errors.pop(0)
        self._inspect(request)
        self._submitted = getattr(self, "_submitted", 0) + 1
        job_id = f"fake-job-{self._submitted}"  # never reused, like real tools
        self.jobs[job_id] = request
        return job_id

    async def poll(self, job):
        job_id = job.job_id
        if job_id not in self.jobs:
            return JobPoll(state="expired")
        self.polls[job_id] += 1
        if self.expire_polls and self.polls[job_id] == 1:
            self.expire_polls -= 1
            del self.jobs[job_id]
            return JobPoll(state="expired")
        if self.polls[job_id] <= self.pending_polls:
            return JobPoll(state="pending", retry_after_seconds=1)
        return JobPoll(state="done", result=self._result(self.jobs[job_id]))

    async def aclose(self):
        self.closed += 1
