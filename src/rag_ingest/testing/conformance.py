"""Conformance tests for any Recognizer implementation.

Usage, in your own test suite:

    import unittest
    from rag_ingest.testing.conformance import RecognizerConformance

    class MyToolConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
        def make_recognizer(self):
            return MyRecognizer(api_key=...)

Run it against the real tool (it sends three small sample requests). It
checks only what the contract promises, so it works for any tool:
declared identity and capabilities, exact page coverage and numbering,
job-style submit/poll behavior, error categories, and safe shutdown.
Override sample_text to supply a known string for an accuracy check.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from rag_ingest.contracts.errors import IngestionError, PermanentInputError
from rag_ingest.contracts.recognition import (
    DocumentInput,
    JobPoll,
    RecognitionRequest,
    RecognitionResult,
    RecognizerCapabilities,
    SubmittedJob,
)
from rag_ingest.domain.limits import MAX_PAGE_CHARACTERS
from rag_ingest.domain.pipeline import ComponentSpec
from rag_ingest.testing.samples import build_pdf, sample_png

SECRET_HINTS = ("key", "token", "secret", "password", "credential", "authorization")


class RecognizerConformance:
    """Mix into unittest.IsolatedAsyncioTestCase and implement make_recognizer()."""

    poll_timeout_seconds = 300
    poll_interval_seconds = 2
    sample_text = "CONFORMANCE SAMPLE"

    def make_recognizer(self):
        raise NotImplementedError("Return the Recognizer under test")

    async def asyncSetUp(self):
        self.recognizer = self.make_recognizer()
        self._temp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        await self.recognizer.aclose()
        self._temp.cleanup()

    # -- helpers --------------------------------------------------------------

    async def store_sample(self, data, name):
        """Override for storage-reading tools: store data, return a pinned ObjectRef."""
        return None

    async def _input(self, data, name):
        caps = self.recognizer.capabilities
        if "path" in caps.inputs:
            path = Path(self._temp.name) / name
            path.write_bytes(data)
            return DocumentInput(kind="path", path=path, size_bytes=len(data))
        if "bytes" in caps.inputs:
            return DocumentInput(kind="bytes", data=data, size_bytes=len(data))
        if "storage_ref" in caps.inputs:
            ref = await self.store_sample(data, name)
            if ref is not None:
                return DocumentInput(kind="storage_ref", ref=ref, size_bytes=len(data))
        self.skipTest(
            "Tool reads from storage: override store_sample() to store the "
            "sample and return its pinned ObjectRef"
        )

    async def _request(self):
        caps = self.recognizer.capabilities
        tenant = uuid4()
        if caps.unit == "page_image":
            data, media, pages, select = (
                sample_png(self.sample_text),
                "image/png",
                (3,),
                False,
            )
            if "image/png" not in caps.image_media_types:
                from PIL import Image
                import io

                buffer = io.BytesIO()
                Image.open(io.BytesIO(data)).convert("RGB").save(buffer, format="JPEG")
                data, media = buffer.getvalue(), "image/jpeg"
            source = await self._input(data, "page.img")
        else:
            data = build_pdf(
                [
                    ("text", "first page"),
                    ("image", self.sample_text),
                    ("image", "THIRD"),
                ]
            )
            media = "application/pdf"
            if caps.page_selection:
                pages, select = (2, 3), True
            else:
                pages, select = (1, 2, 3), False
            source = await self._input(data, "doc.pdf")
        return RecognitionRequest(
            tenant_id=tenant,
            document_id=uuid4(),
            generation_id=uuid4(),
            request_id="conformance-" + uuid4().hex,
            input=source,
            media_type=media,
            page_numbers=pages,
            select_pages=select,
            timeout_seconds=120,
        )

    async def _run(self, request):
        r = self.recognizer
        if r.capabilities.call_style == "sync":
            return await r.recognize(request)
        job_id = await r.submit(request)
        self.assertIsInstance(job_id, str)
        self.assertTrue(job_id, "submit() must return a non-empty job id")
        loop = asyncio.get_running_loop()
        end = loop.time() + self.poll_timeout_seconds
        while True:
            outcome = await r.poll(
                SubmittedJob(
                    job_id=job_id,
                    request_id=request.request_id,
                    page_numbers=request.page_numbers,
                    select_pages=request.select_pages,
                )
            )
            self.assertIsInstance(outcome, JobPoll, "poll() must return a JobPoll")
            if outcome.state == "done":
                return outcome.result
            self.assertEqual(
                outcome.state, "pending", "a fresh job must not expire or fail"
            )
            self.assertLess(
                loop.time(), end, "job did not finish within poll_timeout_seconds"
            )
            await asyncio.sleep(
                max(self.poll_interval_seconds, outcome.retry_after_seconds or 0)
            )

    # -- tests ------------------------------------------------------------------

    def test_identity_and_capabilities_are_declared(self):
        r = self.recognizer
        self.assertIsInstance(r.spec, ComponentSpec)
        self.assertIsInstance(r.capabilities, RecognizerCapabilities)
        options = json.loads(r.spec.options_json)
        leaked = [k for k in options if any(h in k.lower() for h in SECRET_HINTS)]
        self.assertEqual(
            leaked, [], "spec options are fingerprinted; never put secrets there"
        )
        needed = (
            ("submit", "poll") if r.capabilities.call_style == "job" else ("recognize",)
        )
        for name in needed + ("aclose",):
            self.assertTrue(callable(getattr(r, name, None)), f"missing {name}()")

    async def test_returns_exactly_the_requested_pages(self):
        request = await self._request()
        result = await self._run(request)
        self.assertIsInstance(result, RecognitionResult)
        result.validate_for(request, max_page_characters=MAX_PAGE_CHARACTERS)
        for failure in result.failures:
            self.assertIsInstance(failure.error, IngestionError)
        self.assertTrue(result.pages, "the sample pages should be recognizable")
        if self.recognizer.capabilities.unit == "page_image":
            self.assertEqual([p.page_number for p in result.pages], [3])

    async def test_unknown_job_is_reported_not_crashed(self):
        if self.recognizer.capabilities.call_style != "job":
            self.skipTest("sync tool")
        try:
            outcome = await self.recognizer.poll(
                SubmittedJob(
                    job_id="conformance-unknown-" + uuid4().hex,
                    request_id="conformance-" + uuid4().hex,
                    page_numbers=(1,),
                )
            )
        except PermanentInputError:
            return
        self.assertIsInstance(outcome, JobPoll)
        self.assertIn(outcome.state, ("expired", "failed"))

    async def test_close_is_idempotent(self):
        await self.recognizer.aclose()
        await self.recognizer.aclose()
