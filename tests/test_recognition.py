"""Recognition: routing, page images, documents, jobs, cache, budgets, identity.

Runs the real engine, worker, scheduler and PostgreSQL metadata (opt-in DSN),
with in-process S3/SQS and the package's FakeRecognizer. No network tool is
used: these tests prove the generic contract, not any vendor's accuracy.
"""

from rag_ingest.runtime.search import search as run_search
import io
import json
import logging
import unittest
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import sqlalchemy as sa
from pypdf import PdfReader, PdfWriter

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RetryableError,
)
from rag_ingest.contracts.recognition import RecognitionResult, RecognizedPage
from rag_ingest.domain import extraction_recipe as recipe
from rag_ingest.domain.enums import JobStatus as JS, ProcessingStage as ST
from rag_ingest.domain.pipeline import ComponentSpec
from rag_ingest.runtime import recognition
from rag_ingest.runtime.factory import build_engine
from rag_ingest.runtime.runners import publish_once, scheduler_once, worker_once
from rag_ingest.testing.conformance import RecognizerConformance
from rag_ingest.testing.fakes import FakeRecognizer
from rag_ingest.testing.samples import build_pdf

from test_postgres_integration import DSN, PostgresIntegration

MIXED = [  # page 1 text, 2 scanned, 3 text+image, 4 blank
    ("text", "Native text page about leases, outboxes and recovery."),
    ("image", "SCANNED"),
    ("mixed", ("Figure caption with enough native text", "CHART")),
    ("blank", None),
]
OPTS_DOC = {
    "recognizer": {
        "driver": "f",
        "implementation_version": "1",
        "options": {},
        "unit": "page_image",
        "output_format": "text",
    }
}


class ClassificationRules(unittest.TestCase):
    def options(self, mode):
        return recipe.validate(
            {"mode": mode, **(OPTS_DOC if mode != "text_only" else {})}
        )

    def test_page_kinds_per_mode(self):
        scanned, text, mixed, blank = (
            ("", 1, 40),
            ("Plenty of native text here", 0, 90),
            ("Plenty of native caption text", 1, 90),
            ("", 0, 0),
        )
        expect = {
            "text_only": [
                ("unrecognized", False),
                ("text", False),
                ("text", False),
                ("blank", False),
            ],
            "recognize_unreadable": [
                ("text", True),
                ("text", False),
                ("text", False),
                ("blank", False),
            ],
            "recognize_with_images": [
                ("text", True),
                ("text", False),
                ("text", True),
                ("blank", False),
            ],
            "recognize_all": [("text", True)] * 4,
        }
        for mode, wanted in expect.items():
            got = [
                recipe.classify(*p, self.options(mode), is_pdf=True)
                for p in (scanned, text, mixed, blank)
            ]
            self.assertEqual(got, wanted, mode)

    def test_garbage_text_layer_counts_as_unreadable(self):
        garbage = "����������������ab"
        self.assertGreater(recipe.garbage_ratio(garbage), 0.3)
        self.assertEqual(
            recipe.classify(
                garbage, 1, 50, self.options("recognize_unreadable"), is_pdf=True
            ),
            ("text", True),
        )

    def test_vector_drawn_page_is_visual(self):
        self.assertEqual(
            recipe.classify(
                "", 0, 50_000, self.options("recognize_unreadable"), is_pdf=True
            ),
            ("text", True),
        )

    def test_plain_text_files_never_need_recognition(self):
        self.assertEqual(
            recipe.classify("hello", 0, 0, self.options("recognize_all"), is_pdf=False),
            ("text", False),
        )

    def test_recipe_rejects_inconsistent_options(self):
        with self.assertRaises(ValueError):
            recipe.validate({"mode": "recognize_all"})  # no recognizer
        with self.assertRaises(ValueError):
            recipe.validate({"mode": "text_only", **OPTS_DOC})
        with self.assertRaises(ValueError):
            recipe.validate({"unknown": 1})


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class RecognitionPipeline(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = PostgresIntegration.asyncSetUp
    asyncTearDown = PostgresIntegration.asyncTearDown

    # ---- helpers ------------------------------------------------------------

    def use(self, recognizer=None, **settings):
        e = self.engine
        self.engine = build_engine(
            replace(e.settings, **settings),
            metadata=e.metadata,
            objects=e.objects,
            queue=e.queue,
            vectors=e.vectors,
            embedder=e.embedder,
            **({"recognizer": recognizer} if recognizer else {}),
        )
        return recognizer

    async def ingest(self, pages=MIXED, allow_duplicate=False):
        data = build_pdf(pages)
        r = await self.engine.initiate_upload(
            self.tenant,
            filename="doc.pdf",
            size_bytes=len(data),
            request_key=str(uuid4()),
            allow_duplicate=allow_duplicate,
        )
        u = r["upload"]
        self.engine.objects.client.put_object(
            Bucket=u.bucket, Key=u.object_key, Body=data
        )
        await self.engine.complete_upload(self.tenant, u.upload_id)
        await self.engine.verify_upload(self.tenant, u.upload_id)
        await self.engine.metadata.admit_generations(
            tenant_id=self.tenant, max_active=16
        )
        self.data = data
        return u

    async def drive(self, rounds=6, poll=False):
        for _ in range(rounds):
            await scheduler_once(self.engine)
            if poll:
                await self.sql("UPDATE recognition_jobs SET next_poll_at = now()")
                await recognition.poll_once(self.engine)
            await publish_once(self.engine, "p")
            while await worker_once(self.engine, "w", wait_seconds=0):
                pass
            await self.sql("UPDATE jobs SET due_at = now() WHERE status = 'retry_wait'")

    async def sql(self, statement, **args):
        async with self.engine.metadata.db.engine.begin() as c:
            return await c.execute(sa.text(statement), args)

    async def extract_job(self, u):
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant,
            generation_id=doc.desired_generation_id,
            stage=ST.EXTRACT,
        )
        report = await self.engine.metadata.job_report(self.tenant, jobs[0].job_id)
        return jobs[0], report["failure_code"]

    async def extracted(self, u):
        job, _ = await self.extract_job(u)
        self.assertEqual(job.status, JS.SUCCEEDED)
        manifest = await self.engine.checkpoint_manifest(job)
        pages = json.loads(
            await self.engine.read_output(self.engine._output(manifest, "page_map"))
        )
        text = (
            await self.engine.read_output(
                self.engine._output(manifest, "extracted_text")
            )
        ).decode()
        return [p["method"] for p in pages["pages"]], text

    def page_image(self, **kw):
        return FakeRecognizer(unit="page_image", **kw)

    def document(self, **kw):
        return FakeRecognizer(unit="document", **kw)

    # ---- phase 1: routing, no silent drops -----------------------------------

    async def test_text_only_records_unrecognized_pages_and_warns(self):
        u = await self.ingest()
        with self.assertLogs("rag_ingest", level=logging.WARNING) as logs:
            await self.drive()
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["text", "unrecognized", "text", "blank"])
        self.assertIn("pages_unrecognized", "".join(logs.output))

    async def test_text_only_fail_policy_rejects_unreadable_pages(self):
        self.use(unreadable_pages="fail")
        u = await self.ingest()
        await self.drive()
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "recognition_required"))

    async def test_scanned_document_without_recognizer_fails_clearly(self):
        u = await self.ingest([("image", "ONLY A SCAN")])
        await self.drive()
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "no_extractable_text"))

    def test_recognition_mode_without_recognizer_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "needs an injected Recognizer"):
            self.use(extraction_mode="recognize_all")
        broken = SimpleNamespace(
            spec=self.page_image().spec,
            capabilities=self.page_image(call_style="job").capabilities,
        )
        with self.assertRaisesRegex(ValueError, "submit, poll"):
            self.use(broken, extraction_mode="recognize_all")

    # ---- phase 2: page-image tools ----------------------------------------------

    async def test_page_images_only_for_pages_that_need_them(self):
        fake = self.use(
            self.page_image(max_image_pixels=2_000_000, max_bytes_per_request=600_000),
            extraction_mode="recognize_unreadable",
        )
        u = await self.ingest()
        await self.drive()
        methods, text = await self.extracted(u)
        self.assertEqual(methods, ["text", "recognized", "text", "blank"])
        self.assertIn("recognized text of page 2", text)
        self.assertEqual([r.page_numbers for r in fake.requests], [(2,)])
        image = fake.requests[0].input.data
        from PIL import Image

        rendered = Image.open(io.BytesIO(image))
        self.assertLessEqual(rendered.width * rendered.height, 2_000_000)
        self.assertLessEqual(len(image), 600_000)

    async def test_recognize_with_images_includes_text_pages_with_images(self):
        fake = self.use(self.page_image(), extraction_mode="recognize_with_images")
        u = await self.ingest()
        await self.drive()
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["text", "recognized", "recognized", "blank"])
        self.assertEqual(sorted(r.page_numbers for r in fake.requests), [(2,), (3,)])

    async def test_recognize_all_covers_every_page(self):
        fake = self.use(self.page_image(), extraction_mode="recognize_all")
        u = await self.ingest()
        await self.drive()
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["recognized"] * 4)
        self.assertEqual(len(fake.requests), 4)

    async def test_retry_after_partial_failure_recognizes_only_missing_pages(self):
        pages = [
            ("image", "A"),
            ("text", "A text page with plenty of characters"),
            ("image", "B"),
        ]
        fake = self.use(
            self.page_image(failures={3: RetryableError("busy")}),
            extraction_mode="recognize_unreadable",
        )
        u = await self.ingest(pages)
        await self.drive(rounds=1)
        job, code = await self.extract_job(u)
        self.assertEqual(
            (job.status, code), (JS.RETRY_WAIT, "temporary_dependency_failure")
        )
        fake.failures.clear()
        await self.drive()
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["recognized", "text", "recognized"])
        from collections import Counter

        calls = Counter(p for r in fake.requests for p in r.page_numbers)
        self.assertEqual(calls, {1: 1, 3: 2})  # page 1 cached, never re-billed

    async def test_permanent_page_failure_fails_the_build(self):
        self.use(
            self.page_image(failures={2: PermanentInputError("unreadable")}),
            extraction_mode="recognize_unreadable",
        )
        u = await self.ingest()
        await self.drive()
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "recognition_failed"))

    async def test_result_with_wrong_pages_is_a_contract_violation(self):
        class Wrong(FakeRecognizer):
            def _result(self, request):
                return RecognitionResult(
                    pages=(RecognizedPage(page_number=99, text="x"),)
                )

        self.use(Wrong(unit="page_image"), extraction_mode="recognize_unreadable")
        u = await self.ingest()
        await self.drive()
        job, code = await self.extract_job(u)
        self.assertEqual(
            (job.status, code), (JS.FAILED, "recognizer_contract_violation")
        )

    async def test_low_confidence_fails_when_a_minimum_is_set(self):
        self.use(
            self.page_image(confidence=0.2),
            extraction_mode="recognize_unreadable",
            min_confidence=0.5,
        )
        u = await self.ingest()
        await self.drive()
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "recognition_low_confidence"))

    async def test_worker_with_a_different_recognizer_refuses_the_build(self):
        self.use(
            self.page_image(model="model-a"), extraction_mode="recognize_unreadable"
        )
        u = await self.ingest()
        self.use(
            self.page_image(model="model-b"), extraction_mode="recognize_unreadable"
        )
        with self.assertRaises(OperatorActionRequiredError):
            await self.drive()
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "operator_action_required"))

    def test_recognizer_identity_is_fingerprinted(self):
        base = self.engine.settings
        a = replace(
            base,
            extraction_mode="recognize_all",
            recognizer_spec=self.page_image(model="a").spec,
            recognizer_capabilities=self.page_image().capabilities,
        )
        b = replace(a, recognizer_spec=self.page_image(model="b").spec)
        prints = {
            base.pipeline().fingerprint,
            a.pipeline().fingerprint,
            b.pipeline().fingerprint,
        }
        self.assertEqual(len(prints), 3)

    # ---- phase 3: document tools ------------------------------------------------

    async def test_whole_document_in_one_request(self):
        fake = self.use(self.document(), extraction_mode="recognize_all")
        u = await self.ingest()
        await self.drive()
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["recognized"] * 4)
        (request,) = fake.requests
        self.assertEqual(
            (request.page_numbers, request.select_pages), ((1, 2, 3, 4), False)
        )
        self.assertEqual(request.input.data, self.data)

    async def test_page_selection_sends_whole_file_with_page_list(self):
        pages = MIXED[:3] + [("image", "SECOND SCAN")]
        fake = self.use(
            self.document(page_selection=True), extraction_mode="recognize_unreadable"
        )
        u = await self.ingest(pages)
        await self.drive()
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["text", "recognized", "text", "recognized"])
        (request,) = fake.requests
        self.assertEqual((request.page_numbers, request.select_pages), ((2, 4), True))
        self.assertEqual(fake.part_page_counts[request.request_id], 4)

    async def test_page_subset_pdf_when_tool_cannot_select_pages(self):
        pages = MIXED[:3] + [("image", "SECOND SCAN")]
        fake = self.use(self.document(), extraction_mode="recognize_unreadable")
        u = await self.ingest(pages)
        await self.drive()
        await self.extracted(u)
        (request,) = fake.requests
        self.assertEqual((request.page_numbers, request.select_pages), ((2, 4), False))
        self.assertEqual(fake.part_page_counts[request.request_id], 2)

    async def test_requests_split_to_the_page_limit(self):
        pages = [("image", f"P{i}") for i in range(5)]
        fake = self.use(
            self.document(max_pages_per_request=2), extraction_mode="recognize_all"
        )
        u = await self.ingest(pages)
        await self.drive()
        await self.extracted(u)
        self.assertEqual(  # requests run concurrently: compare as a set
            sorted(r.page_numbers for r in fake.requests), [(1, 2), (3, 4), (5,)]
        )

    async def test_parts_halve_until_they_fit_the_size_limit(self):
        pages = [("image", f"P{i}") for i in range(4)]
        single = PdfWriter()
        single.add_page(PdfReader(io.BytesIO(build_pdf(pages))).pages[0])
        buffer = io.BytesIO()
        single.write(buffer)
        limit = buffer.tell() + 2000  # one page fits, two do not
        fake = self.use(
            self.document(max_bytes_per_request=limit), extraction_mode="recognize_all"
        )
        u = await self.ingest(pages)
        await self.drive()
        await self.extracted(u)
        self.assertEqual(
            sorted(r.page_numbers for r in fake.requests), [(1,), (2,), (3,), (4,)]
        )
        self.assertTrue(all(r.input.size_bytes <= limit for r in fake.requests))

    async def test_storage_ref_inputs_pin_exact_versions(self):
        fake = self.use(
            self.document(inputs={"storage_ref"}), extraction_mode="recognize_all"
        )
        u = await self.ingest()
        await self.drive()
        await self.extracted(u)
        (request,) = fake.requests
        ref = request.input.ref
        source = await self.engine.metadata.get_verification_ref(
            tenant_id=self.tenant, upload_id=u.upload_id
        )
        self.assertEqual(
            (ref.bucket, ref.key, ref.version_id),
            (source.bucket, source.key, source.version_id),
        )
        subset = self.use(
            self.document(inputs={"storage_ref"}),
            extraction_mode="recognize_unreadable",
        )
        u2 = await self.ingest(allow_duplicate=True)  # same PDF, another mode
        await self.drive()
        await self.extracted(u2)
        part = subset.requests[0].input.ref
        self.assertIn("/recognition/parts/", part.key)
        self.assertTrue(part.version_id)

    async def test_signed_url_inputs_need_explicit_opt_in(self):
        self.use(self.document(inputs={"signed_url"}), extraction_mode="recognize_all")
        u = await self.ingest()
        with self.assertRaises(OperatorActionRequiredError):
            await self.drive()
        _, code = await self.extract_job(u)
        self.assertEqual(code, "operator_action_required")
        fake = self.use(
            self.document(inputs={"signed_url"}),
            extraction_mode="recognize_all",
            allow_signed_url=True,
        )
        u2 = await self.ingest()
        await self.drive()
        await self.extracted(u2)
        self.assertTrue(fake.requests[0].input.url.startswith("https://"))

    # ---- phase 3: job-style tools -------------------------------------------

    async def test_job_tool_waits_without_a_worker_and_completes(self):
        fake = self.use(
            self.document(call_style="job", pending_polls=1),
            extraction_mode="recognize_all",
        )
        u = await self.ingest()
        await self.drive(rounds=1)
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.WAITING, "awaiting_recognition"))
        attempts = job.attempt
        await self.drive(rounds=1, poll=True)  # first poll: still pending
        job, _ = await self.extract_job(u)
        self.assertEqual(job.status, JS.WAITING)
        self.assertEqual(job.attempt, attempts, "waiting must not spend attempts")
        await self.drive(rounds=3, poll=True)
        methods, text = await self.extracted(u)
        self.assertEqual(methods, ["recognized"] * 4)
        self.assertEqual(len(fake.jobs), 1, "submitted exactly once")

    async def test_first_attempt_skips_the_analysis_lookup_and_resume_reuses_it(self):
        self.use(
            self.document(call_style="job", pending_polls=1),
            extraction_mode="recognize_all",
        )
        lookups, analyses = [], []
        stat, open_pages = self.engine.objects.stat, self.engine.extractor.open_pages

        async def counted_stat(ref):
            if ref.key.endswith("/recognition/analysis.json"):
                lookups.append(ref.key)
            return await stat(ref)

        def counted_open_pages(*args, **kwargs):
            analyses.append(1)
            return open_pages(*args, **kwargs)

        self.engine.objects.stat = counted_stat
        self.engine.extractor.open_pages = counted_open_pages
        u = await self.ingest()
        await self.drive(rounds=1)
        job, _ = await self.extract_job(u)
        self.assertEqual(job.status, JS.WAITING)
        self.assertEqual((lookups, len(analyses)), ([], 1), "attempt 1: no lookup")
        await self.drive(rounds=4, poll=True)
        methods, _ = await self.extracted(u)
        self.assertEqual(methods, ["recognized"] * 4)
        self.assertEqual(len(lookups), 1, "the resume looks the saved analysis up")
        self.assertEqual(len(analyses), 1, "and does not parse the PDF again")

    async def test_rerun_while_pending_never_resubmits(self):
        fake = self.use(
            self.document(call_style="job", pending_polls=99),
            extraction_mode="recognize_all",
        )
        u = await self.ingest()
        await self.drive(rounds=1)
        job, _ = await self.extract_job(u)
        async with self.engine.metadata.db.transaction(self.tenant) as (c, now):
            await self.engine.metadata._requeue(
                c, self.tenant, job, now
            )  # a stray duplicate run
        await self.drive(rounds=1)
        job, _ = await self.extract_job(u)
        self.assertEqual(job.status, JS.WAITING)
        self.assertEqual(len(fake.requests), 1)

    async def test_expired_provider_job_is_resubmitted(self):
        fake = self.use(
            self.document(call_style="job", expire_polls=1),
            extraction_mode="recognize_all",
        )
        u = await self.ingest()
        await self.drive(rounds=5, poll=True)
        await self.extracted(u)
        self.assertEqual(len(fake.requests), 2)
        self.assertNotEqual(fake.requests[0].request_id, fake.requests[1].request_id)

    async def test_resubmission_is_bounded(self):
        self.use(
            self.document(call_style="job", expire_polls=99),
            extraction_mode="recognize_all",
        )
        u = await self.ingest()
        await self.drive(rounds=8, poll=True)
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "recognition_resubmit_limit"))

    async def test_permanent_failure_reported_by_job_fails_the_build(self):
        self.use(
            self.document(call_style="job", failures={2: PermanentInputError("bad")}),
            extraction_mode="recognize_all",
        )
        u = await self.ingest()
        await self.drive(rounds=4, poll=True)
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "recognition_failed"))

    async def test_parking_after_jobs_finished_requeues_immediately(self):
        u = await self.ingest()
        job, _ = await self.extract_job(u)
        m = self.engine.metadata
        claimed = await m.claim_job(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id="w",
            lease_id=uuid4(),
            lease_seconds=60,
        )
        row = await m.record_recognition_job(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id="w",
            lease_id=claimed.lease.lease_id,
            request_id="r" * 20,
            submission=0,
            provider_job_id="p-1",
            page_numbers=(1,),
            poll_after_seconds=0,
        )
        await m.finish_recognition_poll(
            tenant_id=self.tenant, recognition_id=row["id"], state="done"
        )
        parked = await m.park_job_waiting(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id="w",
            lease_id=claimed.lease.lease_id,
            wait_id=uuid4(),
        )
        self.assertEqual(parked.status, JS.QUEUED)

    async def test_deleted_document_abandons_provider_jobs(self):
        self.use(
            self.document(call_style="job", pending_polls=99),
            extraction_mode="recognize_all",
        )
        u = await self.ingest()
        await self.drive(rounds=1)
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=u.document_id, deletion_id=uuid4()
        )
        await self.engine.metadata.reconcile(self.tenant)
        await self.engine.metadata.release_waiting_jobs(tenant_id=self.tenant)
        states = (await self.sql("SELECT state FROM recognition_jobs")).scalars().all()
        self.assertEqual(states, ["abandoned"])

    async def test_waiting_past_the_job_deadline_fails(self):
        from rag_ingest.domain.policy import ExecutionPolicy

        policy = ExecutionPolicy(
            max_job_seconds=4,
            max_attempt_seconds=3,
            lease_seconds=2,
            heartbeat_seconds=1,
        )
        core = replace(self.engine.settings.core, execution_policy=policy)
        self.use(
            self.document(call_style="job", pending_polls=99),
            extraction_mode="recognize_all",
            core=core,
        )
        u = await self.ingest()
        await self.drive(rounds=1)
        import asyncio

        await asyncio.sleep(4.5)
        await self.engine.metadata.reconcile(self.tenant)
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.FAILED, "execution_budget_exhausted"))

    # ---- phase 4: budgets and rate --------------------------------------------

    async def test_shared_rate_limiter_token_bucket(self):
        m = self.engine.metadata
        key = "test:" + uuid4().hex
        waits = [await m.acquire_rate(key=key, rate=1, burst=2) for _ in range(3)]
        self.assertEqual(waits[:2], [0.0, 0.0])
        self.assertGreater(waits[2], 0.5)

    async def test_daily_page_budget_is_idempotent_per_request(self):
        m = self.engine.metadata
        args = dict(tenant_id=self.tenant, daily_limit=3)
        self.assertEqual(
            await m.reserve_recognition_pages(request_id="a" * 12, pages=2, **args), 0
        )
        self.assertEqual(
            await m.reserve_recognition_pages(request_id="a" * 12, pages=2, **args), 0
        )
        self.assertGreater(
            await m.reserve_recognition_pages(request_id="b" * 12, pages=2, **args), 0
        )

    async def test_budget_reset_after_the_job_deadline_fails_clearly(self):
        self.use(
            self.page_image(),
            extraction_mode="recognize_unreadable",
            tenant_daily_recognized_pages=1,
        )
        u = await self.ingest([("image", "A"), ("image", "B")])
        await self.drive(rounds=1)
        job, code = await self.extract_job(u)
        self.assertEqual(
            (job.status, code), (JS.FAILED, "recognition_budget_exhausted")
        )

    async def test_budget_wait_that_fits_is_recorded_as_a_retry(self):
        self.use(
            self.page_image(),
            extraction_mode="recognize_unreadable",
            tenant_daily_recognized_pages=1,
        )
        u = await self.ingest([("image", "A")])

        async def short_wait(**kwargs):
            return 5.0  # budget frees up well before the job deadline

        self.engine.metadata.reserve_recognition_pages = short_wait
        await self.drive(rounds=1)
        job, code = await self.extract_job(u)
        self.assertEqual((job.status, code), (JS.RETRY_WAIT, "recognition_budget_wait"))

    async def test_request_larger_than_daily_budget_fails(self):
        self.use(
            self.document(),
            extraction_mode="recognize_all",
            tenant_daily_recognized_pages=2,
        )
        u = await self.ingest()
        await self.drive()
        _, code = await self.extract_job(u)
        self.assertEqual(code, "recognition_budget_too_small")

    async def test_document_page_cap(self):
        self.use(
            self.page_image(), extraction_mode="recognize_all", max_recognized_pages=3
        )
        u = await self.ingest()
        await self.drive()
        _, code = await self.extract_job(u)
        self.assertEqual(code, "recognition_page_budget_exceeded")

    # ---- end to end -------------------------------------------------------------

    async def test_scanned_text_becomes_searchable(self):
        self.use(
            self.page_image(text_for=lambda p: f"Recovered scan text for page {p}"),
            extraction_mode="recognize_unreadable",
        )
        u = await self.ingest([("image", "ONLY A SCAN")])
        for _ in range(30):
            await self.drive(rounds=1)
            doc = await self.engine.metadata.get_document(
                tenant_id=self.tenant, document_id=u.document_id
            )
            if doc.active_generation_id:
                break
        self.assertIsNotNone(doc.active_generation_id)
        rows = await run_search(self.engine, self.tenant, "Recovered scan text", limit=50)
        self.assertTrue(any("Recovered scan text" in r.text for r in rows))


# ---- conformance kit: passes for well-behaved tools, fails for broken ones ----------


class SyncImageConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
    def make_recognizer(self):
        return FakeRecognizer(unit="page_image")


class SyncDocumentConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
    def make_recognizer(self):
        return FakeRecognizer(unit="document", page_selection=True, inputs={"path"})


class JobDocumentConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
    poll_interval_seconds = 0

    def make_recognizer(self):
        return FakeRecognizer(unit="document", call_style="job", pending_polls=2)


class BrokenToolIsCaught(unittest.TestCase):
    def test_conformance_fails_for_a_tool_that_drops_pages(self):
        class Drops(FakeRecognizer):
            def _result(self, request):
                return RecognitionResult(pages=())

        class Suite(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
            def make_recognizer(self):
                return Drops(unit="document")

        result = unittest.TestResult()
        Suite("test_returns_exactly_the_requested_pages").run(result)
        self.assertEqual(len(result.failures) + len(result.errors), 1)

    def test_conformance_fails_for_secrets_in_the_spec(self):
        class Leaky(FakeRecognizer):
            @property
            def spec(self):
                return ComponentSpec(
                    driver="x",
                    implementation_version="1",
                    options_json='{"api_key": "sk-123"}',
                )

        class Suite(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
            def make_recognizer(self):
                return Leaky(unit="page_image")

        result = unittest.TestResult()
        Suite("test_identity_and_capabilities_are_declared").run(result)
        self.assertEqual(len(result.failures), 1)


if __name__ == "__main__":
    unittest.main()


# ---- found in the live AWS run: NUL characters in extracted PDF text ----------

NUL_PAGES = [("text", "Diffusion models generate images\x00 from noise, step by step.")]


class TextNormalization(unittest.TestCase):
    def test_normalize_removes_unstorable_characters_only(self):
        raw = "a\x00b\x07c\td\ne\rf\ud800g é 中文 \u00a0"
        self.assertEqual(recipe.normalize(raw), "abc\td\ne\rfg é 中文 \u00a0")

    def test_recipes_saved_before_the_option_keep_their_behavior(self):
        self.assertFalse(recipe.validate({})["normalize_text"])


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class UnstorableText(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = PostgresIntegration.asyncSetUp
    asyncTearDown = PostgresIntegration.asyncTearDown
    use, ingest, drive, sql = (
        RecognitionPipeline.use,
        RecognitionPipeline.ingest,
        RecognitionPipeline.drive,
        RecognitionPipeline.sql,
    )
    extract_job, extracted = (
        RecognitionPipeline.extract_job,
        RecognitionPipeline.extracted,
    )

    async def test_database_data_errors_are_permanent_not_uncertain(self):
        with self.assertRaises(PermanentInputError) as caught:
            async with self.engine.metadata.db.transaction(self.tenant) as (c, now):
                await c.execute(sa.text("SELECT CAST('\"\\u0000\"' AS jsonb)"))
        self.assertEqual(caught.exception.code, "unstorable_data")

    async def test_new_builds_strip_nul_and_publish(self):
        u = await self.ingest(NUL_PAGES)
        for _ in range(20):
            await self.drive(rounds=1)
            doc = await self.engine.metadata.get_document(
                tenant_id=self.tenant, document_id=u.document_id
            )
            if doc.active_generation_id:
                break
        self.assertIsNotNone(doc.active_generation_id)
        _, text = await self.extracted(u)
        self.assertNotIn("\x00", text)
        self.assertIn("generate images from noise", text)

    async def test_old_recipe_fails_fast_with_a_clear_code(self):
        original = self.engine.settings.extractor_options

        def without_normalization():
            options = original()
            options["normalize_text"] = False
            return options

        object.__setattr__(
            self.engine.settings, "extractor_options", without_normalization
        )
        u = await self.ingest(NUL_PAGES)
        with self.assertLogs("rag_ingest", level=logging.WARNING) as logs:
            await self.drive(rounds=8)
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant,
            generation_id=doc.desired_generation_id,
            stage=ST.INDEX,
        )
        report = await self.engine.metadata.job_report(self.tenant, jobs[0].job_id)
        self.assertEqual(
            (report["status"], report["failure_code"]), ("failed", "unstorable_data")
        )
        self.assertEqual(
            report["attempt"], 1, "a deterministic data error must not be retried"
        )
        self.assertIn("job_failed", "".join(logs.output))


# ---- found in the live AWS run: idempotent replay across a recipe change -------


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class ReplayAcrossConfigChange(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = PostgresIntegration.asyncSetUp
    asyncTearDown = PostgresIntegration.asyncTearDown
    use, sql = RecognitionPipeline.use, RecognitionPipeline.sql

    async def initiate(self, key, size=100):
        result = await self.engine.initiate_upload(
            self.tenant, filename="a.pdf", size_bytes=size, request_key=key
        )
        return result["upload"]

    async def test_replay_after_recipe_change_returns_the_original_upload(self):
        first = await self.initiate("client-key-1")
        self.use(chunk_size=500)  # server recipe changes (deploy, config edit)
        again = await self.initiate("client-key-1")
        self.assertEqual(again.upload_id, first.upload_id)

    async def test_different_request_with_the_same_key_is_still_refused(self):
        await self.initiate("client-key-3")
        with self.assertRaises(ValueError):
            await self.initiate("client-key-3", size=200)
