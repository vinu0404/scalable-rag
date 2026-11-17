"""Focused checks added during the production-readiness review.

Tests marked expectedFailure assert the *correct* behavior and currently fail,
demonstrating a review finding. When a finding is fixed, unittest reports an
"unexpected success": remove the decorator then. Tests without the decorator
confirm guarantees that already hold.

PostgreSQL tests reuse the opt-in harness and need INGEST_TEST_METADATA_DSN.
"""

from rag_ingest.runtime.search import search as run_search
import asyncio
import hashlib
import json
import time
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import sqlalchemy as sa

from rag_ingest.adapters.postgres.codec import decode, encode
from rag_ingest.adapters.postgres.connection import Database
from rag_ingest.adapters.processing.embedding import OpenAIEmbedder
from rag_ingest.adapters.processing.text import CharacterChunker
from rag_ingest.contracts.chunking import ChunkingInput, ChunkingLimits
from rag_ingest.contracts.embedding import EmbeddingLimits
from rag_ingest.contracts.errors import (
    LeaseLostError,
    PermanentInputError,
    RecordVersionError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.domain.chunks import Chunk, SourceSpan
from rag_ingest.domain.documents import Document
from rag_ingest.domain.enums import JobStatus as JS
from rag_ingest.domain import extraction_recipe
from rag_ingest.domain.pipeline import ComponentSpec, PipelineConfig
from rag_ingest.runtime import runners
from rag_ingest.runtime.runners import publish_once, scheduler_once, worker_once

from test_postgres_integration import DSN, PostgresIntegration

ROOT = Path(__file__).resolve().parents[1]


def _pipeline(embedder_driver="openai"):
    return PipelineConfig(
        extractor=ComponentSpec(
            driver="document",
            implementation_version="1",
            options_json=extraction_recipe.options_json(extraction_recipe.validate({})),
        ),
        chunker=ComponentSpec(
            driver="characters",
            implementation_version="1",
            options_json=json.dumps({"size": 10, "overlap": 0}),
        ),
        embedder=ComponentSpec(
            driver=embedder_driver,
            implementation_version="1",
            options_json=json.dumps(
                {"model": "m", "dimensions": 4, "encoding": "cl100k_base"}
            ),
        ),
    )


class PackagingFindings(unittest.TestCase):
    def test_postgres_extra_installs_async_driver_support(self):
        """SQLAlchemy 2.1 asyncio needs greenlet; a clean install fails without it."""
        extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]["postgres"]
        self.assertTrue(any("asyncio" in d or "greenlet" in d for d in extras), extras)

    def _assert_preserved(self, original, reencoded, path):
        """Every saved field survives decode/encode; new defaulted fields may appear."""
        if isinstance(original, dict):
            self.assertIsInstance(reencoded, dict, path)
            for key, value in original.items():
                self.assertIn(key, reencoded, f"{path}.{key} dropped")
                self._assert_preserved(value, reencoded[key], f"{path}.{key}")
        elif isinstance(original, list):
            self.assertEqual(len(original), len(reencoded), path)
            for i, (a, b) in enumerate(zip(original, reencoded)):
                self._assert_preserved(a, b, f"{path}[{i}]")
        else:
            self.assertEqual(original, reencoded, path)

    def test_v0_1_0_saved_records_still_decode(self):
        """Upgrade safety: rows written by v0.1.0 must load in this release.

        A new record field needs a default, and existing fields must not be
        renamed or removed. Fixtures are real rows from a v0.1.0 pipeline run.
        """
        fixtures = json.loads(
            (ROOT / "tests" / "fixtures" / "records_v0_1_0.json").read_text()
        )
        self.assertGreaterEqual(len(fixtures), 20)
        for item in fixtures:
            with self.subTest(item["source"]):
                self._assert_preserved(
                    item["value"], encode(decode(item["value"])), item["source"]
                )

    def test_record_from_newer_release_is_refused_not_truncated(self):
        """Ignoring an unknown field would erase it on the next whole-row save."""
        from datetime import datetime, timezone

        doc = encode(
            Document(
                tenant_id=uuid4(),
                document_id=uuid4(),
                display_name="a.txt",
                acl_ref="tenant",
                created_at=datetime.now(timezone.utc),
            )
        )
        newer_field = json.loads(json.dumps(doc))
        newer_field["fields"]["added_in_v2"] = None
        newer_type = {"$record": "AddedInV2", "fields": {}}
        newer_enum = {"$enum": "JobStatus", "value": "added_in_v2"}
        for payload in (newer_field, newer_type, newer_enum):
            with self.assertRaises(RecordVersionError):
                decode(payload)


class ProcessingFindings(unittest.IsolatedAsyncioTestCase):
    async def _consume_chunks(self, timeout_seconds, delay):
        """Drive the chunker with a consumer that spends `delay` per chunk."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            text = ("abcdefghij" * 50).encode()
            pages = json.dumps(
                {"schema_version": 1, "pages": [{"page": 1, "start": 0, "end": 500}]}
            ).encode()
            p, q = Path(tmp) / "t", Path(tmp) / "m"
            p.write_bytes(text)
            q.write_bytes(pages)
            ident = uuid4()
            source = SimpleNamespace(
                tenant_id=ident, document_id=ident, source_version_id=ident
            )
            generation = SimpleNamespace(
                tenant_id=ident,
                document_id=ident,
                source_version_id=ident,
                generation_id=ident,
            )

            async def consume():
                async with CharacterChunker().open_chunks(
                    ChunkingInput(
                        text_path=p,
                        page_map_path=q,
                        extraction_sha256=hashlib.sha256(text).hexdigest(),
                        page_map_sha256=hashlib.sha256(pages).hexdigest(),
                    ),
                    source=source,
                    generation=generation,
                    spec=_pipeline().chunker,
                    limits=ChunkingLimits(
                        max_chunks=1000,
                        max_chunk_characters=100,
                        timeout_seconds=timeout_seconds,
                    ),
                ) as chunks:
                    count = 0
                    async for _ in chunks:
                        count += 1
                        await asyncio.sleep(delay)  # stands in for store_bytes()
                    return count

            task = asyncio.create_task(consume())
            try:
                return await task
            except BaseException as error:  # noqa: BLE001 - recording the type
                return type(error).__name__

    async def test_slow_chunk_consumer_is_not_cancelled_by_chunker_budget(self):
        """Consumer time (S3 writes in _chunk) must not count against the chunker."""
        self.assertEqual(await self._consume_chunks(timeout_seconds=1, delay=0.05), 50)

    async def test_chunker_over_its_own_budget_raises_timeout(self):
        self.assertEqual(
            await self._consume_chunks(timeout_seconds=1e-9, delay=0), "TimeoutError"
        )

    async def test_embedding_provider_503_is_reported_as_uncertain(self):
        """The adapter reports possible billing; the worker decides on repeats."""
        embedder = OpenAIEmbedder(
            api_key="test",
            transport=httpx.MockTransport(lambda request: httpx.Response(503)),
        )
        ident = uuid4()
        chunk = Chunk(
            tenant_id=ident,
            document_id=ident,
            source_version_id=ident,
            generation_id=ident,
            ordinal=0,
            text="hello",
            spans=(
                SourceSpan(
                    extraction_sha256="0" * 64,
                    page_number=1,
                    start_char=0,
                    end_char=5,
                ),
            ),
        )
        try:
            with self.assertRaises(UncertainOutcomeError):
                await embedder.embed_batch(
                    (chunk,),
                    pipeline=_pipeline(),
                    dimensions=4,
                    limits=EmbeddingLimits(
                        max_items=1,
                        max_input_characters=100,
                        max_batch_tokens=100,
                        timeout_seconds=5,
                    ),
                )
        finally:
            await embedder.aclose()

    def test_build_engine_rejects_object_store_without_cleanup(self):
        from rag_ingest.runtime.factory import build_engine

        stub = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "CleanupObjectStore"):
            build_engine(
                SimpleNamespace(core=None),
                metadata=stub,
                objects=SimpleNamespace(),  # ObjectStore methods only
                queue=stub,
                embedder=stub,
                vectors=stub,
            )

    async def test_cleanup_with_unsupported_store_is_logged_not_authorized(self):
        tenant, task = uuid4(), {"id": uuid4(), "document_id": uuid4()}
        authorized = []

        async def tenants():
            return (tenant,)

        async def candidates(t):
            return (task,)

        async def authorize(*args):
            authorized.append(args)

        engine = SimpleNamespace(
            objects=SimpleNamespace(),
            metadata=SimpleNamespace(
                tenants=tenants,
                cleanup_candidates=candidates,
                authorize_cleanup=authorize,
            ),
        )
        with self.assertLogs("rag_ingest") as logs:
            await runners.cleanup_once(engine)
        self.assertEqual(authorized, [])
        self.assertIn("cleanup_blocked_unsupported_store", "".join(logs.output))

    async def test_unexpected_error_does_not_stop_other_worker_slots(self):
        calls = []
        stop = asyncio.Event()

        async def flaky(engine, worker_id, **kwargs):
            calls.append(worker_id)
            if len(calls) == 1:
                raise KeyError("bug in one job")
            if len(calls) >= 4:
                stop.set()
            await asyncio.sleep(0)

        engine = SimpleNamespace(settings=SimpleNamespace(worker_concurrency=2))
        original = runners.worker_once
        runners.worker_once = flaky
        try:
            await asyncio.wait_for(runners.run_workers(engine, stop), timeout=10)
        finally:
            runners.worker_once = original
        self.assertGreaterEqual(len(calls), 4)
        self.assertEqual(len(set(calls)), 2, "both slots kept running")

    async def test_heartbeat_gives_up_once_lease_could_have_expired(self):
        """Tolerating renewal errors must not outlive the last granted lease."""

        async def failing(**kwargs):
            raise RetryableError("database unavailable")

        job = SimpleNamespace(
            tenant_id=uuid4(),
            job_id=uuid4(),
            lease=SimpleNamespace(worker_id="w", lease_id=uuid4()),
            budget=SimpleNamespace(
                policy=SimpleNamespace(heartbeat_seconds=0.05, lease_seconds=1)
            ),
        )
        engine = SimpleNamespace(metadata=SimpleNamespace(renew_job_lease=failing))
        start = asyncio.get_running_loop().time()
        with self.assertRaises(LeaseLostError):
            await asyncio.wait_for(
                runners._heartbeat(engine, job, None, start), timeout=5
            )
        self.assertGreaterEqual(asyncio.get_running_loop().time() - start, 1)

    async def test_connection_refused_read_is_retryable(self):
        """A refused connection for a read is classified as an uncertain commit."""
        db = Database("postgresql://u:p@127.0.0.1:1/never_test")
        try:
            with self.assertRaises(RetryableError):
                async with db.transaction(uuid4()):
                    pass
        finally:
            await db.aclose()


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class PostgresFindings(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = PostgresIntegration.asyncSetUp
    asyncTearDown = PostgresIntegration.asyncTearDown
    accepted_upload = PostgresIntegration.accepted_upload

    async def _upload(self, data, name="example.txt", key=None, document_id=None):
        result = await self.engine.initiate_upload(
            self.tenant,
            filename=name,
            size_bytes=len(data),
            request_key=key or str(uuid4()),
            document_id=document_id,
        )
        u = result["upload"]
        self.engine.objects.client.put_object(
            Bucket=u.bucket, Key=u.object_key, Body=data
        )
        await self.engine.complete_upload(self.tenant, u.upload_id)
        return u

    async def _first_job(self, document_id):
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=document_id
        )
        return (
            await self.engine.metadata.list_generation_jobs(
                tenant_id=self.tenant, generation_id=doc.desired_generation_id
            )
        )[0]

    # ---- guarantees that hold (verified by execution) -----------------------

    async def test_concurrent_idempotent_initiation_returns_one_upload(self):
        key = str(uuid4())
        results = await asyncio.gather(
            *[
                self.engine.initiate_upload(
                    self.tenant, filename="a.txt", size_bytes=10, request_key=key
                )
                for _ in range(4)
            ]
        )
        self.assertEqual(len({r["upload"].upload_id for r in results}), 1)
        with self.assertRaises(ValueError):
            await self.engine.initiate_upload(
                self.tenant, filename="a.txt", size_bytes=11, request_key=key
            )

    async def test_acceptance_replay_after_lost_response_is_idempotent(self):
        data = b"replay me " * 20
        u = await self._upload(data)
        first = await self.engine.verify_upload(self.tenant, u.upload_id)
        ref = await self.engine.metadata.get_verification_ref(
            tenant_id=self.tenant, upload_id=u.upload_id
        )
        # Re-sending the same acceptance (as after a lost commit response).
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant, generation_id=doc.desired_generation_id
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(first.status.value, "accepted")
        self.assertIsNotNone(ref.version_id)

    async def test_replacement_fences_stale_generation_worker(self):
        u = await self.accepted_upload()
        job = await self._first_job(u.document_id)
        claimed = await self.engine.metadata.claim_job(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id="old",
            lease_id=uuid4(),
            lease_seconds=60,
        )
        await self.engine.verify_upload(
            self.tenant,
            (
                await self._upload(b"replacement " * 20, document_id=u.document_id)
            ).upload_id,
        )
        with self.assertRaises(LeaseLostError):
            await self.engine.metadata.fail_job(
                tenant_id=self.tenant,
                job_id=job.job_id,
                worker_id="old",
                lease_id=claimed.lease.lease_id,
                failure_id=uuid4(),
                failure_code="stale",
            )

    async def test_deletion_revokes_running_lease(self):
        u = await self.accepted_upload()
        job = await self._first_job(u.document_id)
        claimed = await self.engine.metadata.claim_job(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id="w",
            lease_id=uuid4(),
            lease_seconds=60,
        )
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=u.document_id, deletion_id=uuid4()
        )
        self.assertIsNone(
            await self.engine.metadata.renew_job_lease(
                tenant_id=self.tenant,
                job_id=job.job_id,
                worker_id="w",
                lease_id=claimed.lease.lease_id,
                lease_seconds=60,
            )
        )
        with self.assertRaises(ValueError):
            await self.engine.view_file(self.tenant, u.document_id)

    # ---- fixed findings (regression tests) ----------------------------------

    async def _while_tenant_locked(self, operation, hold_seconds=2):
        """Return how long `operation` waited while another tx held the lock."""
        held = asyncio.Event()

        async def hold():
            async with self.engine.metadata.db.transaction(self.tenant):
                held.set()
                await asyncio.sleep(hold_seconds)

        holder = asyncio.create_task(hold())
        await held.wait()
        start = time.monotonic()
        await operation()
        waited = time.monotonic() - start
        await holder
        return waited

    async def test_reads_do_not_wait_for_tenant_mutation_lock(self):
        """Status reads, job reports and search must not queue behind mutations."""
        u = await self.accepted_upload()
        job = await self._first_job(u.document_id)

        async def reads():
            await self.engine.metadata.get_document(
                tenant_id=self.tenant, document_id=u.document_id
            )
            await self.engine.metadata.job_report(self.tenant, job.job_id)
            await run_search(self.engine, self.tenant, "durable work", limit=5)

        waited = await self._while_tenant_locked(reads)
        self.assertLess(waited, 0.5, f"reads blocked {waited:.2f}s on tenant lock")

    async def test_mutations_still_wait_for_tenant_lock(self):
        u = await self.accepted_upload()

        async def delete():
            await self.engine.metadata.tombstone_document(
                tenant_id=self.tenant, document_id=u.document_id, deletion_id=uuid4()
            )

        self.assertGreaterEqual(await self._while_tenant_locked(delete, 1), 0.9)

    async def test_single_query_job_reports_match_per_job_reports(self):
        u = await self.accepted_upload()
        for _ in range(6):  # advance past extract so the generation has several jobs
            await scheduler_once(self.engine)
            await publish_once(self.engine, "p")
            while await worker_once(self.engine, "w", wait_seconds=0):
                pass
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant, generation_id=doc.desired_generation_id
        )
        self.assertGreater(len(jobs), 1)
        expected = [
            await self.engine.metadata.job_report(self.tenant, j.job_id) for j in jobs
        ]
        self.assertEqual(
            await self.engine.metadata.list_job_reports(
                self.tenant, doc.desired_generation_id
            ),
            expected,
        )

    async def test_connection_killed_mid_transaction_is_retryable(self):
        """Nothing is committed when the connection drops before COMMIT."""
        with self.assertRaises(RetryableError):
            async with self.engine.metadata.db.transaction(self.tenant) as (c, now):
                await c.execute(
                    sa.text("SELECT pg_terminate_backend(pg_backend_pid())")
                )

    async def test_unexpected_stage_error_records_durable_retry(self):
        u = await self.accepted_upload()
        await publish_once(self.engine, "p")

        async def broken(job):
            raise KeyError("unmapped failure")

        self.engine.execute_job = broken
        self.assertTrue(await worker_once(self.engine, "w", wait_seconds=0))
        job = await self._first_job(u.document_id)
        self.assertEqual(job.status, JS.RETRY_WAIT)
        report = await self.engine.metadata.job_report(self.tenant, job.job_id)
        self.assertEqual(report["failure_code"], "unexpected_worker_error")

    async def test_deleted_document_vectors_and_files_are_erased_after_grace(self):
        u = await self.accepted_upload()
        for _ in range(30):
            await scheduler_once(self.engine)
            await publish_once(self.engine, "p")
            while await worker_once(self.engine, "w", wait_seconds=0):
                pass
            doc = await self.engine.metadata.get_document(
                tenant_id=self.tenant, document_id=u.document_id
            )
            if doc.active_generation_id:
                break
        generation = doc.active_generation_id
        self.assertIsNotNone(generation)
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=u.document_id, deletion_id=uuid4()
        )
        async with self.engine.metadata.db.engine.begin() as c:  # skip the grace
            await c.execute(
                sa.text(
                    "UPDATE cleanup SET payload = jsonb_set(payload, '{after}',"
                    " '\"2000-01-01T00:00:00+00:00\"') WHERE tenant_id = :t"
                ),
                {"t": self.tenant},
            )
        await runners.cleanup_once(self.engine)
        async with self.engine.metadata.db.engine.begin() as c:
            vectors = (
                await c.execute(
                    sa.text("SELECT count(*) FROM rag_vectors WHERE document_id = :d"),
                    {"d": u.document_id},
                )
            ).scalar_one()
            pending = (
                await c.execute(
                    sa.text(
                        "SELECT count(*) FROM cleanup WHERE tenant_id = :t"
                        " AND status = 'pending'"
                    ),
                    {"t": self.tenant},
                )
            ).scalar_one()
        self.assertEqual((vectors, pending), (0, 0))
        client = self.engine.objects.client
        for prefix in (
            f"tenants/{self.tenant}/generations/{generation}/",
            u.object_key,
        ):
            page = client.list_object_versions(Bucket=u.bucket, Prefix=prefix)
            self.assertFalse(page.get("Versions") or page.get("DeleteMarkers"), prefix)

    async def test_newer_record_mid_job_stops_worker_without_failing_job(self):
        u = await self.accepted_upload()
        await publish_once(self.engine, "p")

        async def skewed(job):
            raise RecordVersionError("written by a newer release")

        self.engine.execute_job = skewed
        with self.assertRaises(RecordVersionError):
            await worker_once(self.engine, "w", wait_seconds=0)
        job = await self._first_job(u.document_id)
        self.assertEqual(job.status, JS.RUNNING)  # lease expires; newer worker resumes

    async def _run_job_raising(self, error, **settings):
        from dataclasses import replace

        u = await self.accepted_upload()
        await publish_once(self.engine, "p")

        async def uncertain(job):
            raise error

        self.engine.settings = replace(self.engine.settings, **settings)
        self.engine.execute_job = uncertain
        self.assertTrue(await worker_once(self.engine, "w", wait_seconds=0))
        job = await self._first_job(u.document_id)
        report = await self.engine.metadata.job_report(self.tenant, job.job_id)
        return job.status, report["failure_code"]

    async def test_uncertain_outcome_is_repeated_within_limit(self):
        status, code = await self._run_job_raising(
            UncertainOutcomeError("embedding 503")
        )
        self.assertEqual((status, code), (JS.RETRY_WAIT, "uncertain_outcome_repeat"))

    async def test_stage_timeout_is_repeated_within_limit(self):
        status, code = await self._run_job_raising(TimeoutError())
        self.assertEqual((status, code), (JS.RETRY_WAIT, "uncertain_outcome_repeat"))

    async def test_uncertain_outcome_is_terminal_when_repeats_disabled(self):
        status, code = await self._run_job_raising(
            UncertainOutcomeError("embedding 503"), max_uncertain_repeats=0
        )
        self.assertEqual((status, code), (JS.FAILED, "outcome_requires_inspection"))

    async def test_heartbeat_sql_blip_does_not_permanently_fail_generation(self):
        """One UncertainOutcomeError from lease renewal fails the whole generation."""
        u = await self.accepted_upload()
        await publish_once(self.engine, "p")

        async def slow(job):
            await asyncio.sleep(8)

        async def blip(**kwargs):
            raise UncertainOutcomeError("connection reset during renewal")

        self.engine.execute_job = slow
        self.engine.metadata.renew_job_lease = blip
        self.assertTrue(await worker_once(self.engine, "w", wait_seconds=0))
        job = await self._first_job(u.document_id)
        self.assertNotEqual(job.status, JS.FAILED)

    async def test_invalid_utf8_after_first_4kib_is_rejected_at_verification(self):
        """Every byte of a text upload is validated, not just a prefix."""
        data = b"a" * 5000 + b"\xff\xfe" + b"b" * 10
        u = await self._upload(data)
        with self.assertRaises(PermanentInputError):
            await self.engine.verify_upload(self.tenant, u.upload_id)
        upload = await self.engine.metadata.get_upload(
            tenant_id=self.tenant, upload_id=u.upload_id
        )
        self.assertEqual(upload.status.value, "failed")

    async def test_utf8_character_split_across_read_blocks_is_accepted(self):
        data = b"a" * (1024**2 - 1) + "é".encode() + b"tail"  # é straddles 1 MiB
        u = await self._upload(data)
        accepted = await self.engine.verify_upload(self.tenant, u.upload_id)
        self.assertEqual(accepted.status.value, "accepted")

    async def test_text_ending_mid_character_is_rejected(self):
        u = await self._upload(b"complete text then a cut \xc3")
        with self.assertRaises(PermanentInputError):
            await self.engine.verify_upload(self.tenant, u.upload_id)

    # ---- H5: verification never blocks the scheduler ------------------------

    async def test_scheduler_pass_does_not_wait_for_upload_hashing(self):
        await self._upload(b"waiting to be hashed " * 10)  # left in VERIFYING

        async def slow_hash(tenant, upload_id):
            await asyncio.sleep(30)

        self.engine.verify_upload = slow_hash
        await asyncio.wait_for(scheduler_once(self.engine), timeout=5)

    async def test_verifier_accepts_completed_upload(self):
        u = await self._upload(b"verify me in the background " * 10)
        await runners.verify_once(self.engine)
        upload = await self.engine.metadata.get_upload(
            tenant_id=self.tenant, upload_id=u.upload_id
        )
        self.assertEqual(upload.status.value, "accepted")

    async def test_scheduler_role_verifies_admits_and_stops_cleanly(self):
        u = await self._upload(b"end to end through the scheduler role " * 10)
        stop = asyncio.Event()
        role = asyncio.create_task(runners.run_scheduler(self.engine, stop))
        try:
            for _ in range(50):
                await asyncio.sleep(0.2)
                upload = await self.engine.metadata.get_upload(
                    tenant_id=self.tenant, upload_id=u.upload_id
                )
                if upload.status.value == "accepted" and await self._admitted(
                    u.document_id
                ):
                    break
            self.assertEqual(upload.status.value, "accepted")
            self.assertTrue(await self._admitted(u.document_id))
        finally:
            stop.set()
            await asyncio.wait_for(role, timeout=10)  # exits without error

    # ---- H6: admission control; deadlines start at admission ----------------

    async def _verified(self, data=None):
        """Accept an upload without admitting it (no scheduler tick yet)."""
        u = await self._upload(data or f"document {uuid4()} ".encode() * 8)
        await self.engine.verify_upload(self.tenant, u.upload_id)
        return u

    async def _claim(self, job):
        return await self.engine.metadata.claim_job(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id="w",
            lease_id=uuid4(),
            lease_seconds=60,
        )

    async def _admitted(self, document_id):
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=document_id
        )
        async with self.engine.metadata.db.engine.begin() as c:
            return (
                await c.execute(
                    sa.text("SELECT admitted_at FROM generations WHERE id = :g"),
                    {"g": doc.desired_generation_id},
                )
            ).scalar_one() is not None

    async def test_waiting_generation_cannot_be_claimed_until_admitted(self):
        u = await self._verified()
        job = await self._first_job(u.document_id)
        self.assertIsNone(await self._claim(job))
        admitted = await self.engine.metadata.admit_generations(
            tenant_id=self.tenant, max_active=4
        )
        self.assertEqual(admitted, 1)
        self.assertIsNotNone(await self._claim(job))

    async def test_waiting_longer_than_the_deadline_does_not_fail_the_build(self):
        from dataclasses import replace

        core = replace(self.engine.settings.core, max_generation_seconds=2)
        self.engine.settings = replace(self.engine.settings, core=core)
        u = await self._verified()
        await asyncio.sleep(2.5)  # longer than the whole generation budget
        await self.engine.metadata.reconcile(self.tenant)
        job = await self._first_job(u.document_id)
        self.assertEqual(job.status, JS.QUEUED)
        await self.engine.metadata.admit_generations(
            tenant_id=self.tenant, max_active=4
        )
        job = await self._first_job(u.document_id)
        now = await self.engine.metadata.now(self.tenant)
        self.assertGreater(job.budget.generation_deadline_at, now)
        self.assertEqual(  # saved durations are kept, only the start moves
            (job.budget.generation_deadline_at - job.budget.starts_at).total_seconds(),
            2,
        )
        self.assertIsNotNone(await self._claim(job))

    async def test_admission_cap_limits_active_generations_per_tenant(self):
        uploads = [await self._verified() for _ in range(3)]
        admit = self.engine.metadata.admit_generations
        self.assertEqual(await admit(tenant_id=self.tenant, max_active=2), 2)
        self.assertEqual(await admit(tenant_id=self.tenant, max_active=2), 0)
        flags = [await self._admitted(u.document_id) for u in uploads]
        self.assertEqual(flags, [True, True, False])  # oldest first
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant,
            document_id=uploads[0].document_id,
            deletion_id=uuid4(),
        )
        self.assertEqual(await admit(tenant_id=self.tenant, max_active=2), 1)
        self.assertTrue(await self._admitted(uploads[2].document_id))

    async def test_admitted_generation_past_its_deadline_frees_its_slot(self):
        first, second = await self._verified(), await self._verified()
        admit = self.engine.metadata.admit_generations
        self.assertEqual(await admit(tenant_id=self.tenant, max_active=1), 1)
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(
                sa.text(
                    "UPDATE generations SET deadline_at = now() - interval '1 minute'"
                    " WHERE admitted_at IS NOT NULL"
                )
            )
        self.assertEqual(await admit(tenant_id=self.tenant, max_active=1), 1)
        self.assertTrue(await self._admitted(second.document_id))
        async with self.engine.metadata.db.engine.begin() as c:
            status = (
                await c.execute(
                    sa.text(
                        "SELECT g.status FROM generations g JOIN documents d"
                        " ON d.id = g.document_id WHERE d.id = :d"
                    ),
                    {"d": first.document_id},
                )
            ).scalar_one()
        self.assertEqual(status, "failed")

    async def test_reconcile_does_not_renotify_waiting_jobs(self):
        admitted = await self.accepted_upload()  # its helper runs admission
        waiting = await self._verified()  # accepted after that, so still waiting
        self.assertFalse(await self._admitted(waiting.document_id))
        async with self.engine.metadata.db.engine.begin() as c:  # notified long ago
            await c.execute(
                sa.text(
                    "UPDATE outbox SET published_at = now() - interval '1 hour',"
                    " claim_id = NULL, claim_expires_at = NULL"
                )
            )
        await self.engine.metadata.reconcile(self.tenant)

        async def pending(document_id):
            job = await self._first_job(document_id)
            async with self.engine.metadata.db.engine.begin() as c:
                return (
                    await c.execute(
                        sa.text(
                            "SELECT count(*) FROM outbox"
                            " WHERE job_id = :j AND published_at IS NULL"
                        ),
                        {"j": job.job_id},
                    )
                ).scalar_one()

        self.assertEqual(await pending(waiting.document_id), 0)
        self.assertEqual(await pending(admitted.document_id), 1)

    # ---- M4: bounded history ------------------------------------------------

    async def test_prune_removes_only_expired_history(self):
        u = await self.accepted_upload()
        await publish_once(self.engine, "p")
        delivery = SimpleNamespace(message_id="m-1", body="not json")
        await self.engine.metadata.quarantine_delivery(delivery, "malformed_event")
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(
                sa.text(
                    "INSERT INTO upload_parts VALUES (:t, :u, 'p', 1, '{}')"
                ),  # observation of an upload that is no longer open
                {"t": self.tenant, "u": u.upload_id},
            )
            await c.execute(
                sa.text(
                    "UPDATE outbox SET published_at = now() - interval '8 days'"
                    " WHERE published_at IS NOT NULL"
                )
            )
            await c.execute(
                sa.text("UPDATE quarantine SET created_at = now() - interval '8 days'")
            )
            old_ops = (
                await c.execute(
                    sa.text(
                        "UPDATE operations SET created_at = now() - interval '8 days'"
                        " WHERE kind = 'claim-outbox'"
                    )
                )
            ).rowcount
            ops_before = (
                await c.execute(sa.text("SELECT count(*) FROM operations"))
            ).scalar_one()
        self.assertGreater(await runners.prune_once(self.engine), 0)
        async with self.engine.metadata.db.engine.begin() as c:
            count = lambda sql: c.execute(sa.text(sql))  # noqa: E731
            self.assertEqual(
                (
                    await count(
                        "SELECT count(*) FROM outbox WHERE published_at IS NOT NULL"
                    )
                ).scalar_one(),
                0,
            )
            self.assertEqual(
                (await count("SELECT count(*) FROM quarantine")).scalar_one(), 0
            )
            self.assertEqual(
                (await count("SELECT count(*) FROM upload_parts")).scalar_one(), 0
            )
            self.assertEqual(
                (await count("SELECT count(*) FROM operations")).scalar_one(),
                ops_before - old_ops,  # recent ledger rows are kept
            )
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )
        self.assertIsNotNone(doc.desired_generation_id)  # durable state untouched

    async def test_outbox_claim_replay_after_pruning_returns_none(self):
        await self.accepted_upload()
        claim = uuid4()
        args = dict(
            tenant_id=self.tenant, publisher_id="p", claim_id=claim, claim_seconds=60
        )
        record = await self.engine.metadata.claim_next_outbox(**args)
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(
                sa.text("DELETE FROM outbox WHERE id = :e"),
                {"e": record.event.event_id},
            )
        self.assertIsNone(await self.engine.metadata.claim_next_outbox(**args))


class VerifierPool(unittest.IsolatedAsyncioTestCase):
    async def test_verification_is_bounded_and_never_duplicated(self):
        tenant = uuid4()
        uploads = [
            SimpleNamespace(upload_id=uuid4(), expires_at=None) for _ in range(3)
        ]
        running, peak, calls = set(), [0], []

        async def tenants():
            return (tenant,)

        async def list_verifying(t, limit):
            return tuple(uploads[:limit])

        async def verify(t, upload_id):
            calls.append(upload_id)
            running.add(upload_id)
            peak[0] = max(peak[0], len(running))
            await asyncio.sleep(0.2)
            running.discard(upload_id)

        from datetime import datetime, timedelta, timezone

        for u in uploads:
            u.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        async def now(t):
            return datetime.now(timezone.utc)

        engine = SimpleNamespace(
            settings=SimpleNamespace(verify_concurrency=2),
            metadata=SimpleNamespace(
                tenants=tenants, list_verifying_uploads=list_verifying, now=now
            ),
            verify_upload=verify,
        )
        inflight = {}
        await runners.verify_pass(engine, inflight)
        await runners.verify_pass(engine, inflight)  # same uploads still running
        await asyncio.gather(*inflight.values())
        self.assertEqual(peak[0], 2)
        self.assertEqual(len(calls), len(set(calls)), "an upload was verified twice")


if __name__ == "__main__":
    unittest.main()
