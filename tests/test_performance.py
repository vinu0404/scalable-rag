"""Round-trip and byte-count optimizations keep every safety check in place."""

import hashlib
import io
import json
import unittest
from collections import Counter
from dataclasses import replace
from uuid import uuid4

import boto3
import sqlalchemy as sa

from rag_ingest.adapters.postgres.codec import decode, encode
from rag_ingest.contracts.errors import PermanentInputError, RetryableError
from rag_ingest.domain.enums import JobStatus as JS, ProcessingStage as ST
from rag_ingest.adapters.aws.s3 import (
    EXPECT_CONTINUE_MIN_BYTES,
    S3ObjectStore,
    expect_continue_for_large_bodies,
)

# Modules, not classes: importing a TestCase class would run its tests here too.
import test_embedding_cache as segments
import test_postgres_integration as integration

DSN = integration.DSN


class _Model:
    http = {"method": "PUT"}


class ExpectContinue(unittest.TestCase):
    def headers_for(self, size):
        params = {
            "body": io.BytesIO(b"x" * min(size, 16)),
            "headers": {"Content-Length": str(size)},
        }
        expect_continue_for_large_bodies(_Model(), params)
        return params["headers"]

    def test_small_bodies_skip_the_extra_round_trip(self):
        self.assertNotIn("Expect", self.headers_for(20_000))
        self.assertNotIn("Expect", self.headers_for(EXPECT_CONTINUE_MIN_BYTES - 1))

    def test_large_bodies_keep_expect_continue(self):
        self.assertEqual(
            self.headers_for(EXPECT_CONTINUE_MIN_BYTES)["Expect"], "100-continue"
        )

    def test_store_client_uses_the_size_aware_handler(self):
        from botocore.handlers import add_expect_header

        session = boto3.Session(
            aws_access_key_id="x", aws_secret_access_key="x", region_name="us-east-1"
        )
        store = S3ObjectStore(region="us-east-1", session=session)
        names = list(
            store.client.meta.events._emitter._handlers.prefix_search(
                "before-call.s3.PutObject"
            )
        )
        self.assertIn(expect_continue_for_large_bodies, names)
        self.assertNotIn(add_expect_header, names)


# ---- embedding jobs read one plan row, not the whole inventory ---------------


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class BatchPlans(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown
    ingest = segments.EmbeddingSegments.ingest
    drive = segments.EmbeddingSegments.drive
    doc = segments.EmbeddingSegments.doc
    embed_job = segments.EmbeddingSegments.embed_job

    def count_inventory_reads(self):
        reads, original = [], self.engine.inventory

        async def counted(job):
            reads.append(job.stage)
            return await original(job)

        self.engine.inventory = counted
        return reads

    async def plan_rows(self, generation):
        async with self.engine.metadata.db.engine.begin() as c:
            rows = await c.execute(
                sa.text(
                    "SELECT batch_key, plan FROM scheduled_batches"
                    " WHERE tenant_id=:t AND generation_id=:g"
                ),
                {"t": self.tenant, "g": generation},
            )
            return {key: plan for key, plan in rows}

    async def test_embedding_jobs_never_download_the_inventory(self):
        reads = self.count_inventory_reads()
        u = await self.ingest()
        await self.drive()
        d = await self.doc(u)
        self.assertEqual(d.active_generation_id, d.desired_generation_id)
        self.assertNotIn(ST.EMBED, reads)
        inv = await self.engine.metadata.get_inventory(
            tenant_id=self.tenant, generation_id=d.active_generation_id
        )
        rows = await self.plan_rows(d.active_generation_id)
        self.assertEqual(set(rows), {b.batch_key for b in inv.batches})
        for key, plan in rows.items():
            self.assertEqual(decode(plan), inv.plan_for(key))

    async def test_a_missing_plan_is_refused(self):
        u = await self.ingest()
        for _ in range(10):
            await self.drive(rounds=1)
            job = await self.embed_job(u)
            if job:
                break
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(sa.text("DELETE FROM scheduled_batches WHERE tenant_id=:t"), {"t": self.tenant})
        with self.assertRaisesRegex(ValueError, "sealed inventory"):
            await self.engine._batch_plan(job)

    async def test_a_plan_from_another_inventory_is_refused(self):
        u = await self.ingest()
        for _ in range(10):
            await self.drive(rounds=1)
            job = await self.embed_job(u)
            if job:
                break
        rows = await self.plan_rows(job.generation_id)
        plan = decode(rows[job.shard_key])
        forged = replace(plan, inventory_sha256="f" * 64)
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(
                sa.text(
                    "UPDATE scheduled_batches SET plan = CAST(:p AS jsonb)"
                    " WHERE tenant_id=:t AND batch_key=:k"
                ),
                {"p": json.dumps(encode(forged)), "t": self.tenant, "k": job.shard_key},
            )
        with self.assertRaisesRegex(ValueError, "sealed inventory"):
            await self.engine._batch_plan(job)



# ---- publication verifies digests instead of re-reading every vector ---------


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class PublishDigests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown
    ingest = segments.EmbeddingSegments.ingest
    drive = segments.EmbeddingSegments.drive
    doc = segments.EmbeddingSegments.doc

    def count_segment_reads(self):
        """Segment reads per stage, counted where the engine reads stored outputs."""
        reads, original = Counter(), self.engine.read_output
        stage = {"name": None}
        execute = self.engine.execute_job

        async def tracked(job):
            stage["name"] = job.stage
            try:
                return await execute(job)
            finally:
                stage["name"] = None

        async def counted(output, *args, **kwargs):
            if output.object_key.endswith(".seg"):
                reads[stage["name"]] += 1
            return await original(output, *args, **kwargs)

        self.engine.execute_job, self.engine.read_output = tracked, counted
        return reads

    def after_index(self, sql, ordinal):
        """Change one stored vector right after the index stage commits its writes."""
        original = self.engine._index

        async def index_then_change(job, *args):
            result = await original(job, *args)
            async with self.engine.metadata.db.engine.begin() as c:
                await c.execute(
                    sa.text(sql), {"t": self.tenant, "id": f"%:{ordinal}"}
                )
            return result

        self.engine._index = index_then_change

    async def publish_job(self, u):
        d = await self.doc(u)
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant, generation_id=d.desired_generation_id, stage=ST.PUBLISH
        )
        return jobs[0] if jobs else None

    async def test_publish_reads_no_vectors_when_every_digest_matches(self):
        reads = self.count_segment_reads()
        u = await self.ingest()
        await self.drive()
        d = await self.doc(u)
        self.assertEqual(d.active_generation_id, d.desired_generation_id)
        inv = await self.engine.metadata.get_inventory(
            tenant_id=self.tenant, generation_id=d.active_generation_id
        )
        self.assertGreater(len(inv.batches), 1)
        self.assertEqual(reads[ST.INDEX], len(inv.batches))  # index reads each once
        self.assertEqual(reads[ST.PUBLISH], 0)

    async def test_a_changed_vector_still_blocks_publication_permanently(self):
        self.after_index(
            "UPDATE rag_vectors SET embedding = CAST('[1,1,1,1,1,1,1,1]' AS vector)"
            " WHERE tenant_id=:t AND vector_id LIKE :id",
            0,
        )
        with self.assertLogs("rag_ingest.runtime.engine", "WARNING") as logs:
            u = await self.ingest()
            await self.drive()
        self.assertTrue(any("publish_digest_mismatch" in m for m in logs.output))
        d = await self.doc(u)
        self.assertIsNone(d.active_generation_id)
        self.assertEqual((await self.publish_job(u)).status, JS.FAILED)

    async def test_a_missing_vector_blocks_publication_and_is_retried(self):
        self.after_index(
            "DELETE FROM rag_vectors WHERE tenant_id=:t AND vector_id LIKE :id", 1
        )
        u = await self.ingest()
        for _ in range(25):
            await self.drive(rounds=1)
            job = await self.publish_job(u)
            if job and job.attempt >= 1 and job.status != JS.QUEUED:
                break
        self.assertIsNone((await self.doc(u)).active_generation_id)
        self.assertEqual(job.status, JS.RETRY_WAIT)

    async def test_digest_covers_payload_and_exact_vector(self):
        u = await self.ingest()
        await self.drive()
        d = await self.doc(u)
        vectors = self.engine.vectors
        inv = await self.engine.metadata.get_inventory(
            tenant_id=self.tenant, generation_id=d.active_generation_id
        )
        job = await self.publish_job(u)
        mapping = await self.engine._embed_jobs(job, inv)
        entry = inv.batches[0]
        items = await self.engine._batch_items(
            job, inv, entry, mapping[entry.batch_key]
        )
        from rag_ingest.contracts.vectors import VectorScope

        scope = VectorScope(
            tenant_id=self.tenant,
            document_id=d.document_id,
            generation_id=d.active_generation_id,
        )
        ids = tuple(i.chunk.vector_id for i in items)
        (stored,) = await vectors.stored_digests(scope, batches=(ids,))
        self.assertEqual(stored, vectors.expected_digest(scope, items))
        nudged = replace(
            items[0], vector=(items[0].vector[0] + 1e-3,) + items[0].vector[1:]
        )
        self.assertNotEqual(
            stored, vectors.expected_digest(scope, (nudged,) + items[1:])
        )
        (none,) = await vectors.stored_digests(scope, batches=((ids[0] + "9999",),))
        self.assertIsNone(none)



# ---- pgvector writes: one insert and one read-back per batch -----------------


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class BatchedVectorWrites(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    def items(self, count, changed=()):
        from test_embedding_segments import item

        return tuple(
            item(n, tuple(float(n + d + (0.5 if n in changed else 0)) for d in range(8)))
            for n in range(count)
        )

    def scope(self):
        from rag_ingest.contracts.vectors import VectorScope
        from test_embedding_segments import IDS

        return VectorScope(tenant_id=IDS, document_id=IDS, generation_id=IDS)

    def count_statements(self):
        statements = []
        sa.event.listen(
            self.engine.vectors.db.engine.sync_engine,
            "before_cursor_execute",
            lambda *a, **k: statements.append(a[2]),
        )
        return statements

    async def test_statement_count_does_not_grow_with_batch_size(self):
        vectors, scope = self.engine.vectors, self.scope()
        statements = self.count_statements()
        results = await vectors.upsert_batch(scope, items=self.items(60))
        writes = [q for q in statements if "rag_vectors" in q]
        self.assertEqual(len(writes), 2, writes)  # one INSERT, one read-back
        self.assertTrue(all(r.error is None for r in results))
        statements.clear()
        checks = await vectors.verify_batch(scope, expected=self.items(60))
        self.assertEqual(len([q for q in statements if "rag_vectors" in q]), 1)
        self.assertTrue(all(c.readiness.value == "ready" for c in checks))

    async def test_existing_rows_are_kept_and_a_different_payload_is_refused(self):
        from rag_ingest.contracts.errors import PermanentInputError

        vectors, scope = self.engine.vectors, self.scope()
        await vectors.upsert_batch(scope, items=self.items(4))
        again = await vectors.upsert_batch(scope, items=self.items(4, changed={2}))
        self.assertEqual(
            [type(r.error) if r.error else None for r in again],
            [None, None, PermanentInputError, None],
        )
        checks = await vectors.verify_batch(scope, expected=self.items(4))
        self.assertTrue(all(c.readiness.value == "ready" for c in checks))



# ---- intake: skip the second download when storage verified the SHA-256 -------


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class StorageVerifiedIntake(unittest.IsolatedAsyncioTestCase):
    """moto does not report SHA-256 checksums, so S3's answer is simulated.

    Real S3 behaviour this relies on (checked against the live bucket): a
    presigned single PUT carrying x-amz-checksum-sha256 is refused on a
    mismatch (BadDigest), and HEAD with ChecksumMode=ENABLED then returns
    that SHA-256 with ChecksumType FULL_OBJECT.
    """

    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown
    PDF = b"%PDF-1.4\n" + b"x" * 9000

    def storage_reports(self, sha256=None):
        objects, full_reads = self.engine.objects, []
        open_reader = objects.open_reader

        async def verified_sha256(ref):
            data = objects.client.get_object(
                Bucket=ref.bucket, Key=ref.key, VersionId=ref.version_id
            )["Body"].read()
            return len(data), sha256 or hashlib.sha256(data).hexdigest()

        def counted_open_reader(ref, **kwargs):
            full_reads.append(ref.key)
            return open_reader(ref, **kwargs)

        objects.verified_sha256 = verified_sha256
        objects.open_reader = counted_open_reader
        return full_reads

    async def upload(self, data, *, name="paper.pdf", declare=True):
        r = await self.engine.initiate_upload(
            self.tenant,
            filename=name,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest() if declare else None,
            request_key=str(uuid4()),
        )
        u = r["upload"]
        self.engine.objects.client.put_object(
            Bucket=u.bucket, Key=u.object_key, Body=data
        )
        await self.engine.complete_upload(self.tenant, u.upload_id)
        return u

    async def verify(self, u):
        try:
            await self.engine.verify_upload(self.tenant, u.upload_id)
        except PermanentInputError:
            pass
        return await self.engine.metadata.get_upload(
            tenant_id=self.tenant, upload_id=u.upload_id
        )

    async def test_storage_verified_pdf_is_accepted_without_a_full_read(self):
        full_reads = self.storage_reports()
        u = await self.verify(await self.upload(self.PDF))
        self.assertEqual(u.status.value, "accepted")
        self.assertEqual(full_reads, [])

    async def test_a_storage_digest_other_than_the_declared_one_is_rejected(self):
        self.storage_reports(sha256="0" * 64)
        u = await self.verify(await self.upload(self.PDF))
        self.assertEqual(u.status.value, "failed")

    async def test_a_file_that_is_not_a_pdf_is_still_rejected(self):
        full_reads = self.storage_reports()
        u = await self.verify(await self.upload(b"PK\x03\x04" + b"x" * 500))
        self.assertEqual(u.status.value, "failed")
        self.assertEqual(full_reads, [])

    async def test_without_a_declared_sha256_every_byte_is_read(self):
        full_reads = self.storage_reports()
        u = await self.verify(await self.upload(self.PDF, declare=False))
        self.assertEqual(u.status.value, "accepted")
        self.assertEqual(len(full_reads), 1)

    async def test_text_files_are_always_read_in_full_for_utf8(self):
        full_reads = self.storage_reports()
        u = await self.verify(await self.upload(b"plain text " * 50, name="a.txt"))
        self.assertEqual(u.status.value, "accepted")
        self.assertEqual(len(full_reads), 1)

    async def test_s3_adapter_without_a_full_object_sha256_falls_back(self):
        """moto, like an upload without the header, reports no SHA-256."""
        u = await self.upload(self.PDF)
        ref = await self.engine.metadata.get_verification_ref(
            tenant_id=self.tenant, upload_id=u.upload_id
        )
        self.assertIsNone(await self.engine.objects.verified_sha256(ref))
        unpinned = replace(ref, version_id=None)
        self.assertIsNone(await self.engine.objects.verified_sha256(unpinned))
        self.assertEqual(await self.engine.objects.read_prefix(ref, 5), b"%PDF-")



# ---- bounded concurrency: chunk uploads and index/publish read-ahead ----------


class ReadAhead(unittest.IsolatedAsyncioTestCase):
    async def test_order_is_kept_and_loads_in_flight_are_bounded(self):
        import asyncio

        from rag_ingest.runtime.engine import IngestionEngine

        state = {"now": 0, "max": 0}

        def loader(n):
            async def load():
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
                await asyncio.sleep(0.001 * (7 - n % 7))  # finish out of order
                state["now"] -= 1
                return n

            return load

        got = [n async for n in IngestionEngine._read_ahead([loader(n) for n in range(30)], 3)]
        self.assertEqual(got, list(range(30)))
        self.assertLessEqual(state["max"], 3)

    async def test_stopping_early_cancels_the_loads_still_in_flight(self):
        import asyncio
        from contextlib import aclosing

        from rag_ingest.runtime.engine import IngestionEngine

        started, cancelled = [], []

        def loader(n):
            async def load():
                started.append(n)
                try:
                    await asyncio.sleep(0 if n == 0 else 10)
                except asyncio.CancelledError:
                    cancelled.append(n)
                    raise
                return n

            return load

        async with aclosing(
            IngestionEngine._read_ahead([loader(n) for n in range(10)], 2)
        ) as loads:
            self.assertEqual(await anext(loads), 0)
        self.assertEqual(sorted(cancelled), sorted(started[1:]))
        self.assertLessEqual(len(started), 3)


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class ParallelChunkUploads(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown
    ingest = segments.EmbeddingSegments.ingest
    drive = segments.EmbeddingSegments.drive
    doc = segments.EmbeddingSegments.doc
    TEXT = segments.TEXT * 6  # more batches than CHUNK_UPLOADS at batch_size=2

    def slow_chunk_writes(self, fail_key=None):
        import asyncio

        state, original = {"now": 0, "max": 0, "failed": False}, self.engine.store_bytes

        async def store_bytes(key, *args, **kwargs):
            if "/chunks-" not in key:
                return await original(key, *args, **kwargs)
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
            try:
                await asyncio.sleep(0.01)
                if fail_key and key.endswith(fail_key) and not state["failed"]:
                    state["failed"] = True
                    raise RetryableError("storage hiccup")
                return await original(key, *args, **kwargs)
            finally:
                state["now"] -= 1

        self.engine.store_bytes = store_bytes
        return state

    async def test_uploads_overlap_up_to_the_limit_and_keep_batch_order(self):
        from rag_ingest.runtime.engine import CHUNK_UPLOADS

        state = self.slow_chunk_writes()
        u = await self.ingest(self.TEXT)
        await self.drive(rounds=60)
        d = await self.doc(u)
        self.assertEqual(d.active_generation_id, d.desired_generation_id)
        inv = await self.engine.metadata.get_inventory(
            tenant_id=self.tenant, generation_id=d.active_generation_id
        )
        self.assertGreater(len(inv.batches), CHUNK_UPLOADS)
        self.assertGreater(state["max"], 1)
        self.assertLessEqual(state["max"], CHUNK_UPLOADS)
        self.assertEqual(
            [b.batch_key for b in inv.batches],
            [f"{n:06d}" for n in range(len(inv.batches))],
        )

    async def test_a_failed_upload_fails_the_attempt_and_the_retry_completes(self):
        state = self.slow_chunk_writes(fail_key="chunks-000003.json")
        u = await self.ingest(self.TEXT)
        await self.drive(rounds=60)
        self.assertTrue(state["failed"])
        d = await self.doc(u)
        self.assertEqual(d.active_generation_id, d.desired_generation_id)



# ---- outbox: one queue call per tenant pass; pools sized to concurrency -------


class _Outbox:
    """Minimal metadata double: due records per tenant and what happened to them."""

    def __init__(self, count):
        from types import SimpleNamespace
        from uuid import uuid4

        from rag_ingest.domain.events import JobAvailable

        self.tenant = uuid4()
        self.due = [
            SimpleNamespace(
                event=JobAvailable(event_id=uuid4(), tenant_id=self.tenant, job_id=uuid4()),
                claim=SimpleNamespace(claim_id=uuid4()),
                publish_attempts=0,
            )
            for _ in range(count)
        ]
        self.published, self.rescheduled = [], []

    async def tenants(self):
        return (self.tenant,)

    async def claim_next_outbox(self, **kwargs):
        return self.due.pop(0) if self.due else None

    async def mark_outbox_published(self, **kwargs):
        self.published.append(kwargs["event_id"])
        return True

    async def reschedule_outbox_send(self, **kwargs):
        self.rescheduled.append(kwargs["event_id"])


class _Queue:
    def __init__(self, refuse=()):
        self.batch_calls, self.refuse = [], set(refuse)

    async def publish_batch(self, events):
        self.batch_calls.append(len(events))
        return tuple(
            RetryableError("busy") if n in self.refuse else None
            for n in range(len(events))
        )


class BatchedOutbox(unittest.IsolatedAsyncioTestCase):
    async def publish(self, metadata, queue):
        from types import SimpleNamespace

        from rag_ingest.runtime.runners import publish_once

        return await publish_once(SimpleNamespace(metadata=metadata, queue=queue), "p")

    async def test_ten_due_records_take_one_queue_call(self):
        outbox, queue = _Outbox(13), _Queue()
        self.assertEqual(await self.publish(outbox, queue), 10)
        self.assertEqual(queue.batch_calls, [10])
        self.assertEqual(await self.publish(outbox, queue), 3)
        self.assertEqual(queue.batch_calls, [10, 3])

    async def test_a_refused_entry_is_rescheduled_and_the_rest_published(self):
        outbox, queue = _Outbox(4), _Queue(refuse={1})
        events = [r.event.event_id for r in outbox.due]
        with self.assertLogs("rag_ingest", "WARNING"):
            self.assertEqual(await self.publish(outbox, queue), 3)
        self.assertEqual(outbox.rescheduled, [events[1]])
        self.assertEqual(outbox.published, [events[0], events[2], events[3]])

    async def test_a_failed_batch_call_reschedules_every_record(self):
        from rag_ingest.contracts.errors import UncertainOutcomeError

        class Down(_Queue):
            async def publish_batch(self, events):
                raise UncertainOutcomeError("timeout after send")

        outbox = _Outbox(3)
        with self.assertLogs("rag_ingest", "WARNING"):
            self.assertEqual(await self.publish(outbox, Down()), 0)
        self.assertEqual(len(outbox.rescheduled), 3)

    async def test_queues_without_batches_still_send_each_event(self):
        class Single:
            sent = []

            async def publish(self, event):
                self.sent.append(event)

        outbox, queue = _Outbox(3), Single()
        self.assertEqual(await self.publish(outbox, queue), 3)
        self.assertEqual(len(queue.sent), 3)

    async def test_sqs_adapter_sends_a_batch_and_reports_each_entry(self):
        from moto import mock_aws

        from rag_ingest.adapters.aws.sqs import SQSQueue

        with mock_aws():
            session = boto3.Session(
                aws_access_key_id="x", aws_secret_access_key="x", region_name="us-east-1"
            )
            url = session.client("sqs").create_queue(QueueName="batch")["QueueUrl"]
            queue = SQSQueue(url, region="us-east-1", session=session)
            events = tuple(r.event for r in _Outbox(10).due)
            self.assertEqual(await queue.publish_batch(events), (None,) * 10)
            got = await queue.receive(max_messages=10, wait_seconds=0, visibility_seconds=30)
            self.assertEqual(len(got), 10)
            with self.assertRaises(ValueError):
                await queue.publish_batch(tuple(r.event for r in _Outbox(11).due))


class PoolSizing(unittest.TestCase):
    def settings(self, **kwargs):
        from rag_ingest.config import Settings
        from rag_ingest.runtime.settings import ServiceSettings

        return ServiceSettings(
            core=Settings(
                metadata_dsn="postgresql://x/y",
                s3_bucket="b",
                sqs_queue_url="https://sqs.us-east-1.amazonaws.com/1/q",
            ),
            vector_dsn="postgresql://x/y",
            bucket="b",
            **kwargs,
        )

    def test_pool_grows_with_concurrency(self):
        small = self.settings(worker_concurrency=2).metadata_pool_size()
        large = self.settings(worker_concurrency=32).metadata_pool_size()
        self.assertGreaterEqual(large, 32 * 4)
        self.assertGreater(large, small)
        self.assertEqual(self.settings(worker_concurrency=32).vector_pool_size(), 34)

    def test_a_pool_smaller_than_the_concurrency_is_refused(self):
        with self.assertRaisesRegex(ValueError, "INGEST_DB_POOL_SIZE"):
            self.settings(worker_concurrency=16, db_pool_size=10).metadata_pool_size()
        self.assertEqual(
            self.settings(worker_concurrency=1, db_pool_size=100).metadata_pool_size(), 100
        )


if __name__ == "__main__":
    unittest.main()
