"""Embedding segments end to end: written once, cache and committed output.

Real PostgreSQL metadata (opt-in DSN) and real boto3 against in-process S3.
Each test targets one guarantee, including the crash windows of the stage.
"""

import json
import logging
import unittest
from collections import Counter
from uuid import uuid4

import sqlalchemy as sa

from rag_ingest.adapters.postgres.codec import decode
from rag_ingest.contracts.embedding import EmbeddingBatchResult, EmbeddingFailure
from rag_ingest.contracts.errors import RetryableError
from rag_ingest.domain.enums import JobStatus as JS, ProcessingStage as ST
from rag_ingest.runtime.embedding_segments import encode_segment
from rag_ingest.runtime.runners import publish_once, scheduler_once, worker_once

from test_postgres_integration import DSN, PostgresIntegration

TEXT = b"Leases protect durable work. Outbox notifications survive crashes. " * 5


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class EmbeddingSegments(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = PostgresIntegration.asyncSetUp
    asyncTearDown = PostgresIntegration.asyncTearDown

    # ---- helpers ------------------------------------------------------------

    def count_cache_calls(self):
        calls = Counter()
        objects = self.engine.objects
        stat, put = objects.stat, objects.put_immutable

        async def counted_stat(ref):
            if "/embedding-cache/" in ref.key:
                calls["stat"] += 1
            return await stat(ref)

        async def counted_put(**kwargs):
            calls[
                "put:"
                + ("segment" if "/embedding-cache/" in kwargs["key"] else "other")
            ] += 1
            if kwargs["key"].endswith("/embeddings.json"):
                calls["embeddings.json"] += 1
            return await put(**kwargs)

        objects.stat, objects.put_immutable = counted_stat, counted_put
        return calls

    def wrap_embedder(self, fail_first_call_for=None):
        embedder, seen, state = self.engine.embedder, [], {"failed": False}
        original = embedder.embed_batch

        async def embed_batch(chunks, **kwargs):
            seen.extend(c.ordinal for c in chunks)
            result = await original(chunks, **kwargs)
            if fail_first_call_for is not None and not state["failed"]:
                state["failed"] = True
                return EmbeddingBatchResult(
                    items=tuple(
                        EmbeddingFailure(chunk=i.chunk, error=RetryableError("busy"))
                        if i.chunk.ordinal in fail_first_call_for
                        else i
                        for i in result.items
                    )
                )
            return result

        embedder.embed_batch = embed_batch
        return seen

    async def ingest(self, data=TEXT):
        r = await self.engine.initiate_upload(
            self.tenant,
            filename="doc.txt",
            size_bytes=len(data),
            request_key=str(uuid4()),
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
        return u

    async def drive(self, rounds=25):
        for _ in range(rounds):
            await scheduler_once(self.engine)
            await publish_once(self.engine, "p")
            while await worker_once(self.engine, "w", wait_seconds=0):
                pass
            async with self.engine.metadata.db.engine.begin() as c:
                await c.execute(
                    sa.text(
                        "UPDATE jobs SET due_at = now() WHERE status = 'retry_wait'"
                    )
                )

    async def doc(self, u):
        return await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )

    async def embed_job(self, u, batch="000000"):
        d = await self.doc(u)
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant, generation_id=d.desired_generation_id, stage=ST.EMBED
        )
        return next((j for j in jobs if j.shard_key == batch), None)

    async def committed(self, u, batch="000000"):
        """The checkpointed manifest outputs of one embedding batch."""
        return (
            await self.engine.checkpoint_manifest(await self.embed_job(u, batch))
        ).outputs

    def segment_keys(self, generation, batch="000000"):
        prefix = (
            f"tenants/{self.tenant}/generations/{generation}/embedding-cache/{batch}/"
        )
        page = self.engine.objects.client.list_object_versions(
            Bucket=self.engine.settings.bucket, Prefix=prefix
        )
        return prefix, sorted(
            v["Key"].rsplit("/", 1)[1] for v in page.get("Versions", [])
        )

    async def until_embed_status(self, u, status, rounds=15):
        for _ in range(rounds):
            await self.drive(rounds=1)
            job = await self.embed_job(u)
            if job and job.status == status:
                return job
        self.fail(f"batch never reached {status}")

    async def pause_before_embedding(self, u):
        """Stop the batch before it embeds anything: attempt 1 fails up front."""
        original, state = self.engine._embed, {"paused": False}

        async def paused(job, *args):
            if job.shard_key == "000000" and not state["paused"]:
                state["paused"] = True
                raise RetryableError("paused by the test before any work")
            return await original(job, *args)

        self.engine._embed = paused
        return await self.until_embed_status(u, JS.RETRY_WAIT)

    async def batch_chunks(self, u, batch_index=0):
        d = await self.doc(u)
        inv = await self.engine.metadata.get_inventory(
            tenant_id=self.tenant, generation_id=d.desired_generation_id
        )
        generation = await self.engine.metadata.get_generation(
            tenant_id=self.tenant, generation_id=d.desired_generation_id
        )
        source = await self.engine.metadata.get_source_version(
            tenant_id=self.tenant, source_version_id=generation.source_version_id
        )
        entry = inv.batches[batch_index]
        job = await self.embed_job(u, entry.batch_key)
        manifest = await self.engine._batch_manifest(
            SimpleJob(job), source, self.engine.settings.pipeline(), entry
        )
        chunks = decode(
            json.loads(
                await self.engine.read_output(self.engine._output(manifest, "chunks"))
            )
        )
        return chunks, inv

    async def embed(self, chunks, inv):
        from rag_ingest.testing.embedder import DeterministicEmbedder
        from rag_ingest.contracts.embedding import EmbeddingLimits

        result = await DeterministicEmbedder().embed_batch(
            tuple(chunks),
            pipeline=self.engine.settings.pipeline(),
            dimensions=inv.dimensions,
            limits=EmbeddingLimits(
                max_items=512,
                max_input_characters=10_000,
                max_batch_tokens=100_000,
                timeout_seconds=10,
            ),
        )
        return list(result.items)

    # ---- the normal path: one write, committed by pinned version --------------

    async def test_vectors_are_written_once_and_committed_by_pinned_version(self):
        calls = self.count_cache_calls()
        u = await self.ingest()
        await self.drive()
        d = await self.doc(u)
        self.assertIsNotNone(d.active_generation_id)
        inv = await self.engine.metadata.get_inventory(
            tenant_id=self.tenant, generation_id=d.active_generation_id
        )
        batches = len(inv.batches)
        self.assertEqual(calls["stat"], 0, "a first attempt has no cache to check")
        self.assertEqual(calls["put:segment"], batches, "one vector write per batch")
        self.assertEqual(calls["embeddings.json"], 0, "no second copy of the vectors")
        (output,) = await self.committed(u)
        self.assertEqual(output.role, "embedding_segment_000")
        head = self.engine.objects.client.head_object(
            Bucket=output.bucket,
            Key=output.object_key,
            VersionId=output.object_version_id,
        )
        self.assertEqual(head["ContentLength"], output.size_bytes)

    # ---- partial failure and every crash window ---------------------------------

    async def test_retry_embeds_only_failed_chunks_and_commits_both_segments(self):
        seen = self.wrap_embedder(fail_first_call_for={1})
        u = await self.ingest()
        await self.drive()
        d = await self.doc(u)
        self.assertIsNotNone(d.active_generation_id)
        counts = Counter(seen)
        self.assertEqual(
            {o for o, n in counts.items() if n > 1}, {1}, "only chunk 1 re-billed"
        )
        roles = [o.role for o in await self.committed(u)]
        self.assertEqual(
            sorted(roles), ["embedding_segment_000", "embedding_segment_001"]
        )

    async def test_crash_after_segment_before_checkpoint_rebills_nothing(self):
        seen = self.wrap_embedder()
        original, crashed = self.engine._save_manifest, {"done": False}

        async def crash_once(job, *args, **kwargs):
            if job.stage == ST.EMBED and not crashed["done"]:
                crashed["done"] = True
                raise RuntimeError("simulated crash after the segment was written")
            return await original(job, *args, **kwargs)

        self.engine._save_manifest = crash_once
        u = await self.ingest()
        await self.drive()
        d = await self.doc(u)
        self.assertTrue(crashed["done"])
        self.assertIsNotNone(d.active_generation_id)
        self.assertEqual(max(Counter(seen).values()), 1, "no chunk embedded twice")
        _, keys = self.segment_keys(d.active_generation_id)
        self.assertEqual(keys, ["000.seg"], "the retry reused the segment, wrote none")
        self.assertEqual(
            [o.role for o in await self.committed(u)], ["embedding_segment_000"]
        )

    async def test_failed_attempt_saves_successes_before_raising(self):
        seen = self.wrap_embedder(fail_first_call_for={1})
        u = await self.ingest()
        job = await self.until_embed_status(u, JS.RETRY_WAIT)
        _, keys = self.segment_keys((await self.doc(u)).desired_generation_id)
        self.assertEqual(keys, ["000.seg"], "successes were saved before the failure")
        self.assertEqual(job.attempt, 1)
        self.assertTrue(seen)

    # ---- concurrency and damage ----------------------------------------------------

    async def test_segment_from_another_attempt_is_reused_not_overwritten(self):
        seen = self.wrap_embedder()
        u = await self.ingest()
        await self.pause_before_embedding(u)
        chunks, inv = await self.batch_chunks(u)
        d = await self.doc(u)
        prefix, _ = self.segment_keys(d.desired_generation_id)
        other = encode_segment(
            await self.embed(chunks, inv),
            recipe=self.engine.settings.pipeline().fingerprint,
        )
        await self.engine.store_bytes(
            prefix + "000.seg", other, "x", "embedding-segment"
        )
        seen.clear()
        await self.drive()
        self.assertIsNotNone((await self.doc(u)).active_generation_id)
        ordinals = {c.ordinal for c in chunks}
        self.assertFalse(
            ordinals & set(seen), "the other attempt's vectors were reused"
        )
        self.assertEqual(self.segment_keys(d.desired_generation_id)[1][0], "000.seg")

    async def test_damaged_segment_is_skipped_and_replaced_by_a_new_number(self):
        u = await self.ingest()
        await self.pause_before_embedding(u)
        chunks, inv = await self.batch_chunks(u)
        d = await self.doc(u)
        prefix, _ = self.segment_keys(d.desired_generation_id)
        good = encode_segment(
            await self.embed(chunks, inv),
            recipe=self.engine.settings.pipeline().fingerprint,
        )
        damaged = bytearray(good)
        damaged[len(damaged) // 2] ^= 0xFF
        await self.engine.store_bytes(
            prefix + "000.seg", bytes(damaged), "x", "embedding-segment"
        )
        with self.assertLogs("rag_ingest", level=logging.WARNING) as logs:
            await self.drive()
        self.assertIn("embedding_cache_entry_ignored", "".join(logs.output))
        self.assertIsNotNone((await self.doc(u)).active_generation_id)
        self.assertEqual(
            self.segment_keys(d.desired_generation_id)[1], ["000.seg", "001.seg"]
        )
        self.assertEqual(
            [o.role for o in await self.committed(u)], ["embedding_segment_001"]
        )

    async def test_later_object_versions_cannot_change_committed_vectors(self):
        u = await self.ingest()
        await self.until_embed_status(u, JS.SUCCEEDED)
        (output,) = await self.committed(u)
        # Bypass create-only and write a NEW version under the committed key.
        self.engine.objects.client.put_object(
            Bucket=output.bucket, Key=output.object_key, Body=b"overwritten"
        )
        await self.drive()  # index and publish read the PINNED version
        d = await self.doc(u)
        self.assertIsNotNone(d.active_generation_id)



class SimpleJob:
    """The fields _batch_manifest reads from a job, for building test fixtures."""

    def __init__(self, job):
        self.tenant_id, self.document_id, self.generation_id = (
            job.tenant_id,
            job.document_id,
            job.generation_id,
        )
        self.lease = type("Lease", (), {"lease_id": uuid4()})()


if __name__ == "__main__":
    unittest.main()
