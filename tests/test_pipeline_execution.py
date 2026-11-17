"""Exercise processing stages with real storage/parser and explicit test doubles.

This checks orchestration and artifact formats, not PostgreSQL transaction
semantics. The separate opt-in PostgreSQL tests cover that integration path.
"""

import hashlib
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4
import boto3
from moto import mock_aws
from rag_ingest.config import Settings
from rag_ingest.runtime.settings import ServiceSettings
from rag_ingest.runtime.engine import IngestionEngine
from rag_ingest.adapters.aws.s3 import S3ObjectStore
from rag_ingest.adapters.processing.text import TextExtractor, CharacterChunker
from rag_ingest.testing.embedder import DeterministicEmbedder
from rag_ingest.domain.documents import SourceVersion, ProcessingGeneration
from rag_ingest.domain.jobs import Job, JobLease
from rag_ingest.domain.enums import ProcessingStage as ST, JobStatus as JS
from rag_ingest.domain.policy import ExecutionBudget, ExecutionPolicy
from rag_ingest.contracts.errors import RetryableError
from rag_ingest.contracts.vectors import VectorWriteResult, VectorCheck, VectorReadiness


class MemoryMetadata:
    """Test double for stage traversal; deliberately does not simulate SQL safety."""

    def __init__(self, source, generation, pipeline):
        self.source = source
        self.generation = generation
        self.pipeline = pipeline
        self.jobs = {}
        self.checkpoints = {}
        self.inventory = None
        self.inventory_ref = None
        self.published = False

    async def now(self, *args):
        return datetime.now(timezone.utc)

    async def get_generation(self, **kw):
        return self.generation

    async def get_source_version(self, **kw):
        return self.source

    async def get_generation_pipeline(self, **kw):
        return self.pipeline

    async def get_checkpoint(self, **kw):
        return self.checkpoints.get(kw["job_id"])

    async def save_checkpoint(self, **kw):
        cp = kw["checkpoint"]
        self.checkpoints[cp.job_id] = cp
        return cp

    async def complete_stage(self, **kw):
        j = self.jobs[kw["job_id"]]
        self.jobs[j.job_id] = replace(j, status=JS.SUCCEEDED, lease=None)

    async def list_generation_jobs(self, **kw):
        return tuple(
            j for j in self.jobs.values() if j.stage == kw.get("stage", j.stage)
        )

    async def seal_generation_inventory(self, **kw):
        self.inventory = kw["inventory"]
        self.inventory_ref = kw["inventory_ref"]
        return self.inventory_ref

    async def get_generation_inventory_ref(self, **kw):
        return self.inventory_ref

    async def get_batch_plan(self, **kw):
        return self.inventory.plan_for(kw["batch_key"])

    async def put_search_chunks(self, *, tenant_id, chunks):
        store = self.__dict__.setdefault("search_chunks", {})
        for c in chunks:
            assert store.setdefault(c.vector_id, c.text) == c.text
        return None

    async def search_chunk_digests(self, *, tenant_id, generation_id, batches):
        from rag_ingest.domain.search import chunk_digest

        store = self.__dict__.get("search_chunks", {})
        return tuple(
            chunk_digest((i, store[i]) for i in batch if i in store)
            if any(i in store for i in batch)
            else None
            for batch in batches
        )

    async def publish_generation(self, **kw):
        assert kw["verified_chunk_count"] == self.inventory.chunk_count
        assert kw["verified_chunk_ids_sha256"] == self.inventory.vector_ids_sha256
        self.published = True

    async def aclose(self):
        pass


class MemoryVectors:
    """Store exact test payloads so publication checks can detect missing writes."""

    target_id = "test-target"

    def __init__(self):
        self.items = {}

    async def upsert_batch(self, scope, *, items):
        for i in items:
            scope.validate_item(i)
            self.items[i.chunk.vector_id] = i
        return tuple(VectorWriteResult(vector_id=i.chunk.vector_id) for i in items)

    async def verify_batch(self, scope, *, expected):
        return tuple(
            VectorCheck(
                vector_id=i.chunk.vector_id,
                readiness=VectorReadiness.READY
                if self.items.get(i.chunk.vector_id) == i
                else VectorReadiness.MISSING,
            )
            for i in expected
        )

    async def aclose(self):
        pass


class PipelineExecution(unittest.IsolatedAsyncioTestCase):
    async def test_saved_stages_recover_and_publish_complete_vector_set(self):
        with mock_aws():
            session = boto3.Session(
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
                region_name="us-east-1",
            )
            objects = S3ObjectStore(region="us-east-1", session=session)
            objects.client.create_bucket(Bucket="stage-test")
            objects.client.put_bucket_versioning(
                Bucket="stage-test", VersioningConfiguration={"Status": "Enabled"}
            )
            data = b"Checkpoint recovery keeps accepted work durable. " * 12
            response = objects.client.put_object(
                Bucket="stage-test", Key="source", Body=data
            )
            now = datetime.now(timezone.utc)
            tenant, doc, src, gen = [uuid4() for _ in range(4)]
            source = SourceVersion(
                tenant_id=tenant,
                document_id=doc,
                source_version_id=src,
                artifact_id=uuid4(),
                bucket="stage-test",
                object_key="source",
                object_version_id=response["VersionId"],
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                created_at=now,
            )
            settings = ServiceSettings(
                core=Settings(
                    metadata_dsn="postgresql://test:test@localhost/test",
                    s3_bucket="stage-test",
                    sqs_queue_url="https://sqs.us-east-1.amazonaws.com/000000000000/test",
                ),
                vector_dsn="postgresql://test:test@localhost/test",
                bucket="stage-test",
                embedding_driver="deterministic",
                embedding_dimensions=8,
                chunk_size=100,
                chunk_overlap=10,
                batch_size=2,
            )
            pipeline = settings.pipeline()
            generation = ProcessingGeneration(
                tenant_id=tenant,
                document_id=doc,
                source_version_id=src,
                generation_id=gen,
                pipeline_fingerprint=pipeline.fingerprint,
                created_at=now,
            )
            metadata = MemoryMetadata(source, generation, pipeline)
            vectors = MemoryVectors()
            engine = IngestionEngine(
                metadata=metadata,
                objects=objects,
                uploads=objects,
                queue=SimpleNamespace(aclose=self.noop),
                extractor=TextExtractor(),
                chunker=CharacterChunker(),
                embedder=DeterministicEmbedder(),
                vectors=vectors,
                settings=settings,
            )

            async def execute(stage, shard="0", recover=False):
                budget = ExecutionBudget(
                    policy=ExecutionPolicy(),
                    starts_at=now,
                    deadline_at=now + timedelta(minutes=30),
                    generation_deadline_at=now + timedelta(hours=1),
                )
                lease = JobLease(
                    worker_id="test-worker",
                    lease_id=uuid4(),
                    acquired_at=now,
                    expires_at=now + timedelta(seconds=120),
                    attempt_deadline_at=now + timedelta(seconds=300),
                )
                job = Job(
                    tenant_id=tenant,
                    document_id=doc,
                    generation_id=gen,
                    job_id=uuid4(),
                    stage=stage,
                    shard_key=shard,
                    status=JS.RUNNING,
                    created_at=now,
                    budget=budget,
                    attempt=1,
                    lease=lease,
                )
                metadata.jobs[job.job_id] = job
                await engine.execute_job(job)
                if recover:
                    # Re-enter the same stage with a new lease after a hypothetical
                    # crash between checkpoint and completion. No re-extraction.
                    metadata.jobs[job.job_id] = job
                    await engine.execute_job(
                        replace(job, lease=replace(lease, lease_id=uuid4()))
                    )

            try:
                await execute(ST.EXTRACT, recover=True)
                await execute(ST.CHUNK)
                for batch in metadata.inventory.batches:
                    await execute(ST.EMBED, batch.batch_key)
                await execute(ST.INDEX)
                key = next(iter(vectors.items))
                saved = vectors.items.pop(key)
                with self.assertRaises(RetryableError):
                    await execute(ST.PUBLISH)
                self.assertFalse(metadata.published)
                vectors.items[key] = saved
                await execute(ST.PUBLISH)
                self.assertTrue(metadata.published)
                self.assertEqual(len(vectors.items), metadata.inventory.chunk_count)
                self.assertGreater(len(metadata.inventory.batches), 1)
            finally:
                await engine.aclose()

    async def noop(self):
        pass
