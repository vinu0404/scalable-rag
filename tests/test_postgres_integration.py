"""Opt-in real PostgreSQL/pgvector tests. Uses only a disposable *_test database.

Tables in the supplied test database are reset for each test. AWS is emulated
in-process; embeddings are explicit deterministic fixtures. No model is billed.
"""

from rag_ingest.runtime.search import search as run_search
import asyncio
import hashlib
import os
import unittest
from uuid import uuid4
import boto3
import sqlalchemy as sa
from moto import mock_aws
from rag_ingest.config import Settings
from rag_ingest.runtime.settings import ServiceSettings
from rag_ingest.runtime.factory import build_engine
from rag_ingest.testing.embedder import DeterministicEmbedder
from rag_ingest.runtime.runners import publish_once, scheduler_once, worker_once
from rag_ingest.adapters.postgres.connection import Database
from rag_ingest.adapters.postgres.metadata import PostgresMetadataStore
from rag_ingest.adapters.postgres.vectors import PgVectorStore
from rag_ingest.adapters.postgres import schema
from rag_ingest.adapters.aws.s3 import S3ObjectStore
from rag_ingest.adapters.aws.sqs import SQSQueue
from rag_ingest.contracts.errors import LeaseLostError

DSN = os.environ.get("INGEST_TEST_METADATA_DSN")


@unittest.skipUnless(
    DSN, "Set INGEST_TEST_METADATA_DSN to a disposable PostgreSQL *_test database"
)
class PostgresIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not (sa.engine.make_url(DSN).database or "").endswith("_test"):
            raise ValueError(
                "Integration test DSN must name a disposable database ending in _test"
            )
        self.mock = mock_aws()
        self.mock.start()
        self.tenant = uuid4()
        session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )
        objects = S3ObjectStore(region="us-east-1", session=session)
        objects.client.create_bucket(Bucket="pipeline-test")
        objects.client.put_bucket_versioning(
            Bucket="pipeline-test", VersioningConfiguration={"Status": "Enabled"}
        )
        url = session.client("sqs").create_queue(QueueName="pipeline-test")["QueueUrl"]
        metadata = PostgresMetadataStore(Database(DSN))
        async with metadata.db.engine.begin() as connection:
            await connection.execute(sa.text("DROP TABLE IF EXISTS rag_vectors"))
            await connection.run_sync(schema.metadata.drop_all)
            await connection.run_sync(schema.metadata.create_all)
        vectors = PgVectorStore(Database(DSN), dimensions=8)
        await vectors.provision()
        settings = ServiceSettings(
            core=Settings(
                metadata_dsn=DSN, s3_bucket="pipeline-test", sqs_queue_url=url
            ),
            vector_dsn=DSN,
            bucket="pipeline-test",
            embedding_driver="deterministic",
            embedding_dimensions=8,
            chunk_size=80,
            chunk_overlap=10,
            batch_size=2,
        )
        self.engine = build_engine(
            settings,
            metadata=metadata,
            objects=objects,
            queue=SQSQueue(url, region="us-east-1", session=session),
            vectors=vectors,
            embedder=DeterministicEmbedder(),
        )

    async def asyncTearDown(self):
        if hasattr(self, "engine"):
            await self.engine.aclose()
        self.mock.stop()

    async def accepted_upload(self):
        data = (
            b"Leases protect durable work. Outbox notifications survive crashes. " * 5
        )
        result = await self.engine.initiate_upload(
            self.tenant,
            filename="example.txt",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            request_key=str(uuid4()),
        )
        upload = result["upload"]
        self.engine.objects.client.put_object(
            Bucket=upload.bucket, Key=upload.object_key, Body=data
        )
        await self.engine.complete_upload(self.tenant, upload.upload_id)
        await self.engine.verify_upload(self.tenant, upload.upload_id)
        # The scheduler admits waiting generations; do it here so tests can claim.
        await self.engine.metadata.admit_generations(
            tenant_id=self.tenant, max_active=16
        )
        return upload

    async def test_pipeline_publishes_and_deletion_hides_search(self):
        upload = await self.accepted_upload()
        for _ in range(60):
            await scheduler_once(self.engine)
            await publish_once(self.engine, "integration-publisher")
            for _ in range(8):
                if not await worker_once(
                    self.engine, "integration-worker", wait_seconds=0
                ):
                    break
            doc = await self.engine.metadata.get_document(
                tenant_id=self.tenant, document_id=upload.document_id
            )
            if doc.active_generation_id:
                break
        self.assertEqual(doc.active_generation_id, doc.desired_generation_id)
        self.assertIsNotNone(doc.active_generation_id)
        jobs = await self.engine.metadata.list_generation_jobs(
            tenant_id=self.tenant, generation_id=doc.active_generation_id
        )
        self.assertTrue(
            all(j.status.value == "succeeded" for j in jobs),
            [(j.stage.value, j.status.value) for j in jobs],
        )
        rows = await run_search(self.engine, self.tenant, "durable work", limit=10)
        self.assertGreater(len(rows), 0)
        self.assertEqual(
            await run_search(self.engine, uuid4(), "durable work", limit=10), ()
        )
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=doc.document_id, deletion_id=uuid4()
        )
        self.assertEqual(
            await run_search(self.engine, self.tenant, "durable work", limit=10), ()
        )

    async def test_only_one_concurrent_claim_and_stale_lease_cannot_fail_job(self):
        upload = await self.accepted_upload()
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=upload.document_id
        )
        job = (
            await self.engine.metadata.list_generation_jobs(
                tenant_id=self.tenant, generation_id=doc.desired_generation_id
            )
        )[0]
        claims = await asyncio.gather(
            *[
                self.engine.metadata.claim_job(
                    tenant_id=self.tenant,
                    job_id=job.job_id,
                    worker_id="worker-" + str(i),
                    lease_id=uuid4(),
                    lease_seconds=120,
                )
                for i in range(4)
            ]
        )
        winner = [j for j in claims if j]
        self.assertEqual(len(winner), 1)
        claimed = winner[0]
        with self.assertRaises(LeaseLostError):
            await self.engine.metadata.fail_job(
                tenant_id=self.tenant,
                job_id=job.job_id,
                worker_id=claimed.lease.worker_id,
                lease_id=uuid4(),
                failure_id=uuid4(),
                failure_code="wrong_lease",
            )
        renewed = await self.engine.metadata.renew_job_lease(
            tenant_id=self.tenant,
            job_id=job.job_id,
            worker_id=claimed.lease.worker_id,
            lease_id=claimed.lease.lease_id,
            lease_seconds=120,
        )
        self.assertEqual(renewed.attempt_deadline_at, claimed.lease.attempt_deadline_at)
        self.assertLessEqual(renewed.expires_at, claimed.lease.attempt_deadline_at)
