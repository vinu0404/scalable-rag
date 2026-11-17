"""Duplicate content: refused before upload when the hash is declared, else at acceptance.

Real PostgreSQL (opt-in DSN) and boto3 against in-process S3. A live
document is one that is not deleted and whose published or current
non-failed build uses the same bytes. Checks run under the tenant lock.
"""

import asyncio
import hashlib
import unittest
from uuid import uuid4

import httpx
import sqlalchemy as sa

from rag_ingest.api.app import create_app
from rag_ingest.contracts.errors import DuplicateContentError
from rag_ingest.domain.uploads import UploadStatus as US

import test_postgres_integration as integration

DSN = integration.DSN
DATA = b"Leases protect durable work. Outbox notifications survive crashes. " * 5


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class Duplicates(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    async def start(
        self, data=DATA, *, name="a.txt", declare=True, allow=False, tenant=None
    ):
        r = await self.engine.initiate_upload(
            tenant or self.tenant,
            filename=name,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest() if declare else None,
            request_key=str(uuid4()),
            allow_duplicate=allow,
        )
        return r["upload"]

    async def finish(self, u, data=DATA, tenant=None):
        """Store the bytes, complete and verify; return the upload's final state."""
        tenant = tenant or self.tenant
        self.engine.objects.client.put_object(
            Bucket=u.bucket, Key=u.object_key, Body=data
        )
        await self.engine.complete_upload(tenant, u.upload_id)
        try:
            await self.engine.verify_upload(tenant, u.upload_id)
        except DuplicateContentError:
            pass
        return await self.engine.metadata.get_upload(
            tenant_id=tenant, upload_id=u.upload_id
        )

    async def ingest(self, data=DATA, **kwargs):
        u = await self.start(data, **kwargs)
        return await self.finish(u, data, kwargs.get("tenant"))

    async def count(self, table):
        async with self.engine.metadata.db.engine.begin() as c:
            return (
                await c.execute(
                    sa.text(f"SELECT count(*) FROM {table} WHERE tenant_id=:t"),
                    {"t": self.tenant},
                )
            ).scalar_one()

    async def sql(self, statement, **params):
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(sa.text(statement), {"t": self.tenant, **params})

    # ---- refused before anything is uploaded ----------------------------------

    async def test_a_declared_duplicate_is_refused_before_any_upload(self):
        original = await self.ingest(name="report.txt")
        self.assertEqual(original.status, US.ACCEPTED)
        documents, uploads = await self.count("documents"), await self.count("uploads")
        with self.assertRaises(DuplicateContentError) as refused:
            await self.start(name="report (copy).txt")
        self.assertEqual(refused.exception.document_id, original.document_id)
        self.assertEqual(refused.exception.filename, "report.txt")
        self.assertEqual(await self.count("documents"), documents, "no new document")
        self.assertEqual(await self.count("uploads"), uploads, "no upload, no grant")

    async def test_replaying_the_original_request_is_not_a_duplicate(self):
        key = str(uuid4())
        args = dict(
            filename="a.txt",
            size_bytes=len(DATA),
            sha256=hashlib.sha256(DATA).hexdigest(),
            request_key=key,
        )
        first = (await self.engine.initiate_upload(self.tenant, **args))["upload"]
        await self.finish(first)
        again = (await self.engine.initiate_upload(self.tenant, **args))["upload"]
        self.assertEqual(again.upload_id, first.upload_id)

    # ---- caught at acceptance when no hash was declared ------------------------

    async def test_an_undeclared_duplicate_is_refused_when_accepted(self):
        original = await self.ingest()
        copy = await self.ingest(name="b.txt", declare=False)
        self.assertEqual(copy.status, US.FAILED)
        self.assertEqual(copy.duplicate_of, original.document_id)
        self.assertEqual(await self.count("generations"), 1, "nothing processed")
        replay = await self.engine.metadata.accept_verified_upload(
            tenant_id=self.tenant,
            upload_id=copy.upload_id,
            verified_ref=None,
            artifact_candidate=None,
            source=None,
            generation=None,
            first_job=None,
            outbox=None,
        )
        self.assertEqual(replay.duplicate_of, original.document_id, "same answer")

    async def test_two_identical_uploads_at_once_keep_exactly_one(self):
        first = await self.start(name="one.txt")
        second = await self.start(name="two.txt")  # neither is live yet
        results = await asyncio.gather(self.finish(first), self.finish(second))
        statuses = sorted(r.status.value for r in results)
        self.assertEqual(statuses, ["accepted", "failed"])
        winner = next(r for r in results if r.status == US.ACCEPTED)
        loser = next(r for r in results if r.status == US.FAILED)
        self.assertEqual(loser.duplicate_of, winner.document_id)

    # ---- what does not count as a duplicate -----------------------------------

    async def test_allow_duplicate_keeps_a_second_copy(self):
        await self.ingest()
        copy = await self.ingest(name="b.txt", allow=True)
        self.assertEqual(copy.status, US.ACCEPTED)
        self.assertEqual(await self.count("generations"), 2)

    async def test_a_failed_build_does_not_block_a_new_upload(self):
        await self.ingest()
        await self.sql("UPDATE generations SET status='failed' WHERE tenant_id=:t")
        self.assertEqual((await self.ingest(name="retry.txt")).status, US.ACCEPTED)

    async def test_a_deleted_document_does_not_block_a_new_upload(self):
        await self.ingest()
        await self.sql("UPDATE documents SET deleted_at=now() WHERE tenant_id=:t")
        self.assertEqual((await self.ingest(name="again.txt")).status, US.ACCEPTED)

    async def test_other_tenants_content_is_never_a_duplicate(self):
        await self.ingest()
        other = uuid4()
        u = await self.ingest(name="theirs.txt", tenant=other)
        self.assertEqual(u.status, US.ACCEPTED)

    async def test_different_bytes_are_not_duplicates(self):
        await self.ingest()
        self.assertEqual(
            (await self.ingest(DATA + b" one more line", name="c.txt")).status,
            US.ACCEPTED,
        )

    # ---- the HTTP contract ------------------------------------------------------

    async def test_api_answers_409_with_the_existing_document(self):
        original = await self.ingest(name="report.txt")
        app = create_app(self.engine, authenticate=lambda token: self.tenant)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post(
                "/v1/uploads",
                headers={"Authorization": "Bearer x", "Idempotency-Key": "k1"},
                json={
                    "filename": "copy.txt",
                    "size_bytes": len(DATA),
                    "sha256": hashlib.sha256(DATA).hexdigest(),
                },
            )
            self.assertEqual(r.status_code, 409)
            self.assertEqual(
                r.json()["duplicate_of"],
                {"document_id": str(original.document_id), "filename": "report.txt"},
            )
            kept = await c.post(
                "/v1/uploads",
                headers={"Authorization": "Bearer x", "Idempotency-Key": "k2"},
                json={
                    "filename": "copy.txt",
                    "size_bytes": len(DATA),
                    "sha256": hashlib.sha256(DATA).hexdigest(),
                    "allow_duplicate": True,
                },
            )
            self.assertEqual(kept.status_code, 201)
            self.assertIsNone(kept.json()["duplicate_of"])


if __name__ == "__main__":
    unittest.main()
