"""Files left by never-accepted uploads are erased after the grace period, once.

Real PostgreSQL (opt-in DSN) and boto3 against in-process S3.
"""

import hashlib
import unittest
from dataclasses import replace
from uuid import uuid4

import sqlalchemy as sa

from rag_ingest.contracts.errors import DuplicateContentError, PermanentInputError
from rag_ingest.domain.uploads import UploadStatus as US
from rag_ingest.runtime.runners import purge_abandoned_uploads

import test_postgres_integration as integration

DSN = integration.DSN
DATA = b"Leases protect durable work. Outbox notifications survive crashes. " * 5


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class AbandonedUploadFiles(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await integration.PostgresIntegration.asyncSetUp(self)
        self.engine.settings = replace(
            self.engine.settings, abandoned_upload_grace_seconds=600
        )

    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    @property
    def s3(self):
        return self.engine.objects.client

    async def upload(self, data=DATA, *, name="a.txt", declared_size=None):
        r = await self.engine.initiate_upload(
            self.tenant,
            filename=name,
            size_bytes=declared_size or len(data),
            request_key=str(uuid4()),
        )
        u = r["upload"]
        self.s3.put_object(Bucket=u.bucket, Key=u.object_key, Body=data)
        await self.engine.complete_upload(self.tenant, u.upload_id)
        try:
            await self.engine.verify_upload(self.tenant, u.upload_id)
        except (DuplicateContentError, PermanentInputError):
            pass
        return await self.engine.metadata.get_upload(
            tenant_id=self.tenant, upload_id=u.upload_id
        )

    def stored(self, u):
        page = self.s3.list_object_versions(Bucket=u.bucket, Prefix=u.object_key)
        return [v for v in page.get("Versions", []) if v["Key"] == u.object_key]

    async def age(self, u, seconds):
        """Move the session's expiry into the past, as time passing would."""
        async with self.engine.metadata.db.engine.begin() as c:
            await c.execute(
                sa.text(
                    "UPDATE uploads SET expires_at = now() - make_interval(secs => :s)"
                    " WHERE tenant_id=:t AND id=:u"
                ),
                {"s": seconds, "t": self.tenant, "u": u.upload_id},
            )

    async def test_a_refused_duplicate_file_is_erased_and_the_original_kept(self):
        original = await self.upload()
        copy = await self.upload(name="copy.txt")
        self.assertEqual(copy.status, US.FAILED)
        self.assertTrue(self.stored(copy), "the refused copy's bytes reached S3")
        await self.age(copy, 3600)
        await self.age(original, 3600)
        with self.assertLogs("rag_ingest", "INFO") as logs:
            self.assertEqual(await purge_abandoned_uploads(self.engine), 1)
        self.assertTrue(any("upload_files_removed" in m for m in logs.output))
        self.assertEqual(self.stored(copy), [])
        self.assertTrue(self.stored(original), "an accepted upload is never touched")
        self.assertEqual(await purge_abandoned_uploads(self.engine), 0, "done once")

    async def test_nothing_is_erased_inside_the_grace_period(self):
        await self.upload()
        copy = await self.upload(name="copy.txt")
        await self.age(copy, 300)  # expired, but less than the 600 s grace
        self.assertEqual(await purge_abandoned_uploads(self.engine), 0)
        self.assertTrue(self.stored(copy))

    async def test_a_failed_verification_file_is_erased(self):
        bad = await self.upload(DATA, declared_size=len(DATA) + 7)  # size mismatch
        self.assertEqual(bad.status, US.FAILED)
        await self.age(bad, 3600)
        self.assertEqual(await purge_abandoned_uploads(self.engine), 1)
        self.assertEqual(self.stored(bad), [])

    async def test_an_expired_multipart_upload_has_its_parts_aborted(self):
        self.engine.settings = replace(
            self.engine.settings, multipart_threshold=1, multipart_part_bytes=5 * 1024**2
        )
        data = b"x" * (6 * 1024**2)
        r = await self.engine.initiate_upload(
            self.tenant,
            filename="big.txt",
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            request_key=str(uuid4()),
        )
        u = r["upload"]
        plan = r["plan"]
        self.s3.upload_part(
            Bucket=u.bucket,
            Key=u.object_key,
            UploadId=plan.provider_upload_id,
            PartNumber=1,
            Body=data[: 5 * 1024**2],
        )
        await self.age(u, 3600)
        await self.engine.metadata.reconcile(self.tenant, limit=100)  # marks it expired
        u = await self.engine.metadata.get_upload(
            tenant_id=self.tenant, upload_id=u.upload_id
        )
        self.assertEqual(u.status, US.EXPIRED)
        await self.age(u, 3600)  # saving the expired record rewrote the column
        self.assertEqual(await purge_abandoned_uploads(self.engine), 1)
        open_uploads = self.s3.list_multipart_uploads(Bucket=u.bucket).get("Uploads", [])
        self.assertNotIn(plan.provider_upload_id, [m["UploadId"] for m in open_uploads])

    async def test_uploads_still_in_progress_are_never_touched(self):
        r = await self.engine.initiate_upload(
            self.tenant, filename="a.txt", size_bytes=len(DATA), request_key=str(uuid4())
        )
        u = r["upload"]
        self.s3.put_object(Bucket=u.bucket, Key=u.object_key, Body=DATA)
        await self.age(u, 3600)  # expired by the clock, but not yet marked expired
        self.assertEqual(await purge_abandoned_uploads(self.engine), 0)
        self.assertTrue(self.stored(u))


if __name__ == "__main__":
    unittest.main()
