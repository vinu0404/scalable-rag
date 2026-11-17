"""Exercise real boto3 adapter requests against an in-process AWS emulator."""

import hashlib
import unittest
from urllib.parse import urlsplit, parse_qs
from uuid import uuid4
import boto3
from moto import mock_aws
from rag_ingest.adapters.aws.s3 import S3ObjectStore
from rag_ingest.adapters.aws.sqs import SQSQueue
from rag_ingest.contracts.storage import ObjectRef
from rag_ingest.contracts.uploads import UploadedPart
from rag_ingest.contracts.errors import PermanentInputError, OperatorActionRequiredError
from rag_ingest.domain.events import JobAvailable


class AWSAdapters(unittest.IsolatedAsyncioTestCase):
    """Keep SDK interception active while asynchronous worker threads run."""

    async def asyncSetUp(self):
        self.mock = mock_aws()
        self.mock.start()
        session = boto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )
        self.store = S3ObjectStore(region="us-east-1", session=session)
        self.bucket = "ingestion-adapter-tests"
        self.store.client.create_bucket(Bucket=self.bucket)
        self.store.client.put_bucket_versioning(
            Bucket=self.bucket, VersioningConfiguration={"Status": "Enabled"}
        )
        sqs = session.client("sqs")
        url = sqs.create_queue(QueueName="ingestion")["QueueUrl"]
        self.queue = SQSQueue(url, region="us-east-1", session=session)

    async def asyncTearDown(self):
        await self.store.aclose()
        await self.queue.aclose()
        self.mock.stop()

    async def put(self, data=b"hello", key="artifact"):
        async def blocks():
            yield data

        return await self.store.put_immutable(
            bucket=self.bucket,
            key=key,
            blocks=blocks(),
            expected_size_bytes=len(data),
            expected_sha256=hashlib.sha256(data).hexdigest(),
        )

    async def test_immutable_replay_and_conflicting_bytes(self):
        first = await self.put()
        second = await self.put()
        self.assertEqual(first.ref, second.ref)
        with self.assertRaises(ValueError):
            await self.put(b"different")
        self.assertEqual((await self.store.stat(first.ref)).size_bytes, 5)

    async def test_pinned_read_survives_later_version(self):
        first = await self.put()
        self.store.client.put_object(
            Bucket=self.bucket, Key=first.ref.key, Body=b"later"
        )
        async with self.store.open_reader(first.ref, block_size=2) as blocks:
            self.assertEqual(b"".join([b async for b in blocks]), b"hello")
        await self.store.delete(first.ref)
        self.assertIsNone(await self.store.stat(first.ref))
        self.assertIsNotNone(
            await self.store.stat(ObjectRef(bucket=self.bucket, key="artifact"))
        )

    async def test_checksum_and_size_rejected_before_write(self):
        async def blocks():
            yield b"abc"

        with self.assertRaises(ValueError):
            await self.store.put_immutable(
                bucket=self.bucket,
                key="bad",
                blocks=blocks(),
                expected_size_bytes=4,
                expected_sha256="0" * 64,
            )
        self.assertIsNone(
            await self.store.stat(ObjectRef(bucket=self.bucket, key="bad"))
        )

    async def test_grants_bind_headers_and_exact_read_version(self):
        info = await self.put()
        grant = await self.store.presign_put(
            ObjectRef(bucket=self.bucket, key="new"),
            expected_size_bytes=5,
            expected_sha256=hashlib.sha256(b"hello").hexdigest(),
            expires_seconds=60,
        )
        query = parse_qs(urlsplit(grant.url).query)
        self.assertIn("if-none-match", query["X-Amz-SignedHeaders"][0])
        self.assertIn("content-length", query["X-Amz-SignedHeaders"][0])
        self.assertEqual(dict(grant.headers)["If-None-Match"], "*")
        view = await self.store.presign_read(
            info.ref, expires_seconds=60, download=False, filename="report.txt"
        )
        self.assertEqual(
            parse_qs(urlsplit(view.url).query)["versionId"], [info.ref.version_id]
        )
        self.assertIn("inline", view.url)

    async def test_multipart_lists_and_completes_observed_parts(self):
        transfer = await self.store.begin_multipart(
            ObjectRef(bucket=self.bucket, key="multipart")
        )
        data = (b"a" * (5 * 1024**2), b"tail")
        for number, body in enumerate(data, 1):
            self.store.client.upload_part(
                Bucket=self.bucket,
                Key="multipart",
                UploadId=transfer.provider_upload_id,
                PartNumber=number,
                Body=body,
            )
        page = await self.store.list_parts(transfer, after_part_number=None, limit=1)
        self.assertEqual(len(page.parts), 1)
        self.assertEqual(page.next_after, 1)
        second = await self.store.list_parts(transfer, after_part_number=1, limit=100)
        with self.assertRaises((ValueError, PermanentInputError)):
            await self.store.complete_multipart(
                transfer,
                parts=(
                    UploadedPart(part_number=1, etag="wrong", size_bytes=len(data[0])),
                    second.parts[0],
                ),
            )
        info = await self.store.complete_multipart(
            transfer, parts=page.parts + second.parts
        )
        self.assertEqual(info.size_bytes, sum(map(len, data)))
        self.assertIsNotNone(info.ref.version_id)

    async def test_unversioned_bucket_rejected(self):
        self.store.client.create_bucket(Bucket="unversioned-tests")
        with self.assertRaises(OperatorActionRequiredError):
            await self.store.doctor("unversioned-tests")

    async def test_queue_round_trip_visibility_and_ack(self):
        event = JobAvailable(event_id=uuid4(), tenant_id=uuid4(), job_id=uuid4())
        await self.queue.publish(event)
        first = (
            await self.queue.receive(
                max_messages=1, wait_seconds=0, visibility_seconds=30
            )
        )[0]
        self.assertEqual(JobAvailable.from_json(first.body), event)
        await self.queue.extend_visibility(first, seconds=0)
        second = (
            await self.queue.receive(
                max_messages=1, wait_seconds=0, visibility_seconds=30
            )
        )[0]
        self.assertGreater(second.receive_count, first.receive_count)
        await self.queue.acknowledge(second)
        self.assertEqual(
            await self.queue.receive(
                max_messages=1, wait_seconds=0, visibility_seconds=30
            ),
            (),
        )

    async def test_authorized_cleanup_erases_only_exact_prefix(self):
        prefix = "tenants/test/generations/one/"
        await self.put(key=prefix + "a")
        await self.put(key="tenants/test/generations/other/a")
        self.store.client.put_object(
            Bucket=self.bucket, Key=prefix + "a", Body=b"version2"
        )
        await self.store.delete_prefix_versions(self.bucket, prefix)
        self.assertIsNone(
            await self.store.stat(ObjectRef(bucket=self.bucket, key=prefix + "a"))
        )
        self.assertIsNotNone(
            await self.store.stat(
                ObjectRef(bucket=self.bucket, key="tenants/test/generations/other/a")
            )
        )
        with self.assertRaises(ValueError):
            await self.store.delete_prefix_versions(self.bucket, "tenants/")
