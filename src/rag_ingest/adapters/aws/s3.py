"""Default S3 object/upload adapter with bounded streams and protected writes.

Original uploads go directly from the client to storage. Only small bounded
processing artifacts are spooled here. Credentials use the standard AWS chain.
"""

import asyncio
import base64
import hashlib
import hmac
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import boto3
from botocore.exceptions import BotoCoreError
from botocore.handlers import add_expect_header
from botocore.utils import determine_content_length
from .common import call, sdk_config
from rag_ingest.domain.limits import (
    MAX_ARTIFACT_BYTES,
    MULTIPART_MAX_PART_BYTES,
    MULTIPART_MIN_PART_BYTES,
)
from rag_ingest.contracts.storage import ObjectRef, ObjectInfo, PresignedRead
from rag_ingest.contracts.uploads import (
    PresignedUpload,
    MultipartTransfer,
    UploadedPart,
    PartPage,
)
from rag_ingest.contracts.errors import (
    PermanentInputError,
    UncertainOutcomeError,
    OperatorActionRequiredError,
)


EXPECT_CONTINUE_MIN_BYTES = 16 * 1024**2


def expect_continue_for_large_bodies(model, params, **kwargs):
    """Send `Expect: 100-continue` only for bodies of at least 16 MiB.

    With the header, the SDK waits one full round trip for S3's "go ahead"
    before sending any bytes (measured on real S3: about 450 ms per PUT).
    That only pays off when S3 may refuse a large body early; the small
    manifests, chunk batches and segments written here skip it.
    """
    add_expect_header(model, params, **kwargs)
    headers = params.get("headers", {})
    if "Expect" not in headers:
        return
    length = headers.get("Content-Length")
    size = int(length) if length is not None else determine_content_length(params["body"])
    if size is not None and size < EXPECT_CONTINUE_MIN_BYTES:
        del headers["Expect"]


def without_small_expect_continue(client):
    """Replace botocore's always-on Expect handler with the size-aware one."""
    client.meta.events.unregister("before-call.s3", add_expect_header)
    client.meta.events.register("before-call.s3", expect_continue_for_large_bodies)
    return client


class S3ObjectStore:
    """Implement ObjectStore and UploadStore for an existing S3 bucket.

    Accepted sources require pinned versions by default. Enabling bucket
    versioning is an explicit deployment step, not a startup side effect.
    """

    def __init__(
        self,
        *,
        region="ap-south-1",
        endpoint_url=None,
        public_endpoint_url=None,
        session=None,
        max_artifact_bytes=MAX_ARTIFACT_BYTES,
        require_versioning=True,
    ):
        """Create separate internal and optionally client-facing signing clients."""
        session = session or boto3.Session()
        self.client = without_small_expect_continue(
            session.client(
                "s3", region_name=region, endpoint_url=endpoint_url, config=sdk_config()
            )
        )
        self.signer = session.client(
            "s3",
            region_name=region,
            endpoint_url=public_endpoint_url or endpoint_url,
            config=sdk_config(),
        )
        self.max_artifact_bytes = max_artifact_bytes
        self.require_versioning = require_versioning
        self.closed = False

    @staticmethod
    def _params(ref):
        """Always include an explicit object version when the record pins one."""
        args = {"Bucket": ref.bucket, "Key": ref.key}
        if ref.version_id is not None:
            args["VersionId"] = ref.version_id
        return args

    @staticmethod
    def _expiry(seconds):
        """Bound grants to one hour; caller also clamps them to session lifetime."""
        if type(seconds) is not int or not 1 <= seconds <= 3600:
            raise ValueError("Presigned duration must be 1..3600 seconds")
        return datetime.now(timezone.utc) + timedelta(seconds=seconds)

    @staticmethod
    def _checksum(value):
        """Convert the public lowercase SHA-256 hex digest to provider base64."""
        if value is None:
            return None
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Invalid SHA-256 digest")
        return base64.b64encode(bytes.fromhex(value)).decode()

    async def stat(self, ref):
        """Return exact inspected identity; distinguish absence from access failure."""
        try:
            r = await call(self.client, "head_object", **self._params(ref))
        except PermanentInputError as e:
            if e.status_code == 404 or e.provider_code in (
                "404",
                "NoSuchKey",
                "NoSuchVersion",
                "NotFound",
            ):
                return None
            raise
        version = r.get("VersionId") or ref.version_id
        return ObjectInfo(
            ref=ObjectRef(bucket=ref.bucket, key=ref.key, version_id=version),
            size_bytes=r["ContentLength"],
        )

    @asynccontextmanager
    async def open_reader(self, ref, *, block_size):
        """Release the SDK response stream on completion, early stop, or cancellation."""
        if type(block_size) is not int or not 1 <= block_size <= 8 * 1024**2:
            raise ValueError("Invalid storage block size")
        response = await call(self.client, "get_object", **self._params(ref))
        body = response["Body"]

        async def blocks():
            """Read one bounded block at a time without buffering the file."""
            try:
                while True:
                    block = await asyncio.to_thread(body.read, block_size)
                    if not block:
                        break
                    yield block
            except asyncio.CancelledError:
                raise
            except (BotoCoreError, OSError):
                from rag_ingest.contracts.errors import RetryableError

                raise RetryableError("Source stream interrupted") from None

        try:
            yield blocks()
        finally:
            await asyncio.to_thread(body.close)

    async def verified_sha256(self, ref):
        """(size, sha256 hex) that S3 itself verified for this pinned version.

        S3 stores a full-object SHA-256 only when the upload sent
        x-amz-checksum-sha256 in a single PUT, and it refuses that upload
        (BadDigest) when the bytes do not match. Multipart uploads (composite
        checksums), uploads without the header and unpinned refs give None,
        and the caller then hashes every byte itself.
        """
        if ref.version_id is None or ref.version_id == "null":
            return None
        r = await call(
            self.client, "head_object", ChecksumMode="ENABLED", **self._params(ref)
        )
        value = r.get("ChecksumSHA256")
        if not value or r.get("ChecksumType") != "FULL_OBJECT":
            return None
        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError:
            return None
        if len(raw) != 32 or type(r.get("ContentLength")) is not int:
            return None
        return r["ContentLength"], raw.hex()

    async def read_prefix(self, ref, length):
        """Read only the first `length` bytes of this exact object version."""
        if type(length) is not int or not 1 <= length <= 1024**2:
            raise ValueError("Invalid prefix length")
        r = await call(
            self.client, "get_object", Range=f"bytes=0-{length - 1}", **self._params(ref)
        )
        body = r["Body"]
        try:
            data = await asyncio.to_thread(body.read, length + 1)
        finally:
            await asyncio.to_thread(body.close)
        return data[:length]

    async def hash_object(self, ref, *, max_bytes):
        """Compute a trusted full-byte digest while enforcing the actual size limit."""
        size = 0
        digest = hashlib.sha256()
        async with self.open_reader(ref, block_size=1024**2) as blocks:
            async for block in blocks:
                size += len(block)
                if size > max_bytes:
                    raise PermanentInputError("Stored file exceeds its allowed size")
                digest.update(block)
        return size, digest.hexdigest()

    async def put_immutable(
        self, *, bucket, key, blocks, expected_size_bytes, expected_sha256
    ):
        """Verify bounded spooled output, then create it with an atomic precondition."""
        self._checksum(expected_sha256)
        if (
            type(expected_size_bytes) is not int
            or not 0 <= expected_size_bytes <= self.max_artifact_bytes
        ):
            raise PermanentInputError(
                "Processing artifact exceeds configured disk limit"
            )
        digest = hashlib.sha256()
        size = 0
        with tempfile.TemporaryFile() as stream:
            async for block in blocks:
                if not isinstance(block, bytes):
                    raise ValueError("Artifact stream must yield bytes")
                size += len(block)
                if size > expected_size_bytes:
                    raise ValueError("Artifact size mismatch")
                digest.update(block)
                await asyncio.to_thread(stream.write, block)
            if size != expected_size_bytes or not hmac.compare_digest(
                digest.hexdigest(), expected_sha256
            ):
                raise ValueError("Artifact hash or size mismatch")
            stream.seek(0)
            try:
                r = await call(
                    self.client,
                    "put_object",
                    write=True,
                    Bucket=bucket,
                    Key=key,
                    Body=stream,
                    ContentLength=size,
                    ChecksumSHA256=self._checksum(expected_sha256),
                    IfNoneMatch="*",
                    ContentType="application/octet-stream",
                )
            except PermanentInputError as e:
                if e.status_code not in (409, 412):
                    raise
                info = await self.stat(ObjectRef(bucket=bucket, key=key))
                if not info:
                    raise UncertainOutcomeError(
                        "Conditional artifact write needs reconciliation"
                    ) from None
                actual_size, actual_hash = await self.hash_object(
                    info.ref, max_bytes=self.max_artifact_bytes
                )
                if (actual_size, actual_hash) != (size, expected_sha256):
                    raise ValueError(
                        "Immutable artifact already contains different bytes"
                    ) from None
                return info
            return ObjectInfo(
                ref=ObjectRef(bucket=bucket, key=key, version_id=r.get("VersionId")),
                size_bytes=size,
            )

    async def read_bytes(self, ref, *, max_bytes):
        """Buffer only bounded metadata or batch artifacts, never arbitrary sources."""
        data = bytearray()
        async with self.open_reader(
            ref, block_size=min(max_bytes + 1, 1024**2)
        ) as blocks:
            async for block in blocks:
                data.extend(block)
                if len(data) > max_bytes:
                    raise PermanentInputError("Artifact exceeds reader byte limit")
        return bytes(data)

    async def presign_read(
        self, ref, *, expires_seconds, download=False, filename=None
    ):
        """Sign a GET only after the service authorizes the exact document version."""
        expiry = self._expiry(expires_seconds)
        name = (filename or "document")[:255]
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
            raise ValueError("Invalid display filename")
        disposition = "attachment" if download else "inline"
        params = {
            **self._params(ref),
            "ResponseContentDisposition": disposition
            + "; filename*=UTF-8''"
            + quote(name, safe=""),
        }
        params["ResponseContentType"] = (
            "application/pdf"
            if name.lower().endswith(".pdf")
            else (
                "text/plain; charset=utf-8"
                if name.lower().endswith(".txt")
                else "application/octet-stream"
            )
        )
        url = await asyncio.to_thread(
            self.signer.generate_presigned_url,
            "get_object",
            Params=params,
            ExpiresIn=expires_seconds,
            HttpMethod="GET",
        )
        return PresignedRead(url=url, expires_at=expiry)

    async def presign_put(
        self, target, *, expected_size_bytes, expected_sha256, expires_seconds
    ):
        """Sign a create-only whole-file PUT; final verification still reads stored bytes."""
        expiry = self._expiry(expires_seconds)
        if (
            target.version_id is not None
            or type(expected_size_bytes) is not int
            or expected_size_bytes < 0
        ):
            raise ValueError("Invalid upload target or size")
        params = {
            "Bucket": target.bucket,
            "Key": target.key,
            "ContentLength": expected_size_bytes,
            "IfNoneMatch": "*",
        }
        headers = [("If-None-Match", "*")]
        checksum = self._checksum(expected_sha256)
        if checksum:
            params["ChecksumSHA256"] = checksum
            headers.append(("x-amz-checksum-sha256", checksum))
        url = await asyncio.to_thread(
            self.signer.generate_presigned_url,
            "put_object",
            Params=params,
            ExpiresIn=expires_seconds,
            HttpMethod="PUT",
        )
        return PresignedUpload(url=url, expires_at=expiry, headers=tuple(headers))

    async def begin_multipart(self, target):
        """Start one transfer; a lost initiation response may require orphan cleanup."""
        if target.version_id is not None:
            raise ValueError("Multipart target cannot specify a version")
        r = await call(
            self.client,
            "create_multipart_upload",
            write=True,
            Bucket=target.bucket,
            Key=target.key,
        )
        return MultipartTransfer(target=target, provider_upload_id=r["UploadId"])

    async def presign_part(
        self,
        transfer,
        *,
        part_number,
        expected_size_bytes,
        expected_sha256,
        expires_seconds,
    ):
        """Sign the fixed part number and byte count from the stored upload plan."""
        expiry = self._expiry(expires_seconds)
        if (
            type(part_number) is not int
            or not 1 <= part_number <= 10000
            or not 0 <= expected_size_bytes <= MULTIPART_MAX_PART_BYTES
        ):
            raise ValueError("Invalid multipart part")
        params = {
            "Bucket": transfer.target.bucket,
            "Key": transfer.target.key,
            "UploadId": transfer.provider_upload_id,
            "PartNumber": part_number,
            "ContentLength": expected_size_bytes,
        }
        headers = []
        checksum = self._checksum(expected_sha256)
        if checksum:
            params["ChecksumSHA256"] = checksum
            headers.append(("x-amz-checksum-sha256", checksum))
        url = await asyncio.to_thread(
            self.signer.generate_presigned_url,
            "upload_part",
            Params=params,
            ExpiresIn=expires_seconds,
            HttpMethod="PUT",
        )
        return PresignedUpload(url=url, expires_at=expiry, headers=tuple(headers))

    async def _raw_parts(self, transfer, after=None, limit=1000):
        """List provider truth in a bounded page without trusting browser receipts."""
        args = {
            "Bucket": transfer.target.bucket,
            "Key": transfer.target.key,
            "UploadId": transfer.provider_upload_id,
            "MaxParts": limit,
        }
        if after is not None:
            args["PartNumberMarker"] = after
        return await call(self.client, "list_parts", **args)

    async def list_parts(self, transfer, *, after_part_number, limit):
        """Return provider-reported part identities with a safe advancing cursor."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Invalid part listing limit")
        r = await self._raw_parts(transfer, after_part_number, limit)
        parts = tuple(
            UploadedPart(
                part_number=p["PartNumber"], etag=p["ETag"], size_bytes=p["Size"]
            )
            for p in r.get("Parts", ())
        )
        return PartPage(
            parts=parts,
            next_after=parts[-1].part_number
            if r.get("IsTruncated") and parts
            else None,
        )

    async def complete_multipart(self, transfer, *, parts):
        """Reconcile the exact ordered part list, then perform create-only completion."""
        if (
            not parts
            or len(parts) > 10000
            or tuple(p.part_number for p in parts) != tuple(range(1, len(parts) + 1))
        ):
            raise ValueError("Completion requires consecutive parts starting at one")
        raw = []
        after = None
        while True:
            page = await self._raw_parts(transfer, after)
            raw.extend(page.get("Parts", ()))
            if not page.get("IsTruncated"):
                break
            after = page["NextPartNumberMarker"]
        if [(p["PartNumber"], p["ETag"], p["Size"]) for p in raw] != [
            (p.part_number, p.etag, p.size_bytes) for p in parts
        ]:
            raise ValueError("Completion parts differ from provider state")
        if any(p.size_bytes < MULTIPART_MIN_PART_BYTES for p in parts[:-1]):
            raise ValueError("Nonfinal parts must be at least 5 MiB")
        entries = [
            {
                k: p[k]
                for k in (
                    "PartNumber",
                    "ETag",
                    "ChecksumSHA256",
                    "ChecksumCRC32",
                    "ChecksumCRC32C",
                    "ChecksumSHA1",
                    "ChecksumCRC64NVME",
                )
                if k in p
            }
            for p in raw
        ]
        r = await call(
            self.client,
            "complete_multipart_upload",
            write=True,
            Bucket=transfer.target.bucket,
            Key=transfer.target.key,
            UploadId=transfer.provider_upload_id,
            MultipartUpload={"Parts": entries},
            IfNoneMatch="*",
        )
        ref = ObjectRef(
            bucket=transfer.target.bucket,
            key=transfer.target.key,
            version_id=r.get("VersionId"),
        )
        info = await self.stat(ref)
        if not info or info.size_bytes != sum(p.size_bytes for p in parts):
            raise UncertainOutcomeError(
                "Multipart completion object needs verification"
            )
        return info

    async def abort_multipart(self, transfer):
        """Abort only an unfinished transfer; never delete a completed source object."""
        try:
            await call(
                self.client,
                "abort_multipart_upload",
                write=True,
                Bucket=transfer.target.bucket,
                Key=transfer.target.key,
                UploadId=transfer.provider_upload_id,
            )
        except PermanentInputError as e:
            if e.provider_code != "NoSuchUpload":
                raise

    async def delete(self, ref):
        """Delete an exact protected object after the cleanup service authorizes it."""
        await call(self.client, "delete_object", write=True, **self._params(ref))

    async def doctor(self, bucket):
        """Inspect configured bucket access and versioning without changing it."""
        await call(self.client, "head_bucket", Bucket=bucket)
        result = await call(self.client, "get_bucket_versioning", Bucket=bucket)
        if self.require_versioning and result.get("Status") != "Enabled":
            raise OperatorActionRequiredError("S3 bucket versioning must be enabled")
        return result

    async def aclose(self):
        """Release both SDK pools safely without changing stored data."""
        if not self.closed:
            self.closed = True
            await asyncio.gather(
                asyncio.to_thread(self.client.close),
                asyncio.to_thread(self.signer.close),
            )

    async def delete_prefix_versions(self, bucket, prefix):
        """Delete a previously authorized generation prefix in bounded version pages.

        Only the janitor calls this after checking retired/deleted state and
        grace periods. It is deliberately not part of the generic delete API.
        """
        if (
            not prefix.startswith("tenants/")
            or "/generations/" not in prefix
            or not prefix.endswith("/")
        ):
            raise ValueError("Unsafe cleanup prefix")
        key_marker = None
        version_marker = None
        while True:
            args = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
            if key_marker:
                args["KeyMarker"] = key_marker
            if version_marker:
                args["VersionIdMarker"] = version_marker
            page = await call(self.client, "list_object_versions", **args)
            entries = page.get("Versions", []) + page.get("DeleteMarkers", [])
            for item in entries:
                await self.delete(
                    ObjectRef(
                        bucket=bucket, key=item["Key"], version_id=item["VersionId"]
                    )
                )
            if not page.get("IsTruncated"):
                break
            key_marker = page.get("NextKeyMarker")
            version_marker = page.get("NextVersionIdMarker")

    async def delete_all_key_versions(self, ref):
        """Erase a unique upload key only after durable reference-safe authorization."""
        after = None
        version = None
        while True:
            args = {"Bucket": ref.bucket, "Prefix": ref.key, "MaxKeys": 1000}
            if after:
                args["KeyMarker"] = after
            if version:
                args["VersionIdMarker"] = version
            page = await call(self.client, "list_object_versions", **args)
            for item in page.get("Versions", []) + page.get("DeleteMarkers", []):
                if item["Key"] == ref.key:
                    await self.delete(
                        ObjectRef(
                            bucket=ref.bucket, key=ref.key, version_id=item["VersionId"]
                        )
                    )
            if not page.get("IsTruncated"):
                break
            after = page.get("NextKeyMarker")
            version = page.get("NextVersionIdMarker")
