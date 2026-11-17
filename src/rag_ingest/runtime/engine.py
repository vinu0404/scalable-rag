"""Coordinate durable ingestion through injected adapters.

Core services own identity, verification, checkpoints, and publication. Queue
messages contain references only; accepted work is always recoverable in SQL.
"""

import asyncio
import codecs
import logging
import hashlib
import json
import tempfile
from collections import deque
from contextlib import aclosing
from functools import partial
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4, uuid5
from rag_ingest.domain.documents import (
    Document,
    ProcessingGeneration,
    SourceVersion,
    normalize_metadata,
    normalize_tags,
)
from rag_ingest.domain.uploads import (
    UploadSession,
    UploadStatus as US,
    request_fingerprint,
)
from rag_ingest.domain.jobs import Job
from rag_ingest.domain.enums import JobStatus as JS, ProcessingStage as ST
from rag_ingest.domain.events import JobAvailable
from rag_ingest.domain.outbox import OutboxRecord
from rag_ingest.domain.artifacts import ContentArtifact
from rag_ingest.domain.multipart import MultipartPlan, ObservedUploadPart
from rag_ingest.domain.policy import ExecutionBudget
from rag_ingest.domain.manifests import ManifestOutput, StageManifest
from rag_ingest.domain.manifest_reader import read_stage_manifest, ManifestReadLimits
from rag_ingest.domain.inventory import ChunkBatch, GenerationInventory
from rag_ingest.domain.search import chunk_digest
from rag_ingest.contracts.keyword import KeywordDocument, KeywordExpectation
from rag_ingest.domain.limits import (
    FILE_TYPE_PREFIX_BYTES,
    MAX_ARTIFACT_BYTES,
    MAX_BATCH_ITEMS,
    MAX_CHUNK_CHARACTERS,
    MAX_DIGEST_IDS,
    MAX_FILENAME_CHARS,
    MAX_GENERATION_CHUNKS,
    MAX_INDEX_RECEIPT_BYTES,
    MAX_INVENTORY_BATCHES,
    MAX_INVENTORY_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PAGE_CHARACTERS,
    MAX_PAGE_MAP_BYTES,
    MAX_REQUEST_KEY_CHARS,
    MULTIPART_MAX_PART_BYTES,
    MULTIPART_MAX_PARTS,
    MULTIPART_MIN_PART_BYTES,
    READ_BLOCK_BYTES,
    MiB,
)
from rag_ingest.domain.inventory_reader import (
    compute_vector_id_digest,
    read_generation_inventory,
    InventoryReadLimits,
)
from rag_ingest.domain.checkpoints import StageCheckpoint
from rag_ingest.contracts.storage import ObjectRef
from rag_ingest.contracts.uploads import MultipartTransfer
from rag_ingest.contracts.extraction import ExtractedPage, ExtractionLimits
from rag_ingest.domain import extraction_recipe
from rag_ingest.logs import log_event
from rag_ingest.runtime.recognition import Recognition, SourceFile
from rag_ingest.runtime.embedding_segments import (
    FORMAT_NAME as SEGMENT_FORMAT,
    FORMAT_VERSION as SEGMENT_VERSION,
    decode_segment,
    encode_segment,
    merge_segments,
)
from rag_ingest.contracts.chunking import ChunkingInput, ChunkingLimits
from rag_ingest.contracts.embedding import EmbeddingLimits, EmbeddingFailure
from rag_ingest.contracts.vectors import VectorScope, VectorReadiness
from rag_ingest.contracts.errors import (
    PermanentInputError,
    UncertainOutcomeError,
    DuplicateContentError,
    RetryableError,
    OperatorActionRequiredError,
    RecordVersionError,
    with_code,
)
from rag_ingest.adapters.postgres.codec import encode, decode

log = logging.getLogger(__name__)
# Bound on cache entries per batch: one per attempt, plus races; far above use.
EMBEDDING_CACHE_ENTRIES = 100
# Batches index/publish load ahead while writing or checking the current one;
# memory is bounded by (this + 1) batches per job, whatever the document size.
INDEX_READ_AHEAD = 2
# Chunk batches uploaded at once; chunking waits beyond this, so memory stays
# bounded by this many batches per job.
CHUNK_UPLOADS = 8
# What a damaged or foreign cache entry can raise while being parsed/checked.
# (Storage errors are not here: an unreachable store must not look like a miss.)
CACHE_DEFECTS = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    PermanentInputError,
    RecordVersionError,
)


def ref_of(output):
    """Translate a manifest's storage fields into a provider-independent reference."""
    return ObjectRef(
        bucket=output.bucket, key=output.object_key, version_id=output.object_version_id
    )


class IngestionEngine:
    """Compose independent adapters; library users can replace any one of them."""

    def __init__(
        self,
        *,
        metadata,
        objects,
        uploads,
        queue,
        extractor,
        chunker,
        embedder,
        vectors,
        settings,
        recognizer=None,
        keyword=None,
        reranker=None,
    ):
        """Receive dependencies instead of constructing provider SDK clients here.

        keyword: an optional KeywordIndex. When given, the index stage writes
        every chunk to it, publication verifies it, cleanup deletes from it,
        and hybrid search can query it.
        reranker: an optional Reranker that reorders the top search candidates.
        """
        self.metadata = metadata
        self.objects = objects
        self.uploads = uploads
        self.queue = queue
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vectors = vectors
        self.settings = settings
        self.recognizer = recognizer
        self.keyword = keyword
        self.reranker = reranker

    async def read_bytes(self, ref, max_bytes):
        """Read only bounded artifacts, checking actual bytes during streaming."""
        data = bytearray()
        async with self.objects.open_reader(ref, block_size=READ_BLOCK_BYTES) as blocks:
            async for block in blocks:
                if len(data) + len(block) > max_bytes:
                    raise PermanentInputError("Artifact exceeds byte policy")
                data.extend(block)
        return bytes(data)

    async def read_output(self, output, max_bytes=MAX_ARTIFACT_BYTES):
        """Verify output size and checksum before decoding or reuse."""
        if output.size_bytes > max_bytes:
            raise PermanentInputError("Output exceeds reader policy")
        data = await self.read_bytes(ref_of(output), max_bytes)
        if (
            len(data) != output.size_bytes
            or hashlib.sha256(data).hexdigest() != output.sha256
        ):
            raise PermanentInputError("Saved output integrity check failed")
        return data

    async def store_bytes(self, key, data, role, format_name, version=1):
        """Write exact bytes immutably and return a checksummed output descriptor."""
        if len(data) > MAX_ARTIFACT_BYTES:
            raise PermanentInputError(
                "Batch artifact exceeds 128 MiB; reduce batch size"
            )

        async def blocks():
            """Yield a small bounded payload to the object-store contract."""
            yield data

        digest = hashlib.sha256(data).hexdigest()
        info = await self.objects.put_immutable(
            bucket=self.settings.bucket,
            key=key,
            blocks=blocks(),
            expected_size_bytes=len(data),
            expected_sha256=digest,
        )
        return ManifestOutput(
            role=role,
            bucket=info.ref.bucket,
            object_key=info.ref.key,
            object_version_id=info.ref.version_id,
            sha256=digest,
            size_bytes=len(data),
            format_name=format_name,
            format_version=version,
        )

    async def _upload(self, tenant, upload_id, required=US.UPLOADING):
        """Authorize current intake state before issuing another transfer operation."""
        u = await self.metadata.get_upload(tenant_id=tenant, upload_id=upload_id)
        now = await self.metadata.now(tenant)
        doc = (
            await self.metadata.get_document(
                tenant_id=tenant, document_id=u.document_id
            )
            if u
            else None
        )
        if (
            not u
            or not doc
            or doc.deleted_at
            or doc.acl_ref != "tenant"
            or u.status != required
            or u.expires_at <= now
        ):
            raise ValueError("Upload is absent, expired, deleted, or in another state")
        return u, now

    async def initiate_upload(
        self,
        tenant,
        *,
        filename,
        size_bytes,
        request_key,
        document_id=None,
        sha256=None,
        allow_duplicate=False,
        tags=None,
        metadata=None,
    ):
        """Durably register intake, then return a direct grant or multipart plan.

        tags and metadata label a new document for search filters; an existing
        document's labels change through update_document_labels instead.

        Unless duplicates are allowed (per request, or service-wide with
        INGEST_ALLOW_DUPLICATES), a declared sha256 that a live document already has raises
        DuplicateContentError before anything is created or uploaded. Without
        a declared sha256 the same check runs when the upload is accepted.
        """
        if type(allow_duplicate) is not bool:
            raise ValueError("allow_duplicate must be a boolean")
        if document_id and (tags is not None or metadata is not None):
            raise ValueError("Change an existing document's labels with PATCH, not an upload")
        tags = normalize_tags(list(tags or ()))
        metadata = normalize_metadata(metadata or {})
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or len(filename) > MAX_FILENAME_CHARS
            or "/" in filename
            or "\\" in filename
            or any(ord(c) < 32 for c in filename)
        ):
            raise ValueError("Invalid filename")
        if not filename.lower().endswith((".pdf", ".txt")):
            raise PermanentInputError(
                "Built-in intake supports PDF and UTF-8 .txt files"
            )
        if (
            type(size_bytes) is not int
            or not 1 <= size_bytes <= self.settings.max_upload_bytes
        ):
            raise ValueError("File size outside configured upload limits")
        if not isinstance(request_key, str) or not 1 <= len(request_key) <= MAX_REQUEST_KEY_CHARS:
            raise ValueError("A bounded idempotency key is required")
        now = await self.metadata.now(tenant)
        recipe = self.settings.pipeline()
        doc = (
            await self.metadata.get_document(tenant_id=tenant, document_id=document_id)
            if document_id
            else None
        )
        if document_id and (not doc or doc.deleted_at or doc.acl_ref != "tenant"):
            raise ValueError("Replacement document is unavailable")
        new = None
        if not document_id:
            document_id = uuid4()
            new = Document(
                tenant_id=tenant,
                document_id=document_id,
                display_name=filename,
                acl_ref="tenant",
                created_at=now,
                tags=tags,
                metadata=metadata,
            )
        id = uuid4()
        u = UploadSession(
            tenant_id=tenant,
            document_id=document_id,
            upload_id=id,
            request_key=request_key,
            original_name=filename,
            bucket=self.settings.bucket,
            object_key=f"tenants/{tenant}/uploads/{id}/source",
            expected_size_bytes=size_bytes,
            expected_sha256=sha256,
            created_at=now,
            expires_at=now + timedelta(seconds=self.settings.upload_seconds),
            allow_duplicate=allow_duplicate or not self.settings.refuse_duplicates,
        )
        # The idempotency key binds the CLIENT's request only. The recipe is a
        # server decision made once at first acceptance; a replay after a
        # config change must return that original upload, not be refused.
        descriptor = {
            "filename": filename,
            "size": size_bytes,
            "sha256": sha256,
            "document": str(doc.document_id) if doc else None,
        }
        if allow_duplicate:  # only when asked, so earlier keys keep their fingerprint
            descriptor["allow_duplicate"] = True
        if tags or metadata:  # likewise only when given
            descriptor["tags"] = list(tags)
            descriptor["metadata"] = dict(metadata)
        fingerprint = request_fingerprint(descriptor)
        u = await self.metadata.initiate_upload(
            upload=u,
            pipeline=recipe,
            request_fingerprint=fingerprint,
            expected_desired_generation_id=doc.desired_generation_id if doc else None,
            new_document=new,
        )
        if u.status != US.UPLOADING:
            return {"upload": u, "plan": None, "grant": None}
        u, now = await self._upload(tenant, u.upload_id)
        ttl = max(
            1,
            min(
                self.settings.presign_seconds, int((u.expires_at - now).total_seconds())
            ),
        )
        target = ObjectRef(bucket=u.bucket, key=u.object_key)
        if u.expected_size_bytes < self.settings.multipart_threshold:
            grant = await self.uploads.presign_put(
                target,
                expected_size_bytes=u.expected_size_bytes,
                expected_sha256=u.expected_sha256,
                expires_seconds=ttl,
            )
            return {"upload": u, "plan": None, "grant": grant}
        plan = await self.metadata.get_multipart_plan(
            tenant_id=tenant, upload_id=u.upload_id
        )
        if not plan:
            size = max(
                self.settings.multipart_part_bytes,
                (u.expected_size_bytes + MULTIPART_MAX_PARTS - 1) // MULTIPART_MAX_PARTS,
                MULTIPART_MIN_PART_BYTES,
            )
            size = ((size + MiB - 1) // MiB) * MiB  # whole MiB parts
            if size > MULTIPART_MAX_PART_BYTES:
                raise ValueError("File exceeds multipart capabilities")
            transfer = await self.uploads.begin_multipart(target)
            plan = MultipartPlan(
                tenant_id=tenant,
                upload_id=u.upload_id,
                provider_upload_id=transfer.provider_upload_id,
                file_size_bytes=u.expected_size_bytes,
                part_size_bytes=size,
                total_parts=(u.expected_size_bytes + size - 1) // size,
            )
            if not await self.metadata.attach_multipart_plan(plan):
                await self.uploads.abort_multipart(transfer)
                plan = await self.metadata.get_multipart_plan(
                    tenant_id=tenant, upload_id=u.upload_id
                )
                if not plan:
                    raise ValueError("Upload state changed during multipart initiation")
        return {
            "upload": await self.metadata.get_upload(
                tenant_id=tenant, upload_id=u.upload_id
            ),
            "plan": plan,
            "grant": None,
        }

    async def part_url(self, tenant, upload_id, number, sha256=None):
        """Sign the same immutable byte range when a client retries a part."""
        u, now = await self._upload(tenant, upload_id)
        plan = await self.metadata.get_multipart_plan(
            tenant_id=tenant, upload_id=upload_id
        )
        if not plan:
            raise ValueError("Upload has no multipart plan")
        ttl = max(
            1,
            min(
                self.settings.presign_seconds, int((u.expires_at - now).total_seconds())
            ),
        )
        return await self.uploads.presign_part(
            MultipartTransfer(
                target=ObjectRef(bucket=u.bucket, key=u.object_key),
                provider_upload_id=plan.provider_upload_id,
            ),
            part_number=number,
            expected_size_bytes=plan.expected_part_size(number),
            expected_sha256=sha256,
            expires_seconds=ttl,
        )

    async def list_parts(self, tenant, upload_id, after=None, limit=1000):
        """Reconcile provider progress into durable, tenant-scoped observations."""
        u, now = await self._upload(tenant, upload_id)
        plan = await self.metadata.get_multipart_plan(
            tenant_id=tenant, upload_id=upload_id
        )
        if not plan:
            raise ValueError("Multipart plan missing")
        page = await self.uploads.list_parts(
            MultipartTransfer(
                target=ObjectRef(bucket=u.bucket, key=u.object_key),
                provider_upload_id=plan.provider_upload_id,
            ),
            after_part_number=after,
            limit=limit,
        )
        for p in page.parts:
            observation = ObservedUploadPart(
                tenant_id=tenant,
                upload_id=upload_id,
                provider_upload_id=plan.provider_upload_id,
                part_number=p.part_number,
                etag=p.etag,
                size_bytes=p.size_bytes,
                observed_at=now,
            )
            observation.validate_against(plan)
            await self.metadata.record_observed_part(observation)
        return page

    async def complete_upload(self, tenant, upload_id):
        """Finish transfer and durably request verification without waiting for hashing."""
        current = await self.metadata.get_upload(tenant_id=tenant, upload_id=upload_id)
        if current and current.status in (US.ACCEPTED, US.VERIFYING):
            return current
        u, now = await self._upload(tenant, upload_id)
        plan = await self.metadata.get_multipart_plan(
            tenant_id=tenant, upload_id=upload_id
        )
        ref = ObjectRef(bucket=u.bucket, key=u.object_key)
        if plan:
            info = await self.objects.stat(ref)
            if info is None:
                all_parts = []
                after = None
                while True:
                    page = await self.list_parts(tenant, upload_id, after)
                    all_parts.extend(page.parts)
                    if page.next_after is None:
                        break
                    after = page.next_after
                if len(all_parts) != plan.total_parts:
                    raise ValueError("Multipart upload is incomplete")
                transfer = MultipartTransfer(
                    target=ref, provider_upload_id=plan.provider_upload_id
                )
                try:
                    info = await self.uploads.complete_multipart(
                        transfer, parts=tuple(all_parts)
                    )
                except UncertainOutcomeError:
                    info = await self.objects.stat(ref)
                    if info is None:
                        raise
            ref = info.ref
        info = await self.objects.stat(ref)
        if not info:
            raise ValueError("Completed upload not found in storage")
        if info.ref.version_id is None or info.ref.version_id == "null":
            raise ValueError("Default intake requires S3 bucket versioning")
        if not await self.metadata.begin_verification(
            tenant_id=tenant, upload_id=upload_id, ref=info.ref
        ):
            raise ValueError("Upload changed before verification")
        return await self.metadata.get_upload(tenant_id=tenant, upload_id=upload_id)

    async def _storage_verified(self, ref):
        """(size, sha256, first 4 KiB) when storage itself verified the SHA-256.

        Uses the store's optional verified_sha256 (S3: the full-object
        x-amz-checksum-sha256 it checked at upload time). None means: read
        and hash every byte as usual.
        """
        probe = getattr(self.objects, "verified_sha256", None)
        if not callable(probe) or not callable(getattr(self.objects, "read_prefix", None)):
            return None
        found = await probe(ref)
        if found is None:
            return None
        size, checksum = found
        prefix = await self.objects.read_prefix(ref, FILE_TYPE_PREFIX_BYTES) if size else b""
        return size, checksum, prefix

    async def verify_upload(self, tenant, upload_id):
        """Check the pinned file, then atomically register accepted processing work.

        Every byte is read and hashed, except for a PDF whose declared SHA-256
        storage already verified at upload time (see _storage_verified).
        """
        u, now = await self._upload(tenant, upload_id, US.VERIFYING)
        ref = await self.metadata.get_verification_ref(
            tenant_id=tenant, upload_id=upload_id
        )
        is_pdf = u.original_name.lower().endswith(".pdf")
        verified = (
            await self._storage_verified(ref)
            if is_pdf and u.expected_sha256
            else None
        )
        if verified is not None:
            # Storage verified the client's SHA-256 over every byte at write
            # time; only the 4 KiB file-type prefix is read here.
            size, checksum, prefix = verified
            if size > min(u.expected_size_bytes, self.settings.max_upload_bytes):
                await self.metadata.reject_verified_upload(
                    tenant_id=tenant,
                    upload_id=upload_id,
                    verified_ref=ref,
                    reason_code="size_mismatch",
                )
                raise PermanentInputError("Stored file exceeds its expected size")
        else:
            digest = hashlib.sha256()
            size = 0
            prefix = b""
            # Validate all text bytes, not a sample; the parser decodes the same way.
            utf8 = None if is_pdf else codecs.getincrementaldecoder("utf-8")()
            utf8_valid = True
            async with self.objects.open_reader(ref, block_size=READ_BLOCK_BYTES) as blocks:
                async for block in blocks:
                    size += len(block)
                    if size > min(u.expected_size_bytes, self.settings.max_upload_bytes):
                        await self.metadata.reject_verified_upload(
                            tenant_id=tenant,
                            upload_id=upload_id,
                            verified_ref=ref,
                            reason_code="size_mismatch",
                        )
                        raise PermanentInputError("Stored file exceeds its expected size")
                    digest.update(block)
                    if len(prefix) < FILE_TYPE_PREFIX_BYTES:
                        prefix = (prefix + block)[:FILE_TYPE_PREFIX_BYTES]
                    if utf8 and utf8_valid:
                        try:
                            utf8.decode(block)
                        except UnicodeDecodeError:
                            utf8_valid = False
            if utf8 and utf8_valid:
                try:
                    utf8.decode(b"", final=True)  # rejects a truncated final character
                except UnicodeDecodeError:
                    utf8_valid = False
            checksum = digest.hexdigest()
        valid = size == u.expected_size_bytes and (
            not u.expected_sha256 or checksum == u.expected_sha256
        )
        if is_pdf:
            valid = valid and prefix.startswith(b"%PDF-")
        else:
            valid = valid and utf8_valid and b"\x00" not in prefix
        if not valid:
            await self.metadata.reject_verified_upload(
                tenant_id=tenant,
                upload_id=upload_id,
                verified_ref=ref,
                reason_code="verification_failed",
            )
            raise PermanentInputError("Stored content does not match the declared file")
        recipe = await self.metadata.get_upload_pipeline(
            tenant_id=tenant, upload_id=upload_id
        )
        now = await self.metadata.now(tenant)
        gen_id = uuid5(upload_id, "generation")
        source_id = uuid5(upload_id, "source")
        artifact = ContentArtifact(
            tenant_id=tenant,
            artifact_id=uuid5(tenant, checksum),
            sha256=checksum,
            size_bytes=size,
            bucket=ref.bucket,
            object_key=ref.key,
            object_version_id=ref.version_id,
            created_at=now,
        )
        source = SourceVersion(
            tenant_id=tenant,
            document_id=u.document_id,
            source_version_id=source_id,
            artifact_id=artifact.artifact_id,
            bucket=ref.bucket,
            object_key=ref.key,
            object_version_id=ref.version_id,
            sha256=checksum,
            size_bytes=size,
            created_at=now,
        )
        gen = ProcessingGeneration(
            tenant_id=tenant,
            document_id=u.document_id,
            source_version_id=source_id,
            generation_id=gen_id,
            pipeline_fingerprint=recipe.fingerprint,
            created_at=now,
        )
        policy = self.settings.core.execution_policy
        end = now + timedelta(seconds=self.settings.core.max_generation_seconds)
        budget = ExecutionBudget(
            policy=policy,
            starts_at=now,
            deadline_at=min(end, now + timedelta(seconds=policy.max_job_seconds)),
            generation_deadline_at=end,
        )
        job = Job(
            tenant_id=tenant,
            document_id=u.document_id,
            generation_id=gen_id,
            job_id=uuid5(gen_id, "extract:0"),
            stage=ST.EXTRACT,
            status=JS.QUEUED,
            created_at=now,
            budget=budget,
        )
        event = JobAvailable(
            tenant_id=tenant, job_id=job.job_id, event_id=uuid5(job.job_id, "initial")
        )
        accepted = await self.metadata.accept_verified_upload(
            tenant_id=tenant,
            upload_id=upload_id,
            verified_ref=ref,
            artifact_candidate=artifact,
            source=source,
            generation=gen,
            first_job=job,
            outbox=OutboxRecord(event=event, created_at=now, available_at=now),
        )
        if not accepted:
            await self.metadata.reject_verified_upload(
                tenant_id=tenant,
                upload_id=upload_id,
                verified_ref=ref,
                reason_code="superseded_intake",
            )
            raise ValueError("Document changed before this upload could be accepted")
        if accepted.duplicate_of:
            existing = await self.metadata.get_document(
                tenant_id=tenant, document_id=accepted.duplicate_of
            )
            raise DuplicateContentError(
                accepted.duplicate_of, existing.display_name if existing else None
            )
        return accepted

    async def view_file(self, tenant, document_id, *, download=False):
        """Authorize a live tenant-wide document before signing an exact source version."""
        doc = await self.metadata.get_document(
            tenant_id=tenant, document_id=document_id
        )
        if not doc or doc.deleted_at or doc.acl_ref != "tenant":
            raise ValueError("Document not found")
        gen_id = doc.active_generation_id or doc.desired_generation_id
        if not gen_id:
            raise ValueError("Document has no accepted source")
        gen = await self.metadata.get_generation(tenant_id=tenant, generation_id=gen_id)
        source = await self.metadata.get_source_version(
            tenant_id=tenant, source_version_id=gen.source_version_id
        )
        return await self.objects.presign_read(
            ObjectRef(
                bucket=source.bucket,
                key=source.object_key,
                version_id=source.object_version_id,
            ),
            expires_seconds=self.settings.presign_seconds,
            download=download,
            filename=doc.display_name,
        )

    async def inventory(self, job):
        """Read only the registered immutable generation inventory."""
        ref = await self.metadata.get_generation_inventory_ref(
            tenant_id=job.tenant_id, generation_id=job.generation_id
        )
        if not ref:
            raise ValueError("Generation has no sealed inventory")
        data = await self.read_output(ref, MAX_INVENTORY_BYTES)
        inv = read_generation_inventory(
            data, expected=ref, limits=InventoryReadLimits()
        )
        if (inv.tenant_id, inv.document_id, inv.generation_id) != (
            job.tenant_id,
            job.document_id,
            job.generation_id,
        ):
            raise ValueError("Inventory scope mismatch")
        return inv

    async def _batch_plan(self, job):
        """This embedding job's sealed batch, without downloading the inventory.

        Reads the one plan row saved when the job was scheduled and checks it
        belongs to this job and to the generation's registered inventory.
        """
        plan = await self.metadata.get_batch_plan(
            tenant_id=job.tenant_id,
            generation_id=job.generation_id,
            batch_key=job.shard_key,
        )
        sealed = await self.metadata.get_generation_inventory_ref(
            tenant_id=job.tenant_id, generation_id=job.generation_id
        )
        if (
            plan is None
            or sealed is None
            or plan.inventory_sha256 != sealed.sha256
            or (plan.tenant_id, plan.document_id, plan.generation_id)
            != (job.tenant_id, job.document_id, job.generation_id)
            or plan.batch.batch_key != job.shard_key
        ):
            raise ValueError("Batch plan does not match this job's sealed inventory")
        return plan

    async def checkpoint_manifest(self, job):
        """Load and validate this job's saved complete stage output if it exists."""
        cp = await self.metadata.get_checkpoint(
            tenant_id=job.tenant_id, job_id=job.job_id
        )
        if not cp:
            return None
        data = await self.read_bytes(
            ObjectRef(
                bucket=cp.manifest_bucket,
                key=cp.manifest_key,
                version_id=cp.manifest_version_id,
            ),
            MAX_MANIFEST_BYTES,
        )
        return read_stage_manifest(data, checkpoint=cp, limits=ManifestReadLimits())

    async def _batch_manifest(self, job, source, pipeline, entry):
        """Decode a pinned chunk manifest and bind it to the sealed expected range."""
        data = await self.read_output(entry.manifest, MAX_MANIFEST_BYTES)
        raw = json.loads(data)
        cp = StageCheckpoint(
            checkpoint_id=uuid4(),
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            generation_id=job.generation_id,
            job_id=UUID(raw["job_id"]),
            producer_lease_id=job.lease.lease_id,
            stage=ST.CHUNK,
            shard_key=entry.batch_key,
            source_sha256=source.sha256,
            input_sha256=raw["input_sha256"],
            pipeline_fingerprint=pipeline.fingerprint,
            manifest_bucket=entry.manifest.bucket,
            manifest_key=entry.manifest.object_key,
            manifest_version_id=entry.manifest.object_version_id,
            manifest_sha256=entry.manifest.sha256,
            manifest_size_bytes=entry.manifest.size_bytes,
            created_at=source.created_at,
        )
        return read_stage_manifest(data, checkpoint=cp, limits=ManifestReadLimits())

    async def _save_manifest(
        self, job, source, pipeline, input_hash, outputs, vector_ids=()
    ):
        """Save a complete stage manifest before its lease-guarded checkpoint commit."""
        manifest = StageManifest(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            generation_id=job.generation_id,
            job_id=job.job_id,
            stage=job.stage,
            shard_key=job.shard_key,
            source_sha256=source.sha256,
            input_sha256=input_hash,
            pipeline_fingerprint=pipeline.fingerprint,
            outputs=tuple(outputs),
            vector_ids=tuple(vector_ids),
        )
        key = f"tenants/{job.tenant_id}/generations/{job.generation_id}/attempts/{job.lease.lease_id}/manifest.json"
        ref = await self.store_bytes(
            key, manifest.to_json().encode(), "manifest", "stage-manifest"
        )
        now = await self.metadata.now(job.tenant_id)
        cp = StageCheckpoint(
            checkpoint_id=uuid5(job.lease.lease_id, "checkpoint"),
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            generation_id=job.generation_id,
            job_id=job.job_id,
            producer_lease_id=job.lease.lease_id,
            stage=job.stage,
            shard_key=job.shard_key,
            source_sha256=source.sha256,
            input_sha256=input_hash,
            pipeline_fingerprint=pipeline.fingerprint,
            manifest_bucket=ref.bucket,
            manifest_key=ref.object_key,
            manifest_version_id=ref.object_version_id,
            manifest_sha256=ref.sha256,
            manifest_size_bytes=ref.size_bytes,
            created_at=now,
        )
        return await self.metadata.save_checkpoint(
            checkpoint=cp, worker_id=job.lease.worker_id, lease_id=job.lease.lease_id
        )

    @staticmethod
    def _output(manifest, role):
        """Require an expected output role rather than guessing a file name."""
        output = next((o for o in manifest.outputs if o.role == role), None)
        if not output:
            raise ValueError("Stage output role is missing")
        return output

    async def execute_job(self, job):
        """Run or resume one bounded durable stage under the runner's heartbeat."""
        gen = await self.metadata.get_generation(
            tenant_id=job.tenant_id, generation_id=job.generation_id
        )
        source = await self.metadata.get_source_version(
            tenant_id=job.tenant_id, source_version_id=gen.source_version_id
        )
        pipeline = await self.metadata.get_generation_pipeline(
            tenant_id=job.tenant_id, generation_id=job.generation_id
        )
        if pipeline.fingerprint != gen.pipeline_fingerprint:
            raise ValueError("Saved pipeline identity mismatch")
        if pipeline.embedder.to_dict() != self.settings.pipeline().embedder.to_dict():
            raise OperatorActionRequiredError(
                "Worker embedding recipe differs from accepted work; use its original deployment"
            )
        if job.stage == ST.PUBLISH:
            await self._publish(job, source, pipeline)
            return
        existing = await self.checkpoint_manifest(job)
        if not existing and job.stage == ST.CHUNK:
            sealed = await self.metadata.get_generation_inventory_ref(
                tenant_id=job.tenant_id, generation_id=job.generation_id
            )
            if sealed:
                await self.inventory(job)
                extracted = await self._stage_job(job, ST.EXTRACT)
                cp = await self._save_manifest(
                    job, source, pipeline, extracted.sha256, (sealed,)
                )
                existing = await self.checkpoint_manifest(job)
        if existing:
            for output in existing.outputs:
                await self.read_output(output)
            cp = await self.metadata.get_checkpoint(
                tenant_id=job.tenant_id, job_id=job.job_id
            )
        else:
            method = {
                ST.EXTRACT: self._extract,
                ST.CHUNK: self._chunk,
                ST.EMBED: self._embed,
                ST.INDEX: self._index,
            }[job.stage]
            input_hash, outputs, ids = await method(job, source, pipeline)
            cp = await self._save_manifest(
                job, source, pipeline, input_hash, outputs, ids
            )
        await self.metadata.complete_stage(
            tenant_id=job.tenant_id,
            job_id=job.job_id,
            worker_id=job.lease.worker_id,
            lease_id=job.lease.lease_id,
            checkpoint_id=cp.checkpoint_id,
            completion_id=uuid5(job.lease.lease_id, "complete"),
            next_jobs=(),
            next_outbox=(),
        )

    async def _extract(self, job, source, pipeline):
        """Run the saved extractor recipe (the document recipe)."""
        spec = pipeline.extractor
        if (spec.driver, spec.implementation_version) != (
            extraction_recipe.DRIVER,
            extraction_recipe.VERSION,
        ):
            raise OperatorActionRequiredError(
                "Build uses an extractor recipe this release does not run; reprocess it"
            )
        return await self._extract_document(job, source, pipeline)

    async def download_verified(self, source, path):
        """Stream the pinned source to disk and prove it is the accepted bytes."""
        digest = hashlib.sha256()
        size = 0
        with path.open("wb") as stream:
            async with self.objects.open_reader(
                ObjectRef(
                    bucket=source.bucket,
                    key=source.object_key,
                    version_id=source.object_version_id,
                ),
                block_size=READ_BLOCK_BYTES,
            ) as blocks:
                async for block in blocks:
                    size += len(block)
                    if size > self.settings.max_source_disk_bytes:
                        raise PermanentInputError("Source exceeds disk budget")
                    digest.update(block)
                    await asyncio.to_thread(stream.write, block)
        if size != source.size_bytes or digest.hexdigest() != source.sha256:
            raise PermanentInputError("Source changed after acceptance")

    def _extraction_limits(self, job):
        return ExtractionLimits(
            max_pages=self.settings.max_pages,
            max_page_characters=min(self.settings.max_extracted_characters, MAX_PAGE_CHARACTERS),
            max_total_characters=self.settings.max_extracted_characters,
            timeout_seconds=max(1, min(240, job.budget.policy.max_attempt_seconds - 1)),
        )

    async def _extract_document(self, job, source, pipeline):
        """Extract native text, recognize marked pages, and record every page's method.

        Pages are never dropped silently: each ends as native text, recognized
        text, blank, or unrecognized (kept empty or failed per the recipe).
        """
        options = extraction_recipe.validate(
            json.loads(pipeline.extractor.options_json)
        )
        if options["recognizer"] is not None:
            r = self.recognizer
            if (
                r is None
                or extraction_recipe.recognizer_block(r.spec, r.capabilities)
                != options["recognizer"]
            ):
                raise OperatorActionRequiredError(
                    "This worker's recognizer differs from the build's recipe; run "
                    "the deployment that accepted it"
                )
        if source.size_bytes > self.settings.max_source_disk_bytes:
            raise PermanentInputError(
                "Source exceeds configured worker temporary-disk budget"
            )
        base = f"tenants/{job.tenant_id}/generations/{job.generation_id}/attempts/{job.lease.lease_id}"
        limits = self._extraction_limits(job)
        loop = asyncio.get_running_loop()
        now = await self.metadata.now(job.tenant_id)
        remaining = (job.lease.attempt_deadline_at - now).total_seconds()
        deadline = loop.time() + max(1.0, remaining * 0.85 - 5)
        with tempfile.TemporaryDirectory(prefix="rag-source-") as temp:
            local = SourceFile(self, source, Path(temp) / "source")
            pages = await self._analysis(job, source, pipeline, local, limits)
            needed = [p.page_number for p in pages if p.needs_recognition]
            recognized = {}
            if needed:
                recognized = await Recognition(
                    self,
                    job=job,
                    source=source,
                    pipeline=pipeline,
                    options=options,
                    page_count=len(pages),
                    local=local,
                    workdir=temp,
                    deadline=deadline,
                ).run(needed)
            final = [
                ExtractedPage(
                    page_number=p.page_number,
                    text=(
                        extraction_recipe.normalize(recognized[p.page_number].text)
                        if options["normalize_text"]
                        else recognized[p.page_number].text
                    ),
                    method="recognized",
                )
                if p.needs_recognition
                else p
                for p in pages
            ]
        unrecognized = [p.page_number for p in final if p.method == "unrecognized"]
        if unrecognized:
            if options["unreadable_pages"] == "fail":
                raise with_code(
                    PermanentInputError(
                        f"{len(unrecognized)} page(s) need recognition; configure a recognizer"
                    ),
                    "recognition_required",
                )
            log_event(
                log,
                "pages_unrecognized",
                logging.WARNING,
                job_id=job.job_id,
                pages=len(unrecognized),
            )
        if not any(p.text.strip() for p in final):
            raise with_code(
                PermanentInputError("Document has no extractable text"),
                "no_extractable_text",
            )
        text_parts, page_map, position = [], [], 0
        for page in final:
            part = page.text + "\n"
            text_parts.append(part)
            page_map.append(
                {
                    "page": page.page_number,
                    "start": position,
                    "end": position + len(part),
                    "method": page.method,
                }
            )
            position += len(part)
        text = "".join(text_parts).encode()
        mapping = json.dumps(
            {"schema_version": 1, "pages": page_map}, sort_keys=True
        ).encode()
        a = await self.store_bytes(
            base + "/text.txt", text, "extracted_text", "utf8-text"
        )
        b = await self.store_bytes(
            base + "/pages.json", mapping, "page_map", "page-map"
        )
        return source.sha256, (a, b), ()

    async def _analysis(self, job, source, pipeline, local, limits):
        """Classified pages; cached per build when recognition is needed, so a
        job resumed after waiting on a recognizer need not download again."""
        key = f"tenants/{job.tenant_id}/generations/{job.generation_id}/recognition/analysis.json"
        # Only an earlier attempt of this build can have saved one, so the
        # first attempt skips the lookup. Skipping is never wrong: analysis is
        # deterministic and a create-only save keeps the first copy.
        info = (
            await self.objects.stat(ObjectRef(bucket=self.settings.bucket, key=key))
            if job.attempt > 1
            else None
        )
        if info is not None:
            saved = json.loads(await self.read_bytes(info.ref, MAX_ARTIFACT_BYTES))
            if (saved.get("recipe"), saved.get("source_sha256")) != (
                pipeline.fingerprint,
                source.sha256,
            ):
                raise PermanentInputError(
                    "Saved page analysis does not match this build"
                )
            return [ExtractedPage(**p) for p in saved["pages"]]
        pages = []
        async with self.extractor.open_pages(
            await local.path(), spec=pipeline.extractor, limits=limits
        ) as stream:
            async for page in stream:
                pages.append(page)
        if any(p.needs_recognition for p in pages):
            data = json.dumps(
                {
                    "recipe": pipeline.fingerprint,
                    "source_sha256": source.sha256,
                    "pages": [
                        {
                            "page_number": p.page_number,
                            "text": p.text,
                            "method": p.method,
                            "needs_recognition": p.needs_recognition,
                        }
                        for p in pages
                    ],
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
            try:
                await self.store_bytes(key, data, "page_analysis", "page-analysis")
            except ValueError:
                pass  # another attempt stored it first; analysis is deterministic
        return pages

    async def _stage_job(self, job, stage):
        """Find the generation's single prerequisite job using durable identity."""
        jobs = await self.metadata.list_generation_jobs(
            tenant_id=job.tenant_id, generation_id=job.generation_id, stage=stage
        )
        if len(jobs) != 1 or jobs[0].status != JS.SUCCEEDED:
            raise ValueError("Prerequisite is not durably complete")
        manifest = await self.checkpoint_manifest(jobs[0])
        if not manifest:
            raise ValueError("Prerequisite checkpoint is missing")
        return manifest

    async def _chunk(self, job, source, pipeline):
        """Save bounded chunk batches, their manifests, and a sealed expected inventory."""
        extracted = await self._stage_job(job, ST.EXTRACT)
        a = self._output(extracted, "extracted_text")
        b = self._output(extracted, "page_map")
        text = await self.read_output(a, self.settings.max_extracted_characters * 4 + 4)
        pages = await self.read_output(b, MAX_PAGE_MAP_BYTES)
        base = f"tenants/{job.tenant_id}/generations/{job.generation_id}/attempts/{job.lease.lease_id}"
        batch = []
        entries = []
        ordinal = 0

        pending = []  # one upload task per batch, in batch order
        uploads = asyncio.Semaphore(CHUNK_UPLOADS)

        async def write(key, chunks, start):
            """Pin one batch's chunk payload and exact logical vector-ID inventory."""
            try:
                output = await self.store_bytes(
                    base + f"/chunks-{key}.json",
                    json.dumps(
                        encode(chunks), sort_keys=True, allow_nan=False
                    ).encode(),
                    "chunks",
                    "chunk-batch",
                )
                manifest = StageManifest(
                    tenant_id=job.tenant_id,
                    document_id=job.document_id,
                    generation_id=job.generation_id,
                    job_id=job.job_id,
                    stage=ST.CHUNK,
                    shard_key=key,
                    source_sha256=source.sha256,
                    input_sha256=extracted.sha256,
                    pipeline_fingerprint=pipeline.fingerprint,
                    outputs=(output,),
                    vector_ids=tuple(c.vector_id for c in chunks),
                )
                ref = await self.store_bytes(
                    base + f"/batch-{key}.json",
                    manifest.to_json().encode(),
                    "chunk_batch",
                    "stage-manifest",
                )
                return ChunkBatch(
                    batch_key=key,
                    start_ordinal=start,
                    end_ordinal=start + len(chunks),
                    manifest=ref,
                )
            finally:
                uploads.release()

        async def flush():
            """Start this batch's upload; waits while CHUNK_UPLOADS are in flight.

            The wait is the back-pressure: chunking pauses instead of holding
            more batches in memory. A failed upload stops chunking early.
            """
            nonlocal ordinal
            await uploads.acquire()
            failed = next(
                (t for t in pending if t.done() and not t.cancelled() and t.exception()),
                None,
            )
            if failed:
                uploads.release()
                raise failed.exception()
            pending.append(
                asyncio.ensure_future(
                    write(f"{len(pending):06d}", tuple(batch), ordinal)
                )
            )
            ordinal += len(batch)
            batch.clear()

        with tempfile.TemporaryDirectory(prefix="rag-chunks-") as temp:
            p = Path(temp) / "text"
            q = Path(temp) / "map"
            p.write_bytes(text)
            q.write_bytes(pages)
            extraction = ChunkingInput(
                text_path=p,
                page_map_path=q,
                extraction_sha256=a.sha256,
                page_map_sha256=b.sha256,
            )
            limits = ChunkingLimits(
                max_chunks=MAX_GENERATION_CHUNKS,
                max_chunk_characters=MAX_CHUNK_CHARACTERS,
                timeout_seconds=max(
                    1, min(240, job.budget.policy.max_attempt_seconds - 1)
                ),
            )
            async with self.chunker.open_chunks(
                extraction,
                source=source,
                generation=await self.metadata.get_generation(
                    tenant_id=job.tenant_id, generation_id=job.generation_id
                ),
                spec=pipeline.chunker,
                limits=limits,
            ) as chunks:
                try:
                    async for chunk in chunks:
                        batch.append(chunk)
                        if len(batch) >= self.settings.batch_size:
                            await flush()
                        if len(pending) > MAX_INVENTORY_BATCHES:
                            raise PermanentInputError(
                                "Batch inventory exceeds policy; increase chunk/batch sizes"
                            )
                    if batch:
                        await flush()
                    for task in pending:  # in batch order; raises the first failure
                        entries.append(await task)
                finally:
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
        if not ordinal:
            raise PermanentInputError("Document produced no searchable text")
        dimensions = json.loads(pipeline.embedder.options_json)["dimensions"]
        inv = GenerationInventory(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            source_version_id=source.source_version_id,
            generation_id=job.generation_id,
            source_sha256=source.sha256,
            pipeline_fingerprint=pipeline.fingerprint,
            vector_target=self.vectors.target_id,
            dimensions=dimensions,
            chunk_count=ordinal,
            vector_ids_sha256=compute_vector_id_digest(
                tenant_id=job.tenant_id,
                document_id=job.document_id,
                generation_id=job.generation_id,
                chunk_count=ordinal,
                max_chunks=MAX_GENERATION_CHUNKS,
            ),
            batches=tuple(entries),
        )
        data = inv.to_json().encode()
        if len(data) > MAX_INVENTORY_BYTES:
            raise PermanentInputError("Inventory exceeds size policy")
        ref = await self.store_bytes(
            base + "/inventory.json", data, "inventory", "generation-inventory"
        )
        await self.metadata.seal_generation_inventory(
            inventory=inv,
            inventory_ref=ref,
            chunking_job_id=job.job_id,
            worker_id=job.lease.worker_id,
            lease_id=job.lease.lease_id,
            sealing_id=uuid5(job.lease.lease_id, "seal"),
        )
        return extracted.sha256, (ref,), ()

    async def _read_segments(self, prefix, by_id, pipeline, plan):
        """Read this batch's segments 000.seg, 001.seg, ... (the progress cache).

        A segment is used only if it verifies completely: its own SHA-256
        trailer, this build's recipe and dimensions, and every chunk exactly one
        of this batch's chunks. A bad segment is skipped whole with a warning
        and its chunks are embedded again: the cache never fails a build.
        Storage errors still propagate. Returns (cache, used, next_number):
        used maps each contributing segment's ManifestOutput to its items,
        first segment wins per chunk.
        """
        cache, used, number = {}, [], 0
        while number < EMBEDDING_CACHE_ENTRIES:
            key = f"{prefix}{number:03d}.seg"
            info = await self.objects.stat(
                ObjectRef(bucket=self.settings.bucket, key=key)
            )
            if info is None:
                break
            role = f"embedding_segment_{number:03d}"
            number += 1
            data = await self.read_bytes(info.ref, MAX_ARTIFACT_BYTES)
            try:
                items = decode_segment(
                    data, recipe=pipeline.fingerprint, dimensions=plan.dimensions
                )
                if any(by_id.get(i.chunk.vector_id) != i.chunk for i in items):
                    raise ValueError("segment holds a chunk of another batch or build")
            except CACHE_DEFECTS as error:
                log_event(
                    log,
                    "embedding_cache_entry_ignored",
                    logging.WARNING,
                    key=key,
                    reason=type(error).__name__,
                )
                continue
            fresh = [i for i in items if i.chunk.vector_id not in cache]
            if not fresh:
                continue  # every chunk already provided by an earlier segment
            for item in fresh:
                cache[item.chunk.vector_id] = item
            used.append(
                ManifestOutput(
                    role=role,
                    bucket=info.ref.bucket,
                    object_key=info.ref.key,
                    object_version_id=info.ref.version_id,
                    sha256=hashlib.sha256(data).hexdigest(),
                    size_bytes=len(data),
                    format_name=SEGMENT_FORMAT,
                    format_version=SEGMENT_VERSION,
                )
            )
        return cache, used, number

    async def _write_segment(self, prefix, number, items, pipeline):
        """Write one attempt's new embeddings as one create-only segment.

        If a concurrent attempt already took this number, the next free number
        is used: segments are never overwritten. Returns its ManifestOutput.
        """
        data = encode_segment(items, recipe=pipeline.fingerprint)
        for candidate in range(number, EMBEDDING_CACHE_ENTRIES):
            try:
                return await self.store_bytes(
                    f"{prefix}{candidate:03d}.seg",
                    data,
                    f"embedding_segment_{candidate:03d}",
                    SEGMENT_FORMAT,
                    SEGMENT_VERSION,
                )
            except ValueError:  # number already holds another attempt's segment
                continue
        raise RetryableError("No free embedding segment number for this batch")

    async def _embed(self, job, source, pipeline):
        """Embed one batch, writing its vectors once into an immutable segment.

        The segment is both the progress cache (a retry reuses it and embeds
        only what is missing, so nothing is billed twice) and the stage
        output: the returned outputs are the segments that together cover the
        batch, which the lease-checked checkpoint then commits by pinned
        version and SHA-256. New embeddings are saved before any failure is
        raised. Typical cost: one segment write (retries also check the cache).
        """
        plan = await self._batch_plan(job)
        entry = plan.batch
        manifest = await self._batch_manifest(job, source, pipeline, entry)
        plan.validate_batch(manifest)
        chunks = decode(
            json.loads(await self.read_output(self._output(manifest, "chunks")))
        )
        if tuple(c.vector_id for c in chunks) != tuple(
            f"{job.tenant_id}:{job.document_id}:{job.generation_id}:{i}"
            for i in range(entry.start_ordinal, entry.end_ordinal)
        ):
            raise ValueError("Saved chunks differ from inventory")
        prefix = (
            f"tenants/{job.tenant_id}/generations/{job.generation_id}"
            f"/embedding-cache/{entry.batch_key}/"
        )
        by_id = {c.vector_id: c for c in chunks}
        if job.attempt > 1:
            cache, segments, next_number = await self._read_segments(
                prefix, by_id, pipeline, plan
            )
        else:
            # The first lease: no earlier attempt can have written a segment
            # (each lease increments the attempt). Writes are create-only, so
            # even then nothing could be overwritten.
            cache, segments, next_number = {}, [], 0
        missing = [c for c in chunks if c.vector_id not in cache]
        failures, succeeded = [], []
        if missing:
            result = await self.embedder.embed_batch(
                tuple(missing),
                pipeline=pipeline,
                dimensions=plan.dimensions,
                limits=EmbeddingLimits(
                    max_items=MAX_BATCH_ITEMS,
                    max_input_characters=MAX_CHUNK_CHARACTERS,
                    max_batch_tokens=self.settings.embedding_max_batch_tokens,
                    timeout_seconds=self.settings.embedding_timeout_seconds,
                ),
            )
            result.validate_for(
                tuple(missing),
                pipeline_fingerprint=pipeline.fingerprint,
                dimensions=plan.dimensions,
            )
            for item in result.items:
                if isinstance(item, EmbeddingFailure):
                    failures.append(item.error)
                else:
                    succeeded.append(item)
        new = succeeded
        if new:  # saved before any failure is raised
            try:
                segments.append(
                    await self._write_segment(prefix, next_number, new, pipeline)
                )
            except ValueError:
                raise with_code(
                    PermanentInputError(
                        "An embedding value cannot be stored as float32"
                    ),
                    "embedding_out_of_range",
                ) from None
        if failures:
            raise failures[0]
        written = {i.chunk.vector_id for i in new}
        if set(cache) | written != set(by_id):
            raise ValueError("Embedding segments do not cover the batch")
        return manifest.sha256, tuple(segments), tuple(c.vector_id for c in chunks)

    @staticmethod
    def _vector_ids(job, entry):
        """The exact logical IDs one sealed batch must produce, in ordinal order."""
        prefix = f"{job.tenant_id}:{job.document_id}:{job.generation_id}:"
        return tuple(
            prefix + str(i) for i in range(entry.start_ordinal, entry.end_ordinal)
        )

    async def _embed_jobs(self, job, inv):
        """Every sealed batch's embedding job, all succeeded; no missing or extra work."""
        jobs = await self.metadata.list_generation_jobs(
            tenant_id=job.tenant_id, generation_id=job.generation_id, stage=ST.EMBED
        )
        mapping = {j.shard_key: j for j in jobs}
        if set(mapping) != {b.batch_key for b in inv.batches}:
            raise ValueError("Embedding job set differs from sealed inventory")
        if any(j.status != JS.SUCCEEDED for j in mapping.values()):
            raise RetryableError("Embedding prerequisites are incomplete")
        return mapping

    async def _batch_items(self, job, inv, entry, embed_job):
        """One batch's committed embeddings, checked against the sealed plan."""
        manifest = await self.checkpoint_manifest(embed_job)
        expected = self._vector_ids(job, entry)
        items = await self._embedding_items(manifest, inv, expected)
        if tuple(i.chunk.vector_id for i in items) != expected or set(
            manifest.vector_ids
        ) != set(expected):
            raise ValueError("Embedding output identity mismatch")
        for item in items:
            if (
                item.pipeline_fingerprint != inv.pipeline_fingerprint
                or item.dimensions != inv.dimensions
            ):
                raise ValueError("Embedding output recipe mismatch")
        return items

    @staticmethod
    async def _read_ahead(loaders, depth=INDEX_READ_AHEAD):
        """Yield each loader's result in order, at most `depth` loads in flight.

        Memory is bounded by depth + 1 batches (the one being used plus those
        loading), never by document size. Unused loads are cancelled on exit.
        """
        pending = deque()
        loaders = iter(loaders)
        try:
            for loader in loaders:
                pending.append(asyncio.ensure_future(loader()))
                if len(pending) >= depth:
                    break
            while pending:
                result = await pending.popleft()
                loader = next(loaders, None)
                if loader is not None:
                    pending.append(asyncio.ensure_future(loader()))
                yield result
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _embedding_batches(self, job, inv, entries=None, mapping=None):
        """Yield (batch, items) for the given sealed batches (default: all), in order."""
        if mapping is None:
            mapping = await self._embed_jobs(job, inv)
        entries = inv.batches if entries is None else entries
        loaders = [
            partial(self._batch_items, job, inv, entry, mapping[entry.batch_key])
            for entry in entries
        ]
        async with aclosing(self._read_ahead(loaders)) as loaded:
            for entry in entries:
                yield entry, await anext(loaded)

    async def _embedding_items(self, manifest, inv, expected):
        """The batch's embeddings, exactly as its checkpointed manifest commits them.

        Every segment is read by its pinned version and must match its
        committed SHA-256 and size; segments merge in role order (first wins)
        and must cover exactly the expected chunks.
        """
        segments = sorted(manifest.outputs, key=lambda o: o.role)
        if not segments or any(
            not o.role.startswith("embedding_segment_")
            or (o.format_name, o.format_version) != (SEGMENT_FORMAT, SEGMENT_VERSION)
            for o in segments
        ):
            raise ValueError("Unsupported embedding output layout")
        parts = []
        for output in segments:
            data = await self.read_output(output)
            parts.append(
                decode_segment(
                    data, recipe=inv.pipeline_fingerprint, dimensions=inv.dimensions
                )
            )
        return merge_segments(parts, expected)

    def _digests(self):
        """The vector store's optional digest capability, or None."""
        expected = getattr(self.vectors, "expected_digest", None)
        stored = getattr(self.vectors, "stored_digests", None)
        return (expected, stored) if callable(expected) and callable(stored) else None

    async def _index(self, job, source, pipeline):
        """Upsert saved immutable vector batches; repeats use exactly the same IDs.

        With a digest-capable store, the receipt also records each batch's
        expected digest, computed from the verified segments, so publication
        can verify the stored rows without reading the vectors again.
        """
        inv = await self.inventory(job)
        if inv.vector_target != self.vectors.target_id:
            raise ValueError("Vector target changed")
        scope = VectorScope(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            generation_id=job.generation_id,
        )
        digests, batch_digests, chunk_digests = self._digests(), {}, {}
        async with aclosing(self._embedding_batches(job, inv)) as batches:
            async for entry, items in batches:
                chunks = tuple(i.chunk for i in items)
                # Text first: any vector that becomes searchable already has its text.
                await self.metadata.put_search_chunks(
                    tenant_id=job.tenant_id, chunks=chunks
                )
                await self._upsert(scope, items)
                if self.keyword is not None:
                    await self._index_keywords(scope, chunks)
                chunk_digests[entry.batch_key] = chunk_digest(
                    (c.vector_id, c.text) for c in chunks
                )
                if digests:
                    batch_digests[entry.batch_key] = digests[0](scope, items)
        receipt = {
            "schema_version": 3,
            "inventory_sha256": inv.sha256,
            "count": inv.chunk_count,
            "ids_sha256": inv.vector_ids_sha256,
            "batch_chunk_digests": chunk_digests,
            # Which keyword index (if any) holds this build's entries.
            "keyword_target": self.keyword.target_id if self.keyword is not None else None,
        }
        if digests:
            receipt["batch_digests"] = batch_digests
        output = await self.store_bytes(
            f"tenants/{job.tenant_id}/generations/{job.generation_id}/attempts/{job.lease.lease_id}/index-receipt.json",
            json.dumps(receipt, sort_keys=True).encode(),
            "index_receipt",
            "index-receipt",
        )
        return inv.sha256, (output,), ()

    async def _index_keywords(self, scope, chunks):
        """Write one batch to the keyword index, in pieces it accepts."""
        size = self.keyword.capabilities.max_batch_items
        documents = tuple(
            KeywordDocument(
                vector_id=c.vector_id,
                document_id=c.document_id,
                generation_id=c.generation_id,
                text=c.text,
                text_sha256=c.text_sha256,
            )
            for c in chunks
        )
        for start in range(0, len(documents), size):
            await self.keyword.index_batch(scope, items=documents[start : start + size])

    async def _upsert(self, scope, items):
        """Write one batch; an uncertain write must be proven READY by a fresh read."""
        try:
            results = await self.vectors.upsert_batch(scope, items=items)
        except UncertainOutcomeError:
            checks = await self.vectors.verify_batch(scope, expected=items)
            if (
                len(checks) != len(items)
                or {c.vector_id for c in checks} != {i.chunk.vector_id for i in items}
                or any(c.readiness != VectorReadiness.READY for c in checks)
            ):
                raise RetryableError(
                    "Uncertain vector write requires another stable-ID pass"
                ) from None
            return
        if len(results) != len(items) or {r.vector_id for r in results} != {
            i.chunk.vector_id for i in items
        }:
            raise ValueError("Vector adapter returned incomplete outcomes")
        for result in results:
            if result.error:
                raise result.error

    async def _index_receipt(self, job, inv):
        """The committed index receipt for this inventory.

        Read through the index job's checkpoint, pinned by version and SHA-256.
        It holds a chunk-text digest per batch, and a vector digest per batch
        when the vector store supports digests.
        """
        jobs = await self.metadata.list_generation_jobs(
            tenant_id=job.tenant_id, generation_id=job.generation_id, stage=ST.INDEX
        )
        index = next((j for j in jobs if j.status == JS.SUCCEEDED), None)
        manifest = await self.checkpoint_manifest(index) if index else None
        output = (
            next((o for o in manifest.outputs if o.role == "index_receipt"), None)
            if manifest
            else None
        )
        if output is None:
            raise ValueError("The index stage left no receipt")
        receipt = json.loads(await self.read_output(output, MAX_INDEX_RECEIPT_BYTES))
        if receipt.get("schema_version") != 3:
            raise ValueError("Unsupported index receipt version")
        if (
            receipt.get("inventory_sha256"),
            receipt.get("count"),
            receipt.get("ids_sha256"),
        ) != (inv.sha256, inv.chunk_count, inv.vector_ids_sha256):
            raise ValueError("Index receipt does not match the sealed inventory")
        return receipt

    @staticmethod
    def _batch_digests(receipt, inv, key):
        """One validated digest per sealed batch from receipt[key], or None if absent."""
        digests = receipt.get(key) if receipt else None
        if digests is None:
            return None
        if (
            not isinstance(digests, dict)
            or set(digests) != {b.batch_key for b in inv.batches}
            or any(
                not isinstance(d, str)
                or len(d) != 64
                or any(c not in "0123456789abcdef" for c in d)
                for d in digests.values()
            )
        ):
            raise ValueError("Index receipt batch digests are invalid")
        return digests

    async def _verify_search_chunks(self, job, inv, receipt):
        """Every chunk's text is stored exactly as indexed (receipts from version 3)."""
        expected = self._batch_digests(receipt, inv, "batch_chunk_digests")
        if expected is None:
            raise ValueError("Index receipt has no chunk-text digests")
        for group in self._digest_groups(job, inv.batches):
            stored = await self.metadata.search_chunk_digests(
                tenant_id=job.tenant_id,
                generation_id=job.generation_id,
                batches=tuple(self._vector_ids(job, b) for b in group),
            )
            if len(stored) != len(group):
                raise ValueError("Digest response has missing or extra batches")
            if any(
                digest != expected[entry.batch_key]
                for entry, digest in zip(group, stored, strict=True)
            ):
                raise with_code(
                    PermanentInputError("Stored chunk text differs from the sealed build"),
                    "search_chunks_mismatch",
                )

    async def _verify_keywords(self, job, inv, receipt, scope):
        """Every chunk is in the keyword index with its exact text (when one is used).

        A build indexed for another keyword target (the index was enabled or
        changed after it was indexed) still publishes: vectors are the source
        of truth. It is logged, and `rag-ingest reindex-keyword` adds it.
        """
        if self.keyword is None:
            return
        if receipt.get("keyword_target") != self.keyword.target_id:
            log_event(
                log,
                "keyword_index_stale",
                logging.WARNING,
                generation_id=str(job.generation_id),
                indexed_for=receipt.get("keyword_target"),
                current=self.keyword.target_id,
                hint="run `rag-ingest reindex-keyword`",
            )
            return
        for group in self._digest_groups(job, inv.batches):
            ids = [i for b in group for i in self._vector_ids(job, b)]
            hashes = await self.metadata.search_chunk_hashes(
                tenant_id=job.tenant_id, generation_id=job.generation_id, vector_ids=ids
            )
            if len(hashes) != len(ids):
                raise ValueError("Chunk rows missing for keyword verification")
            checks = await self.keyword.verify_batch(
                scope,
                expected=tuple(
                    KeywordExpectation(vector_id=i, text_sha256=hashes[i]) for i in ids
                ),
            )
            if len(checks) != len(ids) or {c.vector_id for c in checks} != set(ids):
                raise ValueError("Keyword verification has missing or extra IDs")
            if any(c.readiness == VectorReadiness.MISMATCH for c in checks):
                raise with_code(
                    PermanentInputError("Keyword index holds different text"),
                    "keyword_index_mismatch",
                )
            if any(c.readiness != VectorReadiness.READY for c in checks):
                raise RetryableError("Keyword index is not yet queryable")

    def _digest_groups(self, job, entries):
        """Group batches so each digest query names at most MAX_DIGEST_IDS IDs."""
        group, size = [], 0
        for entry in entries:
            n = entry.end_ordinal - entry.start_ordinal
            if group and size + n > MAX_DIGEST_IDS:
                yield group
                group, size = [], 0
            group.append(entry)
            size += n
        if group:
            yield group

    async def _publish(self, job, source, pipeline):
        """Verify every expected vector before the atomic active-generation switch.

        Fast path: compare each batch's stored-row digest (computed by the
        database) with the digest the index stage recorded from the verified
        segments. Any batch that differs, and every batch when no digests are
        available, is checked row by row with verify_batch, which classifies
        it exactly as before (MISMATCH is permanent, MISSING is retryable).
        """
        inv = await self.inventory(job)
        if inv.vector_target != self.vectors.target_id:
            raise ValueError("Vector target changed")
        scope = VectorScope(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            generation_id=job.generation_id,
        )
        mapping = await self._embed_jobs(job, inv)
        receipt = await self._index_receipt(job, inv)
        expected = (
            self._batch_digests(receipt, inv, "batch_digests")
            if self._digests()
            else None
        )
        count, unverified = 0, list(inv.batches)
        if expected is not None:
            unverified = []
            for group in self._digest_groups(job, inv.batches):
                stored = await self.vectors.stored_digests(
                    scope, batches=tuple(self._vector_ids(job, b) for b in group)
                )
                if len(stored) != len(group):
                    raise ValueError("Digest response has missing or extra batches")
                for entry, digest in zip(group, stored, strict=True):
                    if digest == expected[entry.batch_key]:
                        count += entry.end_ordinal - entry.start_ordinal
                    else:
                        unverified.append(entry)
            if unverified:
                log_event(
                    log,
                    "publish_digest_mismatch",
                    logging.WARNING,
                    generation_id=str(job.generation_id),
                    batches=len(unverified),
                )
        async with aclosing(
            self._embedding_batches(job, inv, unverified, mapping)
        ) as batches:
            async for _entry, items in batches:
                checks = await self.vectors.verify_batch(scope, expected=items)
                if len(checks) != len(items) or {c.vector_id for c in checks} != {
                    i.chunk.vector_id for i in items
                }:
                    raise ValueError(
                        "Verification response has missing or duplicate IDs"
                    )
                if any(c.readiness == VectorReadiness.MISMATCH for c in checks):
                    raise PermanentInputError(
                        "Indexed vector content differs from sealed build"
                    )
                if any(c.readiness != VectorReadiness.READY for c in checks):
                    raise RetryableError("Vector index is not yet queryable")
                count += len(items)
        if count != inv.chunk_count:
            raise ValueError("Publication count mismatch")
        await self._verify_search_chunks(job, inv, receipt)
        await self._verify_keywords(job, inv, receipt, scope)
        now = await self.metadata.now(job.tenant_id)
        await self.metadata.publish_generation(
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            generation_id=job.generation_id,
            job_id=job.job_id,
            worker_id=job.lease.worker_id,
            lease_id=job.lease.lease_id,
            publication_id=uuid5(job.lease.lease_id, "publication"),
            expected_manifest_sha256=inv.sha256,
            verified_chunk_count=count,
            verified_chunk_ids_sha256=inv.vector_ids_sha256,
            vector_target=inv.vector_target,
            verified_at=now,
        )

    async def aclose(self):
        """Close each distinct injected resource once."""
        seen = set()
        for resource in (
            self.metadata,
            self.objects,
            self.uploads,
            self.queue,
            self.extractor,
            self.chunker,
            self.embedder,
            self.vectors,
            *((self.recognizer,) if self.recognizer is not None else ()),
            *((self.keyword,) if self.keyword is not None else ()),
            *((self.reranker,) if self.reranker is not None else ()),
        ):
            if id(resource) not in seen:
                seen.add(id(resource))
                await resource.aclose()
