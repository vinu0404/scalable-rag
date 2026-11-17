"""Implement durable ingestion metadata with short tenant-serialized transactions.

All external I/O happens in services, outside database locks. The operation
ledger resolves lost responses without repeating successful state changes.
"""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from uuid import uuid4, uuid5
import sqlalchemy as sa
from . import schema as s
from .codec import encode, decode
from rag_ingest.contracts.errors import (
    DuplicateContentError,
    LeaseLostError,
    PermanentInputError,
    RetryableError,
    with_code,
)
from rag_ingest.domain.enums import JobStatus as JS, ProcessingStage as ST
from rag_ingest.domain.jobs import Job, JobLease
from rag_ingest.domain.events import JobAvailable
from rag_ingest.domain.outbox import OutboxRecord, PublisherClaim
from rag_ingest.domain.uploads import UploadStatus as US
from rag_ingest.domain.policy import ExecutionBudget
from rag_ingest.domain.limits import MAX_DIGEST_IDS, MAX_GENERATION_JOBS
from rag_ingest.domain.documents import normalize_metadata, normalize_tags
from rag_ingest.domain.search import SearchResult

_MISSING = object()


def binding(value):
    """Hash a request's complete typed payload for safe idempotency-key reuse."""
    return hashlib.sha256(
        json.dumps(
            encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _report(row):
    """Shape a job row as safe status: no lease tokens or source contents."""
    job = decode(row["payload"])
    return {
        "job_id": str(job.job_id),
        "document_id": str(job.document_id),
        "generation_id": str(job.generation_id),
        "stage": job.stage.value,
        "status": job.status.value,
        "attempt": job.attempt,
        "next_attempt_at": job.next_attempt_at.isoformat()
        if job.next_attempt_at
        else None,
        "failure_code": row["failure_code"],
    }


class PostgresMetadataStore:
    """Persist the core contract; use one instance per process and event loop."""

    def __init__(self, database):
        """Receive a Database so the application owns connection lifecycle."""
        self.db = database

    async def _row(self, c, t, tenant, id):
        """Read one tenant-scoped database row inside the caller's transaction."""
        return (
            (await c.execute(sa.select(t).where(t.c.tenant_id == tenant, t.c.id == id)))
            .mappings()
            .first()
        )

    async def _get(self, c, t, tenant, id):
        """Restore one validated domain snapshot or report absence."""
        row = await self._row(c, t, tenant, id)
        return decode(row["payload"]) if row else None

    async def _save(self, c, t, tenant, id, obj, **extra):
        """Update snapshots and indexed state together under the tenant lock."""
        values = {"payload": encode(obj), **extra}
        name = type(obj).__name__
        if name == "Document":
            values.update(
                active_id=obj.active_generation_id,
                desired_id=obj.desired_generation_id,
                deleted_at=obj.deleted_at,
                tags=list(obj.tags),
                metadata=obj.metadata_dict,
                created_at=obj.created_at,
            )
        elif name == "UploadSession":
            values.update(
                document_id=obj.document_id,
                request_key=obj.request_key,
                status=obj.status.value,
                expires_at=obj.expires_at,
            )
        elif name == "ContentArtifact":
            values.update(sha256=obj.sha256, size_bytes=obj.size_bytes)
        elif name == "SourceVersion":
            values.update(document_id=obj.document_id, artifact_id=obj.artifact_id)
        elif name == "ProcessingGeneration":
            values.update(document_id=obj.document_id, source_id=obj.source_version_id)
        elif name == "Job":
            values.update(
                document_id=obj.document_id,
                generation_id=obj.generation_id,
                stage=obj.stage.value,
                shard=obj.shard_key,
                status=obj.status.value,
                due_at=obj.next_attempt_at,
                deadline_at=obj.budget.deadline_at,
                lease_id=obj.lease.lease_id if obj.lease else None,
                lease_expires_at=obj.lease.expires_at if obj.lease else None,
            )
        elif name == "OutboxRecord":
            values.update(
                job_id=obj.event.job_id,
                available_at=obj.available_at,
                published_at=obj.published_at,
                claim_id=obj.claim.claim_id if obj.claim else None,
                claim_expires_at=obj.claim.expires_at if obj.claim else None,
            )
        elif name == "StageCheckpoint":
            values.update(job_id=obj.job_id)
        elif name == "MultipartPlan":
            values.update(upload_id=obj.upload_id)
        if await self._row(c, t, tenant, id):
            await c.execute(
                t.update().where(t.c.tenant_id == tenant, t.c.id == id).values(**values)
            )
        else:
            await c.execute(t.insert().values(tenant_id=tenant, id=id, **values))

    async def _replay(self, c, tenant, id, kind, payload):
        """Return the original committed result and reject conflicting requests."""
        row = (
            (
                await c.execute(
                    sa.select(s.operations).where(
                        s.operations.c.tenant_id == tenant, s.operations.c.id == id
                    )
                )
            )
            .mappings()
            .first()
        )
        if not row:
            return _MISSING
        if row["kind"] != kind or row["binding"] != binding(payload):
            raise ValueError("Operation identity reused with different data")
        return decode(row["result"])

    async def _remember(self, c, tenant, id, kind, payload, result):
        """Commit request identity, payload binding, and result with its mutation."""
        await c.execute(
            s.operations.insert().values(
                tenant_id=tenant,
                id=id,
                kind=kind,
                binding=binding(payload),
                result=encode(result),
            )
        )

    async def now(self, tenant_id):
        """Obtain database time for fixed budgets; later writes still recheck it."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            return now

    async def tenants(self):
        """List configured tenant IDs for background processing, without content."""
        async with self.db.transaction(None, lock=False) as (c, now):
            return tuple(
                (
                    await c.execute(sa.select(s.documents.c.tenant_id).distinct())
                ).scalars()
            )

    async def _read(self, t, tenant, id):
        """Read a tenant-scoped snapshot without granting mutation authority."""
        async with self.db.transaction(tenant, lock=False) as (c, now):
            return await self._get(c, t, tenant, id)

    # ---- search chunks: one copy of each chunk's text ------------------------

    async def put_search_chunks(self, *, tenant_id, chunks):
        """Store each chunk's text once; never replace different text under an ID.

        Idempotent per stable vector ID (a retry of the index stage writes the
        same rows). One insert and one read-back per batch.
        """
        if not isinstance(chunks, tuple) or not 1 <= len(chunks) <= MAX_DIGEST_IDS:
            raise ValueError("Chunk batch must contain 1..8192 chunks")
        rows = []
        for c in chunks:
            if c.tenant_id != tenant_id:
                raise ValueError("Chunk belongs to another tenant")
            if "\x00" in c.text:
                # PostgreSQL text cannot hold NUL; the driver refuses it before
                # sending, so classify it here as the permanent input problem it is.
                raise with_code(
                    PermanentInputError("PostgreSQL rejected data it cannot store"),
                    "unstorable_data",
                )
            rows.append(
                (
                    c.vector_id,
                    str(c.document_id),
                    str(c.generation_id),
                    c.ordinal,
                    c.text,
                    json.dumps(
                        [
                            {
                                "page_number": sp.page_number,
                                "start_char": sp.start_char,
                                "end_char": sp.end_char,
                            }
                            for sp in c.spans
                        ]
                    ),
                    c.text_sha256,
                )
            )
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError("Duplicate chunk identities")
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            await c.execute(
                sa.text("""INSERT INTO search_chunks
                    (tenant_id, vector_id, document_id, generation_id, ordinal, content, spans, text_sha256)
                SELECT :tenant, r.id, r.doc, r.gen, r.ord, r.content, CAST(r.spans AS jsonb), r.hash
                FROM unnest(CAST(:ids AS text[]), CAST(:docs AS uuid[]), CAST(:gens AS uuid[]),
                    CAST(:ords AS int[]), CAST(:contents AS text[]), CAST(:spans AS text[]),
                    CAST(:hashes AS text[])) AS r(id, doc, gen, ord, content, spans, hash)
                ON CONFLICT (tenant_id, vector_id) DO NOTHING"""),
                dict(
                    tenant=tenant_id,
                    ids=[r[0] for r in rows],
                    docs=[r[1] for r in rows],
                    gens=[r[2] for r in rows],
                    ords=[r[3] for r in rows],
                    contents=[r[4] for r in rows],
                    spans=[r[5] for r in rows],
                    hashes=[r[6] for r in rows],
                ),
            )
            stored = dict(
                (
                    await c.execute(
                        sa.text(
                            "SELECT vector_id, text_sha256 FROM search_chunks"
                            " WHERE tenant_id = :tenant AND vector_id = ANY(CAST(:ids AS text[]))"
                        ),
                        dict(tenant=tenant_id, ids=[r[0] for r in rows]),
                    )
                ).all()
            )
        for vector_id, *_, text_hash in rows:
            if stored.get(vector_id) != text_hash:
                raise PermanentInputError("Stored chunk text differs from this build")

    async def search_chunk_digests(self, *, tenant_id, generation_id, batches):
        """chunk_digest() of each ID group's stored rows, recomputed from the text."""
        ids, groups = [], []
        for number, batch in enumerate(batches):
            ids.extend(batch)
            groups.extend([number] * len(batch))
        if len(ids) != len(set(ids)) or len(ids) > MAX_DIGEST_IDS:
            raise ValueError("Digest request must name at most 8192 unique IDs")
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = await c.execute(
                sa.text("""SELECT w.grp, encode(sha256(convert_to(string_agg(
                        sc.vector_id || E'\t' || encode(sha256(convert_to(sc.content, 'UTF8')), 'hex')
                        || E'\n', '' ORDER BY sc.vector_id COLLATE "C"), 'UTF8')), 'hex')
                    FROM unnest(CAST(:ids AS text[]), CAST(:grps AS int[])) AS w(vector_id, grp)
                    JOIN search_chunks sc ON sc.tenant_id = :tenant AND sc.vector_id = w.vector_id
                        AND sc.generation_id = :gen
                    GROUP BY w.grp"""),
                dict(ids=ids, grps=groups, tenant=tenant_id, gen=generation_id),
            )
            found = {grp: digest for grp, digest in rows}
        return tuple(found.get(n) for n in range(len(batches)))

    async def search_chunk_hashes(self, *, tenant_id, generation_id, vector_ids):
        """{vector_id: text_sha256} of this build's stored chunks among vector_ids."""
        if len(vector_ids) > MAX_DIGEST_IDS:
            raise ValueError("At most 8192 IDs")
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            return dict(
                (
                    await c.execute(
                        sa.text(
                            "SELECT vector_id, text_sha256 FROM search_chunks WHERE tenant_id = :tenant"
                            " AND generation_id = :gen AND vector_id = ANY(CAST(:ids AS text[]))"
                        ),
                        dict(tenant=tenant_id, gen=generation_id, ids=list(vector_ids)),
                    )
                ).all()
            )

    async def live_builds(self, *, after=None, limit=100):
        """Published builds of live documents across tenants, in a stable order.

        Returns (tenant_id, document_id, generation_id) tuples after the
        `after` tuple (keyset paging), for maintenance such as reindexing.
        """
        self._limit(limit)
        params = dict(limit=limit)
        where = ""
        if after is not None:
            where = " AND (tenant_id, id) > (CAST(:t AS uuid), CAST(:d AS uuid))"
            params.update(t=str(after[0]), d=str(after[1]))
        async with self.db.transaction(None, lock=False) as (c, now):
            rows = await c.execute(
                sa.text(
                    "SELECT tenant_id, id, active_id FROM documents"
                    " WHERE deleted_at IS NULL AND active_id IS NOT NULL"
                    + where
                    + " ORDER BY tenant_id, id LIMIT :limit"
                ),
                params,
            )
            return tuple((t, d, g) for t, d, g in rows)

    async def is_live_build(self, *, tenant_id, document_id, generation_id):
        """Whether this build is still the active one of a live document."""
        doc = await self.get_document(tenant_id=tenant_id, document_id=document_id)
        return bool(doc and not doc.deleted_at and doc.active_generation_id == generation_id)

    async def search_chunk_page(self, *, tenant_id, generation_id, after=None, limit=500):
        """One page of a build's chunks: (vector_id, document_id, content, text_sha256)."""
        self._limit(limit)
        params = dict(tenant=tenant_id, gen=generation_id, limit=limit, after=after or "")
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = await c.execute(
                sa.text(
                    "SELECT vector_id, document_id, content, text_sha256 FROM search_chunks"
                    " WHERE tenant_id = :tenant AND generation_id = :gen"
                    " AND vector_id COLLATE \"C\" > :after COLLATE \"C\""
                    " ORDER BY vector_id COLLATE \"C\" LIMIT :limit"
                ),
                params,
            )
            return tuple(tuple(r) for r in rows)

    async def delete_search_chunks(self, *, tenant_id, generation_id):
        """Erase one build's chunk text (document deletion or retired build)."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            await c.execute(
                s.search_chunks.delete().where(
                    s.search_chunks.c.tenant_id == tenant_id,
                    s.search_chunks.c.generation_id == generation_id,
                )
            )

    @staticmethod
    def _filter_sql(filters, params):
        """SQL conditions on documents alias d for one SearchFilters."""
        sql = ""
        if filters is None:
            return sql
        if filters.document_ids:
            sql += " AND d.id = ANY(CAST(:f_docs AS uuid[]))"
            params["f_docs"] = [str(d) for d in filters.document_ids]
        if filters.tags_any:
            sql += " AND d.tags && CAST(:f_any AS text[])"
            params["f_any"] = list(filters.tags_any)
        if filters.tags_all:
            sql += " AND d.tags @> CAST(:f_all AS text[])"
            params["f_all"] = list(filters.tags_all)
        if filters.metadata:
            sql += " AND d.metadata @> CAST(:f_meta AS jsonb)"
            params["f_meta"] = json.dumps(dict(filters.metadata))
        if filters.created_after:
            sql += " AND d.created_at >= :f_after"
            params["f_after"] = filters.created_after
        if filters.created_before:
            sql += " AND d.created_at < :f_before"
            params["f_before"] = filters.created_before
        return sql

    _LIVE = (
        " d.deleted_at IS NULL AND d.active_id IS NOT NULL"
        " AND d.payload->'fields'->>'acl_ref' = 'tenant'"
    )

    async def filter_documents(self, *, tenant_id, filters, limit):
        """Live documents matching the filters, at most limit (bounded)."""
        self._limit(limit)
        params = dict(tenant=tenant_id, limit=limit)
        where = self._filter_sql(filters, params)
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = await c.execute(
                sa.text(
                    "SELECT d.id FROM documents d WHERE d.tenant_id = :tenant AND"
                    + self._LIVE + where + " ORDER BY d.id LIMIT :limit"
                ),
                params,
            )
            return tuple(r[0] for r in rows)

    async def hydrate_search_hits(self, *, tenant_id, hits, filters=None):
        """Results for index hits that belong to live documents passing the filters.

        Keeps the hits' order. A hit is dropped unless its document is in this
        tenant, not deleted, tenant-visible and its build is the active one.
        A live hit with no chunk row is returned with text None (the pipeline
        drops and logs it).
        """
        if not hits:
            return ()
        params = dict(
            tenant=tenant_id,
            ids=[h.vector_id for h in hits],
            docs=[str(h.document_id) for h in hits],
            gens=[str(h.generation_id) for h in hits],
            ords=list(range(len(hits))),
        )
        where = self._filter_sql(filters, params)
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = (
                await c.execute(
                    sa.text(
                        """SELECT h.ord, h.vector_id, h.document_id, h.generation_id,
                            d.payload->'fields'->>'display_name' AS filename,
                            sc.content, sc.spans, sc.ordinal, sc.text_sha256
                        FROM unnest(CAST(:ids AS text[]), CAST(:docs AS uuid[]),
                            CAST(:gens AS uuid[]), CAST(:ords AS int[]))
                            AS h(vector_id, document_id, generation_id, ord)
                        JOIN documents d ON d.tenant_id = :tenant AND d.id = h.document_id
                            AND d.active_id = h.generation_id
                        LEFT JOIN search_chunks sc ON sc.tenant_id = :tenant
                            AND sc.vector_id = h.vector_id AND sc.generation_id = h.generation_id
                        WHERE"""
                        + self._LIVE
                        + where
                        + " ORDER BY h.ord"
                    ),
                    params,
                )
            ).mappings().all()
        results = []
        for r in rows:
            hit = hits[r["ord"]]
            missing = r["content"] is None
            results.append(
                SearchResult(
                    vector_id=hit.vector_id,
                    document_id=hit.document_id,
                    generation_id=hit.generation_id,
                    filename=r["filename"],
                    ordinal=-1 if missing else r["ordinal"],
                    text=None if missing else r["content"],
                    spans=() if missing else tuple(r["spans"]),
                    score=hit.score,
                    text_sha256="" if missing else r["text_sha256"],
                )
            )
        return tuple(results)

    async def update_document_labels(self, *, tenant_id, document_id, tags=None, metadata=None):
        """Replace a live document's tags and/or metadata; None leaves one unchanged."""
        async with self.db.transaction(tenant_id) as (c, now):
            doc = await self._get(c, s.documents, tenant_id, document_id)
            if not doc or doc.deleted_at:
                return None
            changed = replace(
                doc,
                tags=normalize_tags(list(tags)) if tags is not None else doc.tags,
                metadata=normalize_metadata(metadata) if metadata is not None else doc.metadata,
            )
            if changed != doc:
                await self._save(c, s.documents, tenant_id, document_id, changed)
            return changed

    async def get_document(self, *, tenant_id, document_id):
        """Return live or tombstoned document state."""
        return await self._read(s.documents, tenant_id, document_id)

    async def get_upload(self, *, tenant_id, upload_id):
        """Return upload state for status reporting and recovery."""
        return await self._read(s.uploads, tenant_id, upload_id)

    async def get_job(self, *, tenant_id, job_id):
        """Return a job snapshot, not ownership."""
        return await self._read(s.jobs, tenant_id, job_id)

    async def get_source_version(self, *, tenant_id, source_version_id):
        """Load the exact immutable source registered for a generation."""
        return await self._read(s.sources, tenant_id, source_version_id)

    async def get_generation(self, *, tenant_id, generation_id):
        """Read source and pipeline identity for a build."""
        return await self._read(s.generations, tenant_id, generation_id)

    async def _column(self, t, tenant, id, column):
        """Restore an auxiliary typed value stored alongside a snapshot."""
        async with self.db.transaction(tenant, lock=False) as (c, now):
            row = await self._row(c, t, tenant, id)
            return decode(row[column]) if row and row[column] is not None else None

    async def get_upload_pipeline(self, *, tenant_id, upload_id):
        """Restore the recipe chosen at initiation, never today's defaults."""
        return await self._column(s.uploads, tenant_id, upload_id, "pipeline")

    async def get_generation_pipeline(self, *, tenant_id, generation_id):
        """Restore a generation's durable processing recipe."""
        return await self._column(s.generations, tenant_id, generation_id, "pipeline")

    async def get_verification_ref(self, *, tenant_id, upload_id):
        """Read the exact protected object pinned for verification."""
        return await self._column(s.uploads, tenant_id, upload_id, "verification_ref")

    async def _live_duplicate(self, c, tenant_id, sha256):
        """The live document that already has these exact bytes, or None.

        Live: not deleted, and its published build or its current non-failed
        build uses this content. A failed build does not count, so a file
        whose processing failed can be uploaded again.
        """
        row = (
            await c.execute(
                sa.text(
                    """SELECT d.payload FROM content_artifacts a
                    JOIN source_versions sv ON sv.tenant_id = a.tenant_id AND sv.artifact_id = a.id
                    JOIN generations g ON g.tenant_id = sv.tenant_id AND g.source_id = sv.id
                    JOIN documents d ON d.tenant_id = g.tenant_id AND d.id = g.document_id
                    WHERE a.tenant_id = :tenant AND a.sha256 = :sha256 AND a.state = 'active'
                      AND d.deleted_at IS NULL
                      AND (d.active_id = g.id OR (d.desired_id = g.id AND g.status <> 'failed'))
                    ORDER BY d.id LIMIT 1"""
                ),
                {"tenant": tenant_id, "sha256": sha256},
            )
        ).scalar_one_or_none()
        return decode(row) if row is not None else None

    async def initiate_upload(
        self,
        *,
        upload,
        pipeline,
        request_fingerprint,
        expected_desired_generation_id,
        new_document=None,
    ):
        """Idempotently create intake and optional document in one transaction."""
        async with self.db.transaction(upload.tenant_id) as (c, now):
            row = (
                (
                    await c.execute(
                        sa.select(s.uploads).where(
                            s.uploads.c.tenant_id == upload.tenant_id,
                            s.uploads.c.request_key == upload.request_key,
                        )
                    )
                )
                .mappings()
                .first()
            )
            if row:
                if row["request_fingerprint"] != request_fingerprint:
                    raise ValueError("Upload request key reused with different details")
                return decode(row["payload"])
            if upload.expected_sha256 and not upload.allow_duplicate:
                # Checked under the tenant lock that also serializes acceptance.
                dup = await self._live_duplicate(c, upload.tenant_id, upload.expected_sha256)
                if dup:
                    raise DuplicateContentError(dup.document_id, dup.display_name)
            doc = await self._get(c, s.documents, upload.tenant_id, upload.document_id)
            if new_document:
                if (
                    doc
                    or new_document.tenant_id != upload.tenant_id
                    or new_document.document_id != upload.document_id
                ):
                    raise ValueError("Invalid new document")
                doc = new_document
                await self._save(c, s.documents, doc.tenant_id, doc.document_id, doc)
            if (
                not doc
                or doc.deleted_at
                or doc.desired_generation_id != expected_desired_generation_id
            ):
                raise ValueError("Document intake state changed")
            if upload.status != US.UPLOADING or upload.expires_at <= now:
                raise ValueError("Invalid upload session")
            await self._save(
                c,
                s.uploads,
                upload.tenant_id,
                upload.upload_id,
                upload,
                pipeline=encode(pipeline),
                request_fingerprint=request_fingerprint,
                baseline=expected_desired_generation_id,
            )
            return upload

    async def abandoned_uploads(self, *, tenant_id, grace_seconds, limit=100):
        """Never-accepted uploads whose stored files can now be erased.

        Failed (refused duplicate, verification failure) or expired uploads,
        files not yet erased, and expired more than grace_seconds ago. Grants
        never outlive expires_at, so after the grace no client PUT can still
        be writing. Returns (upload, multipart plan or None) pairs.
        """
        self._limit(limit)
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(s.uploads.c.payload)
                        .where(
                            s.uploads.c.tenant_id == tenant_id,
                            s.uploads.c.status.in_(["failed", "expired"]),
                            s.uploads.c.files_removed_at.is_(None),
                            s.uploads.c.expires_at
                            <= now - timedelta(seconds=grace_seconds),
                        )
                        .order_by(s.uploads.c.expires_at)
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            result = []
            for payload in rows:
                u = decode(payload)
                if u.source_version_id is not None:  # never an accepted file
                    continue
                plan = await self._get(c, s.plans, tenant_id, u.upload_id)
                result.append((u, plan))
            return tuple(result)

    async def mark_upload_files_removed(self, *, tenant_id, upload_id):
        """Record the erasure once; False if the upload is not abandoned or done."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            r = await c.execute(
                s.uploads.update()
                .where(
                    s.uploads.c.tenant_id == tenant_id,
                    s.uploads.c.id == upload_id,
                    s.uploads.c.status.in_(["failed", "expired"]),
                    s.uploads.c.files_removed_at.is_(None),
                )
                .values(files_removed_at=now)
            )
            return r.rowcount == 1

    async def _eligible_upload(self, c, tenant, id, now, status=US.UPLOADING):
        """Check intake against current document deletion and deadline state."""
        u = await self._get(c, s.uploads, tenant, id)
        if not u or u.status != status or u.expires_at <= now:
            return None
        d = await self._get(c, s.documents, tenant, u.document_id)
        return u if d and not d.deleted_at else None

    async def attach_multipart_plan(self, plan):
        """Attach only one winning transfer and fixed byte plan."""
        async with self.db.transaction(plan.tenant_id) as (c, now):
            u = await self._eligible_upload(c, plan.tenant_id, plan.upload_id, now)
            if not u or plan.file_size_bytes != u.expected_size_bytes:
                return False
            old = await self._get(c, s.plans, plan.tenant_id, plan.upload_id)
            if old:
                return old == plan
            await self._save(c, s.plans, plan.tenant_id, plan.upload_id, plan)
            await self._save(
                c,
                s.uploads,
                u.tenant_id,
                u.upload_id,
                replace(u, multipart_upload_id=plan.provider_upload_id),
            )
            return True

    async def get_multipart_plan(self, *, tenant_id, upload_id):
        """Read the fixed plan attached to this upload."""
        return await self._read(s.plans, tenant_id, upload_id)

    async def record_observed_part(self, part):
        """Keep provider observations scoped to the winning active transfer."""
        async with self.db.transaction(part.tenant_id) as (c, now):
            u = await self._eligible_upload(c, part.tenant_id, part.upload_id, now)
            if not u:
                return False
            plan = await self._get(c, s.plans, part.tenant_id, part.upload_id)
            if not plan:
                return False
            part.validate_against(plan)
            where = (
                s.parts.c.tenant_id == part.tenant_id,
                s.parts.c.upload_id == part.upload_id,
                s.parts.c.provider_id == part.provider_upload_id,
                s.parts.c.part_number == part.part_number,
            )
            row = (await c.execute(sa.select(s.parts).where(*where))).mappings().first()
            if row:
                old = decode(row["payload"])
                if old == part:
                    return True
                if old.observed_at >= part.observed_at:
                    return False
                await c.execute(
                    s.parts.update().where(*where).values(payload=encode(part))
                )
            else:
                await c.execute(
                    s.parts.insert().values(
                        tenant_id=part.tenant_id,
                        upload_id=part.upload_id,
                        provider_id=part.provider_upload_id,
                        part_number=part.part_number,
                        payload=encode(part),
                    )
                )
            return True

    async def list_observed_parts(
        self, *, tenant_id, upload_id, provider_upload_id, after_part_number, limit
    ):
        """Read one bounded ordered page of cached part observations."""
        self._limit(limit)
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            q = (
                sa.select(s.parts.c.payload)
                .where(
                    s.parts.c.tenant_id == tenant_id,
                    s.parts.c.upload_id == upload_id,
                    s.parts.c.provider_id == provider_upload_id,
                    s.parts.c.part_number > (after_part_number or 0),
                )
                .order_by(s.parts.c.part_number)
                .limit(limit)
            )
            return tuple(decode(v) for v in (await c.execute(q)).scalars())

    async def begin_verification(self, *, tenant_id, upload_id, ref):
        """Freeze exact completed bytes before hashing outside the transaction."""
        async with self.db.transaction(tenant_id) as (c, now):
            row = await self._row(c, s.uploads, tenant_id, upload_id)
            if not row:
                return False
            u = decode(row["payload"])
            if u.status == US.VERIFYING:
                return decode(row["verification_ref"]) == ref
            u = await self._eligible_upload(c, tenant_id, upload_id, now)
            if not u or (u.bucket, u.object_key) != (ref.bucket, ref.key):
                return False
            await self._save(
                c,
                s.uploads,
                tenant_id,
                upload_id,
                replace(u, status=US.VERIFYING),
                verification_ref=encode(ref),
            )
            return True

    async def find_content_artifact(self, *, tenant_id, sha256):
        """Use the tenant/hash unique index rather than scanning files."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            value = (
                await c.execute(
                    sa.select(s.artifacts.c.payload).where(
                        s.artifacts.c.tenant_id == tenant_id,
                        s.artifacts.c.sha256 == sha256,
                        s.artifacts.c.state == "active",
                    )
                )
            ).scalar_one_or_none()
            return decode(value) if value else None

    async def _insert_job(self, c, job, outbox):
        """Validate a new job's notification and register both atomically."""
        if (
            job.status != JS.QUEUED
            or job.attempt
            or job.lease
            or outbox.event.tenant_id != job.tenant_id
            or outbox.event.job_id != job.job_id
            or outbox.claim
            or outbox.published_at
        ):
            raise ValueError("Invalid queued job/outbox pair")
        await self._save(c, s.jobs, job.tenant_id, job.job_id, job)
        await self._save(c, s.outbox, job.tenant_id, outbox.event.event_id, outbox)

    async def accept_verified_upload(
        self,
        *,
        tenant_id,
        upload_id,
        verified_ref,
        artifact_candidate,
        source,
        generation,
        first_job,
        outbox,
    ):
        """Deduplicate verified bytes and commit complete durable processing intake."""
        async with self.db.transaction(tenant_id) as (c, now):
            row = await self._row(c, s.uploads, tenant_id, upload_id)
            if not row:
                return None
            u = decode(row["payload"])
            if u.status == US.FAILED and u.duplicate_of:
                return u  # replay of a duplicate refusal
            if u.status == US.ACCEPTED:
                old = await self._get(c, s.sources, tenant_id, u.source_version_id)
                if decode(row["verification_ref"]) != verified_ref or (
                    old.sha256,
                    old.size_bytes,
                ) != (source.sha256, source.size_bytes):
                    raise ValueError("Conflicting acceptance replay")
                return u
            u = await self._eligible_upload(c, tenant_id, upload_id, now, US.VERIFYING)
            if not u or decode(row["verification_ref"]) != verified_ref:
                return None
            doc = await self._get(c, s.documents, tenant_id, u.document_id)
            if doc.desired_generation_id != row["baseline"]:
                return None
            pipeline = decode(row["pipeline"])
            if any(
                x.tenant_id != tenant_id
                for x in (artifact_candidate, source, generation, first_job)
            ):
                raise ValueError("Cross-tenant registration")
            if (
                source.document_id != u.document_id
                or generation.document_id != u.document_id
                or generation.source_version_id != source.source_version_id
                or first_job.document_id != u.document_id
                or first_job.generation_id != generation.generation_id
            ):
                raise ValueError("Registration identity mismatch")
            if (source.bucket, source.object_key, source.object_version_id) != (
                verified_ref.bucket,
                verified_ref.key,
                verified_ref.version_id,
            ):
                raise ValueError("Source differs from frozen object")
            if source.size_bytes != u.expected_size_bytes or (
                u.expected_sha256 and source.sha256 != u.expected_sha256
            ):
                raise ValueError("Verified content does not match intake")
            if (
                (artifact_candidate.sha256, artifact_candidate.size_bytes)
                != (source.sha256, source.size_bytes)
                or generation.pipeline_fingerprint != pipeline.fingerprint
                or first_job.stage != ST.EXTRACT
            ):
                raise ValueError("Content or recipe mismatch")
            arow = (
                (
                    await c.execute(
                        sa.select(s.artifacts).where(
                            s.artifacts.c.tenant_id == tenant_id,
                            s.artifacts.c.sha256 == source.sha256,
                        )
                    )
                )
                .mappings()
                .first()
            )
            if arow and arow["state"] == "deleting":
                raise RetryableError("Matching artifact cleanup is still in progress")
            artifact = (
                decode(arow["payload"])
                if arow and arow["state"] == "active"
                else replace(
                    artifact_candidate, artifact_id=decode(arow["payload"]).artifact_id
                )
                if arow
                else artifact_candidate
            )
            if artifact.size_bytes != source.size_bytes:
                raise ValueError("Hash identity has incompatible size")
            if not u.allow_duplicate:
                dup = await self._live_duplicate(c, tenant_id, source.sha256)
                if dup:
                    refused = replace(u, status=US.FAILED, duplicate_of=dup.document_id)
                    await self._save(
                        c,
                        s.uploads,
                        tenant_id,
                        upload_id,
                        refused,
                        failure_code="duplicate_content",
                    )
                    return refused
            if not arow or arow["state"] == "deleted":
                await self._save(
                    c,
                    s.artifacts,
                    tenant_id,
                    artifact.artifact_id,
                    artifact,
                    state="active",
                )
            source = replace(source, artifact_id=artifact.artifact_id)
            await self._save(c, s.sources, tenant_id, source.source_version_id, source)
            await self._save(
                c,
                s.generations,
                tenant_id,
                generation.generation_id,
                generation,
                pipeline=encode(pipeline),
                deadline_at=first_job.budget.generation_deadline_at,
            )
            await self._insert_job(c, first_job, outbox)
            await self._save(
                c,
                s.documents,
                tenant_id,
                doc.document_id,
                replace(doc, desired_generation_id=generation.generation_id),
            )
            result = replace(
                u, status=US.ACCEPTED, source_version_id=source.source_version_id
            )
            await self._save(c, s.uploads, tenant_id, upload_id, result)
            return result

    async def reject_verified_upload(
        self, *, tenant_id, upload_id, verified_ref, reason_code
    ):
        """Record permanent verification failure without deleting remote files."""
        self._code(reason_code)
        async with self.db.transaction(tenant_id) as (c, now):
            row = await self._row(c, s.uploads, tenant_id, upload_id)
            if not row or decode(row["verification_ref"]) != verified_ref:
                return False
            u = decode(row["payload"])
            if u.status == US.FAILED:
                return row["failure_code"] == reason_code
            if u.status != US.VERIFYING:
                return False
            await self._save(
                c,
                s.uploads,
                tenant_id,
                upload_id,
                replace(u, status=US.FAILED),
                failure_code=reason_code,
            )
            return True

    async def _owner(self, c, tenant, id, worker, lease, now):
        """Check current job authority in the same transaction as its mutation."""
        j = await self._get(c, s.jobs, tenant, id)
        if (
            not j
            or j.status != JS.RUNNING
            or not j.lease
            or j.lease.worker_id != worker
            or j.lease.lease_id != lease
            or j.lease.expires_at <= now
        ):
            raise LeaseLostError("Job lease is no longer current")
        doc = await self._get(c, s.documents, tenant, j.document_id)
        if not doc or doc.deleted_at or doc.desired_generation_id != j.generation_id:
            raise LeaseLostError("Document generation is no longer current")
        generation = await self._row(c, s.generations, tenant, j.generation_id)
        if generation["status"] == "failed":
            raise LeaseLostError("Generation has failed")
        return j

    async def claim_job(self, *, tenant_id, job_id, worker_id, lease_id, lease_seconds):
        """Acquire eligible work once; old acquisition IDs cannot become new claims."""
        args = (job_id, worker_id, lease_seconds)
        async with self.db.transaction(tenant_id) as (c, now):
            replay = await self._replay(c, tenant_id, lease_id, "claim-job", args)
            j = await self._get(c, s.jobs, tenant_id, job_id)
            if replay is not _MISSING:
                return (
                    j
                    if j
                    and j.lease
                    and j.lease.lease_id == lease_id
                    and j.lease.expires_at > now
                    else None
                )
            if not j:
                return None
            doc = await self._get(c, s.documents, tenant_id, j.document_id)
            if (
                not doc
                or doc.deleted_at
                or doc.desired_generation_id != j.generation_id
            ):
                return None
            generation = await self._row(c, s.generations, tenant_id, j.generation_id)
            if generation["status"] == "failed" or generation["admitted_at"] is None:
                return None  # failed, or still waiting for admission
            if j.status not in (JS.QUEUED, JS.RUNNING) or (
                j.lease and j.lease.expires_at > now
            ):
                return None
            p = j.budget.policy
            if j.attempt >= p.retry.max_attempts or now >= j.budget.deadline_at:
                return None
            if (
                type(lease_seconds) is not int
                or not 0 < lease_seconds <= p.lease_seconds
            ):
                raise ValueError("Invalid lease duration")
            end = min(
                now + timedelta(seconds=p.max_attempt_seconds), j.budget.deadline_at
            )
            lease = JobLease(
                worker_id=worker_id,
                lease_id=lease_id,
                acquired_at=now,
                expires_at=min(now + timedelta(seconds=lease_seconds), end),
                attempt_deadline_at=end,
            )
            result = replace(
                j,
                status=JS.RUNNING,
                attempt=j.attempt + 1,
                lease=lease,
                next_attempt_at=None,
            )
            await self._save(c, s.jobs, tenant_id, job_id, result)
            await self._remember(c, tenant_id, lease_id, "claim-job", args, result)
            return result

    async def renew_job_lease(
        self, *, tenant_id, job_id, worker_id, lease_id, lease_seconds
    ):
        """Renew only an unexpired current lease without moving its attempt deadline."""
        async with self.db.transaction(tenant_id) as (c, now):
            try:
                j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            except LeaseLostError:
                return None
            if (
                type(lease_seconds) is not int
                or not 0 < lease_seconds <= j.budget.policy.lease_seconds
            ):
                raise ValueError("Invalid lease duration")
            end = min(j.lease.attempt_deadline_at, j.budget.deadline_at)
            if now >= end:
                return None
            lease = replace(
                j.lease, expires_at=min(now + timedelta(seconds=lease_seconds), end)
            )
            await self._save(c, s.jobs, tenant_id, job_id, replace(j, lease=lease))
            return lease

    async def get_checkpoint(self, *, tenant_id, job_id):
        """Read a checkpoint candidate for later artifact verification."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            v = (
                await c.execute(
                    sa.select(s.checkpoints.c.payload).where(
                        s.checkpoints.c.tenant_id == tenant_id,
                        s.checkpoints.c.job_id == job_id,
                    )
                )
            ).scalar_one_or_none()
            return decode(v) if v else None

    async def save_checkpoint(self, *, checkpoint, worker_id, lease_id):
        """Persist a complete output reference under a current lease."""
        tenant = checkpoint.tenant_id
        async with self.db.transaction(tenant) as (c, now):
            j = await self._owner(
                c, tenant, checkpoint.job_id, worker_id, lease_id, now
            )
            gen = await self._get(c, s.generations, tenant, j.generation_id)
            source = await self._get(c, s.sources, tenant, gen.source_version_id)
            for name in ("document_id", "generation_id", "stage", "shard_key"):
                if getattr(checkpoint, name) != getattr(j, name):
                    raise ValueError("Checkpoint job mismatch")
            if (
                checkpoint.producer_lease_id != lease_id
                or checkpoint.pipeline_fingerprint != gen.pipeline_fingerprint
                or checkpoint.source_sha256 != source.sha256
            ):
                raise ValueError("Checkpoint provenance mismatch")
            old = (
                await c.execute(
                    sa.select(s.checkpoints.c.payload).where(
                        s.checkpoints.c.tenant_id == tenant,
                        s.checkpoints.c.job_id == j.job_id,
                    )
                )
            ).scalar_one_or_none()
            if old:
                old = decode(old)
                if (old.input_sha256, old.manifest_sha256) != (
                    checkpoint.input_sha256,
                    checkpoint.manifest_sha256,
                ):
                    raise ValueError("Incompatible checkpoint")
                return old
            await self._save(
                c, s.checkpoints, tenant, checkpoint.checkpoint_id, checkpoint
            )
            return checkpoint

    async def _new_job(self, c, parent, stage, shard, now):
        """Derive stable continuation identities within the same generation budget."""
        id = uuid5(parent.generation_id, stage.value + ":" + shard)
        old = await self._get(c, s.jobs, parent.tenant_id, id)
        if old:
            return old
        deadline = min(
            now + timedelta(seconds=parent.budget.policy.max_job_seconds),
            parent.budget.generation_deadline_at,
        )
        if deadline <= now:
            return None
        budget = ExecutionBudget(
            policy=parent.budget.policy,
            starts_at=now,
            deadline_at=deadline,
            generation_deadline_at=parent.budget.generation_deadline_at,
        )
        job = Job(
            tenant_id=parent.tenant_id,
            document_id=parent.document_id,
            generation_id=parent.generation_id,
            job_id=id,
            stage=stage,
            status=JS.QUEUED,
            created_at=now,
            budget=budget,
            shard_key=shard,
        )
        event = JobAvailable(
            tenant_id=job.tenant_id, job_id=id, event_id=uuid5(id, "initial")
        )
        await self._insert_job(
            c, job, OutboxRecord(event=event, created_at=now, available_at=now)
        )
        return job

    async def list_generation_jobs(self, *, tenant_id, generation_id, stage=None):
        """Return bounded generation jobs for checkpoint traversal and status."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            q = sa.select(s.jobs.c.payload).where(
                s.jobs.c.tenant_id == tenant_id, s.jobs.c.generation_id == generation_id
            )
            if stage:
                q = q.where(s.jobs.c.stage == stage.value)
            return tuple(
                decode(v)
                for v in (
                    await c.execute(q.order_by(s.jobs.c.shard).limit(MAX_GENERATION_JOBS))
                ).scalars()
            )

    async def complete_stage(
        self,
        *,
        tenant_id,
        job_id,
        worker_id,
        lease_id,
        checkpoint_id,
        completion_id,
        next_jobs,
        next_outbox,
    ):
        """Finish a stage and save exact continuation intent atomically."""
        args = (job_id, worker_id, lease_id, checkpoint_id, next_jobs, next_outbox)
        async with self.db.transaction(tenant_id) as (c, now):
            replay = await self._replay(c, tenant_id, completion_id, "complete", args)
            if replay is not _MISSING:
                return replay
            j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            cp = await self._get(c, s.checkpoints, tenant_id, checkpoint_id)
            if not cp or cp.job_id != job_id or j.stage == ST.PUBLISH:
                raise ValueError("Missing stage checkpoint")
            if len(next_jobs) != len(next_outbox) or len(next_jobs) > 128:
                raise ValueError("Invalid bounded continuation")
            if next_jobs:
                raise ValueError(
                    "Built-in pipeline derives continuations from its saved stage plan"
                )
            grow = await self._row(c, s.generations, tenant_id, j.generation_id)
            if j.stage == ST.CHUNK and not grow["inventory"]:
                raise ValueError("Chunk stage needs a sealed inventory")
            result = replace(j, status=JS.SUCCEEDED, lease=None)
            await self._save(c, s.jobs, tenant_id, job_id, result)
            if j.stage == ST.EXTRACT:
                await self._new_job(c, j, ST.CHUNK, "0", now)
            elif j.stage == ST.EMBED:
                inv = decode(grow["inventory"])
                count = (
                    await c.execute(
                        sa.select(sa.func.count())
                        .select_from(s.jobs)
                        .where(
                            s.jobs.c.tenant_id == tenant_id,
                            s.jobs.c.generation_id == j.generation_id,
                            s.jobs.c.stage == ST.EMBED.value,
                            s.jobs.c.status == JS.SUCCEEDED.value,
                        )
                    )
                ).scalar_one()
                if count == len(inv.batches):
                    await self._new_job(c, j, ST.INDEX, "0", now)
            elif j.stage == ST.INDEX:
                await self._new_job(c, j, ST.PUBLISH, "0", now)
            await self._remember(c, tenant_id, completion_id, "complete", args, result)
            return result

    @staticmethod
    def _code(code):
        """Accept only bounded safe failure identifiers, never raw provider messages."""
        if (
            not isinstance(code, str)
            or not code
            or len(code) > 100
            or any(not (ch.isalnum() or ch in "_-") for ch in code)
        ):
            raise ValueError("Invalid failure code")

    @staticmethod
    def _limit(limit):
        """Bound scheduler and listing batches."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Limit must be between 1 and 1000")

    async def schedule_job_retry(
        self,
        *,
        tenant_id,
        job_id,
        worker_id,
        lease_id,
        retry_id,
        delay_seconds,
        failure_code,
    ):
        """Release ownership and retain a durable next-attempt time."""
        import math

        self._code(failure_code)
        if (
            isinstance(delay_seconds, bool)
            or not isinstance(delay_seconds, (float, int))
            or not math.isfinite(delay_seconds)
            or delay_seconds < 0
        ):
            raise ValueError("Invalid retry delay")
        args = (job_id, worker_id, lease_id, delay_seconds, failure_code)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, retry_id, "retry-job", args)
            if old is not _MISSING:
                return old
            j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            due = now + timedelta(seconds=delay_seconds)
            retry = (
                j.attempt < j.budget.policy.retry.max_attempts
                and due < j.budget.deadline_at
            )
            result = replace(
                j,
                status=JS.RETRY_WAIT if retry else JS.FAILED,
                lease=None,
                next_attempt_at=due if retry else None,
            )
            await self._save(
                c,
                s.jobs,
                tenant_id,
                job_id,
                result,
                retry_id=retry_id,
                failure_code=failure_code,
            )
            if not retry:
                await c.execute(
                    s.generations.update()
                    .where(
                        s.generations.c.tenant_id == tenant_id,
                        s.generations.c.id == j.generation_id,
                    )
                    .values(status="failed")
                )
            await self._remember(c, tenant_id, retry_id, "retry-job", args, result)
            return result

    async def fail_job(
        self, *, tenant_id, job_id, worker_id, lease_id, failure_id, failure_code
    ):
        """Record permanent failure without letting a stale worker mutate state."""
        self._code(failure_code)
        args = (job_id, worker_id, lease_id, failure_code)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, failure_id, "fail-job", args)
            if old is not _MISSING:
                return old
            j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            result = replace(j, status=JS.FAILED, lease=None, next_attempt_at=None)
            await self._save(
                c, s.jobs, tenant_id, job_id, result, failure_code=failure_code
            )
            await c.execute(
                s.generations.update()
                .where(
                    s.generations.c.tenant_id == tenant_id,
                    s.generations.c.id == j.generation_id,
                )
                .values(status="failed")
            )
            await self._remember(c, tenant_id, failure_id, "fail-job", args, result)
            return result

    async def release_due_retries(self, *, tenant_id, limit):
        """Move due waiting jobs to queued and insert one event per retry cycle."""
        self._limit(limit)
        async with self.db.transaction(tenant_id) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(s.jobs)
                        .where(
                            s.jobs.c.tenant_id == tenant_id,
                            s.jobs.c.status == JS.RETRY_WAIT.value,
                            s.jobs.c.due_at <= now,
                        )
                        .order_by(s.jobs.c.due_at)
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            events = []
            for row in rows:
                j = decode(row["payload"])
                doc = await self._get(c, s.documents, tenant_id, j.document_id)
                status = JS.QUEUED
                if doc.deleted_at or doc.desired_generation_id != j.generation_id:
                    status = JS.CANCELLED
                elif (
                    now >= j.budget.deadline_at
                    or j.attempt >= j.budget.policy.retry.max_attempts
                ):
                    status = JS.FAILED
                await self._save(
                    c,
                    s.jobs,
                    tenant_id,
                    j.job_id,
                    replace(j, status=status, next_attempt_at=None),
                )
                if status == JS.QUEUED:
                    event = JobAvailable(
                        tenant_id=tenant_id,
                        job_id=j.job_id,
                        event_id=uuid5(row["retry_id"], "release"),
                    )
                    record = OutboxRecord(event=event, created_at=now, available_at=now)
                    await self._save(c, s.outbox, tenant_id, event.event_id, record)
                    events.append(record)
            return tuple(events)

    async def get_outbox_record(self, *, tenant_id, event_id):
        """Inspect delivery intent without acquiring publishing authority."""
        return await self._read(s.outbox, tenant_id, event_id)

    async def claim_next_outbox(
        self, *, tenant_id, publisher_id, claim_id, claim_seconds
    ):
        """Acquire one due notification with a fresh durable acquisition identity."""
        if type(claim_seconds) is not int or not 1 <= claim_seconds <= 300:
            raise ValueError("Invalid publisher lease")
        args = (publisher_id, claim_seconds)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, claim_id, "claim-outbox", args)
            if old is not _MISSING:
                cur = await self._get(c, s.outbox, tenant_id, old.event.event_id)
                return (
                    cur
                    if cur  # may have been pruned as old history
                    and cur.claim
                    and cur.claim.claim_id == claim_id
                    and cur.claim.expires_at > now
                    else None
                )
            row = (
                (
                    await c.execute(
                        sa.select(s.outbox)
                        .where(
                            s.outbox.c.tenant_id == tenant_id,
                            s.outbox.c.published_at.is_(None),
                            s.outbox.c.available_at <= now,
                            sa.or_(
                                s.outbox.c.claim_expires_at.is_(None),
                                s.outbox.c.claim_expires_at <= now,
                            ),
                        )
                        .order_by(s.outbox.c.available_at)
                        .limit(1)
                    )
                )
                .mappings()
                .first()
            )
            if not row:
                return None
            rec = decode(row["payload"])
            result = replace(
                rec,
                claim=PublisherClaim(
                    publisher_id=publisher_id,
                    claim_id=claim_id,
                    expires_at=now + timedelta(seconds=claim_seconds),
                ),
                publish_attempts=rec.publish_attempts + 1,
            )
            await self._save(c, s.outbox, tenant_id, rec.event.event_id, result)
            await self._remember(c, tenant_id, claim_id, "claim-outbox", args, result)
            return result

    async def _publisher(self, c, tenant, event, publisher, claim, now):
        """Check the exact current publisher acquisition before recording outcomes."""
        r = await self._get(c, s.outbox, tenant, event)
        if (
            not r
            or r.published_at
            or not r.claim
            or r.claim.publisher_id != publisher
            or r.claim.claim_id != claim
            or r.claim.expires_at <= now
        ):
            raise LeaseLostError("Publisher claim lost")
        return r

    async def renew_outbox_claim(
        self, *, tenant_id, event_id, publisher_id, claim_id, claim_seconds
    ):
        """Renew only the matching unexpired publisher acquisition."""
        if type(claim_seconds) is not int or not 1 <= claim_seconds <= 300:
            raise ValueError("Invalid publisher lease")
        async with self.db.transaction(tenant_id) as (c, now):
            try:
                r = await self._publisher(
                    c, tenant_id, event_id, publisher_id, claim_id, now
                )
            except LeaseLostError:
                return None
            claim = replace(r.claim, expires_at=now + timedelta(seconds=claim_seconds))
            await self._save(c, s.outbox, tenant_id, event_id, replace(r, claim=claim))
            return claim

    async def mark_outbox_published(
        self, *, tenant_id, event_id, publisher_id, claim_id
    ):
        """Record confirmed queue acceptance, retaining the send-before-commit window."""
        op = uuid5(claim_id, "published")
        args = (event_id, publisher_id, claim_id)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, op, "published", args)
            if old is not _MISSING:
                return True
            try:
                r = await self._publisher(
                    c, tenant_id, event_id, publisher_id, claim_id, now
                )
            except LeaseLostError:
                return False
            await self._save(
                c,
                s.outbox,
                tenant_id,
                event_id,
                replace(r, claim=None, published_at=now),
            )
            await self._remember(c, tenant_id, op, "published", args, True)
            return True

    async def reschedule_outbox_send(
        self,
        *,
        tenant_id,
        event_id,
        publisher_id,
        claim_id,
        retry_id,
        delay_seconds,
        failure_code,
    ):
        """Keep unresolved delivery intent and release publishing ownership."""
        import math

        self._code(failure_code)
        if (
            not isinstance(delay_seconds, (int, float))
            or isinstance(delay_seconds, bool)
            or not math.isfinite(delay_seconds)
            or delay_seconds < 0
        ):
            raise ValueError("Invalid delay")
        args = (event_id, publisher_id, claim_id, delay_seconds, failure_code)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, retry_id, "retry-outbox", args)
            if old is not _MISSING:
                return old
            r = await self._publisher(
                c, tenant_id, event_id, publisher_id, claim_id, now
            )
            result = replace(
                r, claim=None, available_at=now + timedelta(seconds=delay_seconds)
            )
            await self._save(c, s.outbox, tenant_id, event_id, result)
            await self._remember(c, tenant_id, retry_id, "retry-outbox", args, result)
            return result

    async def seal_generation_inventory(
        self,
        *,
        inventory,
        inventory_ref,
        chunking_job_id,
        worker_id,
        lease_id,
        sealing_id,
    ):
        """Seal a verified immutable expected-batch plan before downstream scheduling."""
        args = (inventory, inventory_ref, chunking_job_id, worker_id, lease_id)
        t = inventory.tenant_id
        async with self.db.transaction(t) as (c, now):
            old = await self._replay(c, t, sealing_id, "seal", args)
            if old is not _MISSING:
                return old
            j = await self._owner(c, t, chunking_job_id, worker_id, lease_id, now)
            gen = await self._row(c, s.generations, t, inventory.generation_id)
            source = await self._get(
                c, s.sources, t, decode(gen["payload"]).source_version_id
            )
            if (
                j.stage != ST.CHUNK
                or j.generation_id != inventory.generation_id
                or inventory.document_id != j.document_id
                or inventory.source_version_id != source.source_version_id
                or inventory.source_sha256 != source.sha256
            ):
                raise ValueError("Inventory source mismatch")
            if (
                inventory.pipeline_fingerprint
                != decode(gen["payload"]).pipeline_fingerprint
            ):
                raise ValueError("Inventory recipe mismatch")
            if (
                inventory_ref.sha256 != inventory.sha256
                or inventory_ref.size_bytes != len(inventory.to_json().encode())
                or inventory_ref.format_name != "generation-inventory"
            ):
                raise ValueError("Inventory reference mismatch")
            if gen["inventory"] and decode(gen["inventory"]) != inventory:
                raise ValueError("Generation inventory is already sealed")
            ref = (
                decode(gen["inventory_ref"]) if gen["inventory_ref"] else inventory_ref
            )
            await c.execute(
                s.generations.update()
                .where(
                    s.generations.c.tenant_id == t,
                    s.generations.c.id == inventory.generation_id,
                )
                .values(
                    inventory=encode(inventory),
                    inventory_ref=encode(ref),
                    chunking_job_id=j.job_id,
                )
            )
            await self._remember(c, t, sealing_id, "seal", args, ref)
            return ref

    async def get_generation_inventory_ref(self, *, tenant_id, generation_id):
        """Return the registered inventory reference, not an untrusted replacement."""
        return await self._column(
            s.generations, tenant_id, generation_id, "inventory_ref"
        )

    async def get_inventory(self, *, tenant_id, generation_id):
        """Read the immutable sealed plan for bounded batch scheduling."""
        return await self._column(s.generations, tenant_id, generation_id, "inventory")

    async def get_batch_plan(self, *, tenant_id, generation_id, batch_key):
        """Return one scheduled batch's plan (None if no such batch is scheduled)."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            value = (
                await c.execute(
                    sa.select(s.scheduled.c.plan).where(
                        s.scheduled.c.tenant_id == tenant_id,
                        s.scheduled.c.generation_id == generation_id,
                        s.scheduled.c.batch_key == batch_key,
                    )
                )
            ).scalar_one_or_none()
        return decode(value) if value is not None else None

    async def release_inventory_batches(self, *, tenant_id, generation_id, limit):
        """Create bounded embedding jobs exactly once per sealed expected batch."""
        self._limit(limit)
        async with self.db.transaction(tenant_id) as (c, now):
            row = await self._row(c, s.generations, tenant_id, generation_id)
            if not row or not row["inventory"]:
                return ()
            gen = decode(row["payload"])
            doc = await self._get(c, s.documents, tenant_id, gen.document_id)
            parent = await self._get(c, s.jobs, tenant_id, row["chunking_job_id"])
            if (
                not doc
                or doc.deleted_at
                or doc.desired_generation_id != generation_id
                or not parent
                or parent.status != JS.SUCCEEDED
                or now >= row["deadline_at"]
            ):
                return ()
            inv = decode(row["inventory"])
            done = set(
                (
                    await c.execute(
                        sa.select(s.scheduled.c.batch_key).where(
                            s.scheduled.c.tenant_id == tenant_id,
                            s.scheduled.c.generation_id == generation_id,
                        )
                    )
                ).scalars()
            )
            result = []
            for batch in (b for b in inv.batches if b.batch_key not in done):
                j = await self._new_job(c, parent, ST.EMBED, batch.batch_key, now)
                if not j:
                    break
                await c.execute(
                    s.scheduled.insert().values(
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        batch_key=batch.batch_key,
                        job_id=j.job_id,
                        plan=encode(inv.plan_for(batch.batch_key)),
                    )
                )
                result.append(
                    await self._get(c, s.outbox, tenant_id, uuid5(j.job_id, "initial"))
                )
                if len(result) >= limit:
                    break
            if not inv.batches:
                await self._new_job(c, parent, ST.INDEX, "0", now)
            if len(done) + len(result) == len(inv.batches):
                await c.execute(
                    s.generations.update()
                    .where(
                        s.generations.c.tenant_id == tenant_id,
                        s.generations.c.id == generation_id,
                    )
                    .values(scheduling_done=True)
                )
            return tuple(result)

    async def publish_generation(
        self,
        *,
        tenant_id,
        document_id,
        generation_id,
        job_id,
        worker_id,
        lease_id,
        publication_id,
        expected_manifest_sha256,
        verified_chunk_count,
        verified_chunk_ids_sha256,
        vector_target,
        verified_at,
    ):
        """Activate only the still-desired verified generation and finish publication."""
        args = (
            document_id,
            generation_id,
            job_id,
            worker_id,
            lease_id,
            expected_manifest_sha256,
            verified_chunk_count,
            verified_chunk_ids_sha256,
            vector_target,
            verified_at,
        )
        async with self.db.transaction(tenant_id) as (c, now):
            replay = await self._replay(c, tenant_id, publication_id, "publish", args)
            if replay is not _MISSING:
                return await self._get(c, s.documents, tenant_id, document_id)
            j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            row = await self._row(c, s.generations, tenant_id, generation_id)
            inv = decode(row["inventory"]) if row else None
            if (
                j.stage != ST.PUBLISH
                or j.document_id != document_id
                or j.generation_id != generation_id
                or not inv
            ):
                raise ValueError("Invalid publication job")
            if (
                expected_manifest_sha256,
                verified_chunk_count,
                verified_chunk_ids_sha256,
                vector_target,
            ) != (
                inv.sha256,
                inv.chunk_count,
                inv.vector_ids_sha256,
                inv.vector_target,
            ):
                raise ValueError("Publication evidence differs from sealed plan")
            if (
                verified_at.tzinfo is None
                or not -5 <= (now - verified_at).total_seconds() <= 60
            ):
                raise ValueError("Publication verification is stale")
            indices = (
                (
                    await c.execute(
                        sa.select(s.jobs.c.status).where(
                            s.jobs.c.tenant_id == tenant_id,
                            s.jobs.c.generation_id == generation_id,
                            s.jobs.c.stage == ST.INDEX.value,
                        )
                    )
                )
                .scalars()
                .all()
            )
            if indices != [JS.SUCCEEDED.value]:
                raise ValueError("Indexing prerequisite incomplete")
            doc = await self._get(c, s.documents, tenant_id, document_id)
            if doc.active_generation_id and doc.active_generation_id != generation_id:
                await self._cleanup(
                    c, doc, doc.active_generation_id, now + timedelta(hours=24)
                )
            result = replace(doc, active_generation_id=generation_id)
            await self._save(c, s.documents, tenant_id, document_id, result)
            await self._save(
                c,
                s.jobs,
                tenant_id,
                job_id,
                replace(j, status=JS.SUCCEEDED, lease=None),
            )
            await c.execute(
                s.generations.update()
                .where(
                    s.generations.c.tenant_id == tenant_id,
                    s.generations.c.id == generation_id,
                )
                .values(status="published")
            )
            await self._remember(c, tenant_id, publication_id, "publish", args, result)
            return result

    async def _cleanup(self, c, doc, generation, after):
        """Retain cleanup intent without deleting shared objects inside a transaction."""
        id = uuid5(doc.document_id, "cleanup:" + str(generation))
        if not await self._row(c, s.cleanup, doc.tenant_id, id):
            await self._save(
                c,
                s.cleanup,
                doc.tenant_id,
                id,
                {"after": after.isoformat()},
                document_id=doc.document_id,
                generation_id=generation,
                status="pending",
            )

    async def tombstone_document(self, *, tenant_id, document_id, deletion_id):
        """Remove logical visibility before asynchronous physical cleanup."""
        async with self.db.transaction(tenant_id) as (c, now):
            replay = await self._replay(
                c, tenant_id, deletion_id, "delete", (document_id,)
            )
            doc = await self._get(c, s.documents, tenant_id, document_id)
            if replay is not _MISSING:
                return doc
            if not doc:
                return None
            if not doc.deleted_at:
                doc = replace(
                    doc,
                    deleted_at=now,
                    active_generation_id=None,
                    desired_generation_id=None,
                )
                await self._save(c, s.documents, tenant_id, document_id, doc)
                maximum = (
                    await c.execute(
                        sa.text(
                            "SELECT COALESCE(MAX(CAST(payload->'fields'->'budget'->'fields'->'policy'->'fields'->>'max_attempt_seconds' AS double precision)),0) FROM jobs WHERE tenant_id=:tenant AND document_id=:doc"
                        ),
                        {"tenant": tenant_id, "doc": document_id},
                    )
                ).scalar_one()
                await self._cleanup(
                    c, doc, None, now + timedelta(seconds=max(3600, maximum) + 60)
                )
            await self._remember(
                c, tenant_id, deletion_id, "delete", (document_id,), doc
            )
            return doc

    async def reconcile(self, tenant_id, limit=100):
        """Recover expired work, missing notifications, and abandoned intake in bounds."""
        self._limit(limit)
        async with self.db.transaction(tenant_id) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(s.jobs)
                        .where(
                            s.jobs.c.tenant_id == tenant_id,
                            s.jobs.c.status.in_(
                                ["queued", "running", "retry_wait", "waiting"]
                            ),
                        )
                        .order_by(
                            s.jobs.c.reconciled_at.asc().nulls_first(), s.jobs.c.id
                        )
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            repaired = 0
            for row in rows:
                j = decode(row["payload"])
                doc = await self._get(c, s.documents, tenant_id, j.document_id)
                await c.execute(
                    s.jobs.update()
                    .where(s.jobs.c.tenant_id == tenant_id, s.jobs.c.id == j.job_id)
                    .values(reconciled_at=now)
                )
                generation = await self._row(
                    c, s.generations, tenant_id, j.generation_id
                )
                if (
                    generation["status"] == "failed"
                    or doc.deleted_at
                    or doc.desired_generation_id != j.generation_id
                ):
                    await self._save(
                        c,
                        s.jobs,
                        tenant_id,
                        j.job_id,
                        replace(
                            j, status=JS.CANCELLED, lease=None, next_attempt_at=None
                        ),
                    )
                    repaired += 1
                    continue
                if generation["admitted_at"] is None:
                    # Waiting for admission: its clock has not started, and
                    # admit_generations sends its notification when it starts.
                    continue
                if now >= j.budget.deadline_at or (
                    j.status == JS.RUNNING
                    and j.lease.expires_at <= now
                    and j.attempt >= j.budget.policy.retry.max_attempts
                ):
                    await self._save(
                        c,
                        s.jobs,
                        tenant_id,
                        j.job_id,
                        replace(j, status=JS.FAILED, lease=None, next_attempt_at=None),
                        failure_code="execution_budget_exhausted",
                    )
                    await c.execute(
                        s.generations.update()
                        .where(
                            s.generations.c.tenant_id == tenant_id,
                            s.generations.c.id == j.generation_id,
                        )
                        .values(status="failed")
                    )
                    repaired += 1
                    continue
                if j.status == JS.RUNNING and j.lease.expires_at <= now:
                    j = replace(j, status=JS.QUEUED, lease=None)
                    await self._save(c, s.jobs, tenant_id, j.job_id, j)
                    repaired += 1
                if j.status == JS.QUEUED:
                    last = (
                        await c.execute(
                            sa.select(sa.func.max(s.outbox.c.published_at)).where(
                                s.outbox.c.tenant_id == tenant_id,
                                s.outbox.c.job_id == j.job_id,
                            )
                        )
                    ).scalar_one()
                    pending = (
                        await c.execute(
                            sa.select(sa.func.count())
                            .select_from(s.outbox)
                            .where(
                                s.outbox.c.tenant_id == tenant_id,
                                s.outbox.c.job_id == j.job_id,
                                s.outbox.c.published_at.is_(None),
                            )
                        )
                    ).scalar_one()
                    if not pending and (not last or (now - last).total_seconds() > 300):
                        e = JobAvailable(
                            event_id=uuid4(), tenant_id=tenant_id, job_id=j.job_id
                        )
                        await self._save(
                            c,
                            s.outbox,
                            tenant_id,
                            e.event_id,
                            OutboxRecord(event=e, created_at=now, available_at=now),
                        )
                        repaired += 1
            uploads = (
                (
                    await c.execute(
                        sa.select(s.uploads)
                        .where(
                            s.uploads.c.tenant_id == tenant_id,
                            s.uploads.c.status.in_(["uploading", "verifying"]),
                            s.uploads.c.expires_at <= now,
                        )
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            for row in uploads:
                u = decode(row["payload"])
                await self._save(
                    c, s.uploads, tenant_id, u.upload_id, replace(u, status=US.EXPIRED)
                )
                repaired += 1
            return repaired

    # ---- external recognition --------------------------------------------

    @staticmethod
    def _recognition(row):
        """Shape a recognition_jobs row; payload holds provider identities."""
        payload = decode(row["payload"])
        return {
            "id": row["id"],
            "job_id": row["job_id"],
            "generation_id": row["generation_id"],
            "request_id": payload["request_id"],
            "submission": payload["submission"],
            "provider_job_id": payload["provider_job_id"],
            "page_numbers": tuple(payload["page_numbers"]),
            "select_pages": payload.get("select_pages", False),
            "state": row["state"],
            "polls": row["polls"],
            "failure_code": row["failure_code"],
            "failures": {int(k): v for k, v in (row["failures"] or {}).items()},
        }

    async def record_recognition_job(
        self,
        *,
        tenant_id,
        job_id,
        worker_id,
        lease_id,
        request_id,
        submission,
        provider_job_id,
        page_numbers,
        poll_after_seconds,
        select_pages=False,
    ):
        """Durably record a submitted provider job under the current lease.

        Called right after the tool accepted the job, before any waiting, so a
        crash never causes an unintended resubmission. Idempotent per
        (request_id, submission).
        """
        if (
            not isinstance(provider_job_id, str)
            or not 1 <= len(provider_job_id) <= 1024
        ):
            raise ValueError("Provider job id must be a bounded string")
        if type(submission) is not int or submission < 0:
            raise ValueError("submission must be a nonnegative integer")
        rid = uuid5(job_id, f"recognition:{request_id}:{submission}")
        t = s.recognition_jobs
        async with self.db.transaction(tenant_id) as (c, now):
            j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            row = (
                (
                    await c.execute(
                        sa.select(t).where(t.c.tenant_id == tenant_id, t.c.id == rid)
                    )
                )
                .mappings()
                .first()
            )
            if row:
                return self._recognition(row)
            payload = {
                "request_id": request_id,
                "submission": submission,
                "provider_job_id": provider_job_id,
                "page_numbers": list(page_numbers),
                "select_pages": bool(select_pages),
            }
            await c.execute(
                t.insert().values(
                    tenant_id=tenant_id,
                    id=rid,
                    payload=encode(payload),
                    job_id=job_id,
                    generation_id=j.generation_id,
                    state="pending",
                    next_poll_at=now + timedelta(seconds=max(0, poll_after_seconds)),
                    polls=0,
                    submitted_at=now,
                )
            )
            row = (
                (
                    await c.execute(
                        sa.select(t).where(t.c.tenant_id == tenant_id, t.c.id == rid)
                    )
                )
                .mappings()
                .first()
            )
            return self._recognition(row)

    async def list_recognition_jobs(self, *, tenant_id, job_id):
        """Return every recorded provider job of one stage job."""
        t = s.recognition_jobs
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(t)
                        .where(t.c.tenant_id == tenant_id, t.c.job_id == job_id)
                        .order_by(t.c.submitted_at)
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._recognition(r) for r in rows)

    async def _pending_recognitions(self, c, tenant, job_id):
        t = s.recognition_jobs
        return (
            await c.execute(
                sa.select(sa.func.count())
                .select_from(t)
                .where(
                    t.c.tenant_id == tenant,
                    t.c.job_id == job_id,
                    t.c.state == "pending",
                )
            )
        ).scalar_one()

    async def _requeue(self, c, tenant, job, now):
        """Return a parked job to the queue with a fresh notification."""
        queued = replace(job, status=JS.QUEUED, lease=None, next_attempt_at=None)
        await self._save(c, s.jobs, tenant, job.job_id, queued, failure_code=None)
        e = JobAvailable(event_id=uuid4(), tenant_id=tenant, job_id=job.job_id)
        await self._save(
            c,
            s.outbox,
            tenant,
            e.event_id,
            OutboxRecord(event=e, created_at=now, available_at=now),
        )
        return queued

    async def park_job_waiting(
        self, *, tenant_id, job_id, worker_id, lease_id, wait_id
    ):
        """Release a job that waits on external work, without spending attempts.

        Checked atomically with the pending recognition count: if every
        recorded provider job already finished, the job is re-queued at once,
        so a fast provider can never leave it waiting forever.
        """
        args = (job_id, worker_id, lease_id)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, wait_id, "wait", args)
            if old is not _MISSING:
                return old
            j = await self._owner(c, tenant_id, job_id, worker_id, lease_id, now)
            if await self._pending_recognitions(c, tenant_id, job_id):
                result = replace(j, status=JS.WAITING, lease=None, next_attempt_at=None)
                await self._save(
                    c,
                    s.jobs,
                    tenant_id,
                    job_id,
                    result,
                    failure_code="awaiting_recognition",
                )
            else:
                result = await self._requeue(c, tenant_id, j, now)
            await self._remember(c, tenant_id, wait_id, "wait", args, result)
            return result

    async def due_recognition_polls(self, *, tenant_id, limit):
        """Pending provider jobs whose next poll time has come."""
        self._limit(limit)
        t = s.recognition_jobs
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(t)
                        .where(
                            t.c.tenant_id == tenant_id,
                            t.c.state == "pending",
                            t.c.next_poll_at <= now,
                        )
                        .order_by(t.c.next_poll_at)
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            return tuple(self._recognition(r) for r in rows)

    async def finish_recognition_poll(
        self,
        *,
        tenant_id,
        recognition_id,
        state,
        failure_code=None,
        failures=None,
        retry_after_seconds=0,
    ):
        """Record one poll outcome; wake the waiting job when nothing is pending.

        Only pending rows change. Results must already be cached before a row
        is marked done, so the re-queued job finds every finished page.
        """
        if state not in ("pending", "done", "failed", "expired"):
            raise ValueError("Invalid recognition state")
        if failure_code is not None:
            self._code(failure_code)
        failures = {str(k): v for k, v in (failures or {}).items()}
        if any(
            v not in ("permanent", "retryable", "uncertain") for v in failures.values()
        ):
            raise ValueError("Invalid page failure category")
        t = s.recognition_jobs
        async with self.db.transaction(tenant_id) as (c, now):
            row = (
                (
                    await c.execute(
                        sa.select(t).where(
                            t.c.tenant_id == tenant_id, t.c.id == recognition_id
                        )
                    )
                )
                .mappings()
                .first()
            )
            if not row or row["state"] != "pending":
                return False
            where = (t.c.tenant_id == tenant_id, t.c.id == recognition_id)
            if state == "pending":
                await c.execute(
                    t.update()
                    .where(*where)
                    .values(
                        next_poll_at=now
                        + timedelta(seconds=max(1, retry_after_seconds)),
                        polls=t.c.polls + 1,
                    )
                )
                return True
            await c.execute(
                t.update()
                .where(*where)
                .values(
                    state=state,
                    finished_at=now,
                    next_poll_at=None,
                    polls=t.c.polls + 1,
                    failure_code=failure_code,
                    failures=failures or None,
                )
            )
            job = await self._get(c, s.jobs, tenant_id, row["job_id"])
            if (
                job
                and job.status == JS.WAITING
                and not await self._pending_recognitions(c, tenant_id, job.job_id)
            ):
                await self._requeue(c, tenant_id, job, now)
            return True

    async def release_waiting_jobs(self, *, tenant_id, limit=100):
        """Sweep: abandon provider jobs of ended work and wake resolved waiters."""
        self._limit(limit)
        t = s.recognition_jobs
        async with self.db.transaction(tenant_id) as (c, now):
            await c.execute(
                t.update()
                .where(
                    t.c.tenant_id == tenant_id,
                    t.c.state == "pending",
                    t.c.job_id.in_(
                        sa.select(s.jobs.c.id).where(
                            s.jobs.c.tenant_id == tenant_id,
                            s.jobs.c.status.in_(["succeeded", "failed", "cancelled"]),
                        )
                    ),
                )
                .values(state="abandoned", finished_at=now, next_poll_at=None)
            )
            rows = (
                await c.execute(
                    sa.select(s.jobs.c.payload)
                    .where(
                        s.jobs.c.tenant_id == tenant_id, s.jobs.c.status == "waiting"
                    )
                    .limit(limit)
                )
            ).scalars()
            released = 0
            for payload in list(rows):
                job = decode(payload)
                if not await self._pending_recognitions(c, tenant_id, job.job_id):
                    await self._requeue(c, tenant_id, job, now)
                    released += 1
            return released

    async def acquire_rate(self, *, key, rate, burst):
        """Shared token bucket across processes; return seconds to wait (0 = go)."""
        if not isinstance(key, str) or not 1 <= len(key) <= 200:
            raise ValueError("Invalid rate key")
        if not (rate > 0 and burst >= 1):
            raise ValueError("Invalid rate or burst")
        async with self.db.transaction(None, lock=False) as (c, now):
            await c.execute(
                sa.text(
                    "INSERT INTO rate_limits (key, tokens, updated_at)"
                    " VALUES (:k, :b, :now) ON CONFLICT DO NOTHING"
                ),
                {"k": key, "b": float(burst), "now": now},
            )
            row = (
                await c.execute(
                    sa.text(
                        "SELECT tokens, updated_at FROM rate_limits WHERE key = :k FOR UPDATE"
                    ),
                    {"k": key},
                )
            ).one()
            elapsed = max(0.0, (now - row.updated_at).total_seconds())
            tokens = min(float(burst), row.tokens + elapsed * rate)
            wait = 0.0
            if tokens >= 1:
                tokens -= 1
            else:
                wait = (1 - tokens) / rate
            await c.execute(
                sa.text(
                    "UPDATE rate_limits SET tokens = :t, updated_at = :now WHERE key = :k"
                ),
                {"t": tokens, "now": max(now, row.updated_at), "k": key},
            )
            return wait

    async def reserve_recognition_pages(
        self, *, tenant_id, request_id, pages, daily_limit
    ):
        """Reserve billed pages for today (UTC); return seconds until it fits (0 = ok).

        Idempotent per request_id, so a retried request is not counted twice.
        daily_limit 0 means unlimited.
        """
        if type(pages) is not int or pages < 1:
            raise ValueError("pages must be a positive integer")
        u = s.recognition_usage
        async with self.db.transaction(tenant_id) as (c, now):
            day = now.astimezone(timezone.utc).date()
            key = (
                u.c.tenant_id == tenant_id,
                u.c.day == day,
                u.c.request_id == request_id,
            )
            if (
                await c.execute(sa.select(sa.func.count()).select_from(u).where(*key))
            ).scalar_one():
                return 0.0
            if daily_limit:
                used = (
                    await c.execute(
                        sa.select(sa.func.coalesce(sa.func.sum(u.c.pages), 0)).where(
                            u.c.tenant_id == tenant_id, u.c.day == day
                        )
                    )
                ).scalar_one()
                if used + pages > daily_limit:
                    tomorrow = datetime.combine(
                        day + timedelta(days=1), time(0), tzinfo=timezone.utc
                    )
                    return max(1.0, (tomorrow - now).total_seconds())
            await c.execute(
                u.insert().values(
                    tenant_id=tenant_id, day=day, request_id=request_id, pages=pages
                )
            )
            return 0.0

    async def aclose(self):
        """Close this metadata adapter's connection pool."""
        await self.db.aclose()

    async def list_verifying_uploads(self, tenant_id, limit=10):
        """Find durable verification work that survived an API disconnect."""
        self._limit(limit)
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = (
                await c.execute(
                    sa.select(s.uploads.c.payload)
                    .where(
                        s.uploads.c.tenant_id == tenant_id,
                        s.uploads.c.status == US.VERIFYING.value,
                        s.uploads.c.expires_at > now,
                    )
                    .order_by(s.uploads.c.id)
                    .limit(limit)
                )
            ).scalars()
            return tuple(decode(v) for v in rows)

    async def admit_generations(self, *, tenant_id, max_active):
        """Start waiting generations, oldest first, up to max_active per tenant.

        Budgets start at admission, not acceptance, so time spent waiting for
        capacity never counts against a build. Saved durations are kept.
        """
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        g, d = s.generations, s.documents
        async with self.db.transaction(tenant_id) as (c, now):
            # A generation past its deadline can never publish (every job
            # deadline precedes it); mark it failed so it frees its slot.
            await c.execute(
                g.update()
                .where(
                    g.c.tenant_id == tenant_id,
                    g.c.status == "processing",
                    g.c.admitted_at.is_not(None),
                    g.c.deadline_at <= now,
                )
                .values(status="failed")
            )
            live = sa.and_(
                g.c.tenant_id == tenant_id,
                g.c.status == "processing",
                d.c.deleted_at.is_(None),
                d.c.desired_id == g.c.id,
            )
            joined = g.join(
                d, sa.and_(d.c.tenant_id == g.c.tenant_id, d.c.id == g.c.document_id)
            )
            active = (
                await c.execute(
                    sa.select(sa.func.count())
                    .select_from(joined)
                    .where(live, g.c.admitted_at.is_not(None))
                )
            ).scalar_one()
            free = max_active - active
            if free <= 0:
                return 0
            waiting = (
                (
                    await c.execute(
                        sa.select(g.c.id, g.c.deadline_at)
                        .select_from(joined)
                        .where(live, g.c.admitted_at.is_(None))
                        .order_by(g.c.payload["fields"]["created_at"]["$time"].astext)
                        .limit(free)
                    )
                )
                .mappings()
                .all()
            )
            for row in waiting:
                deadline = row["deadline_at"]
                job = (
                    await c.execute(
                        sa.select(s.jobs.c.payload).where(
                            s.jobs.c.tenant_id == tenant_id,
                            s.jobs.c.generation_id == row["id"],
                            s.jobs.c.stage == ST.EXTRACT.value,
                        )
                    )
                ).scalar_one_or_none()
                job = decode(job) if job else None
                if job and job.status == JS.QUEUED and job.attempt == 0:
                    b = job.budget
                    budget = ExecutionBudget(
                        policy=b.policy,
                        starts_at=now,
                        deadline_at=now + (b.deadline_at - b.starts_at),
                        generation_deadline_at=now
                        + (b.generation_deadline_at - b.starts_at),
                    )
                    await self._save(
                        c, s.jobs, tenant_id, job.job_id, replace(job, budget=budget)
                    )
                    deadline = budget.generation_deadline_at
                    pending = (
                        await c.execute(
                            sa.select(sa.func.count())
                            .select_from(s.outbox)
                            .where(
                                s.outbox.c.tenant_id == tenant_id,
                                s.outbox.c.job_id == job.job_id,
                                s.outbox.c.published_at.is_(None),
                            )
                        )
                    ).scalar_one()
                    if not pending:
                        e = JobAvailable(
                            event_id=uuid4(), tenant_id=tenant_id, job_id=job.job_id
                        )
                        await self._save(
                            c,
                            s.outbox,
                            tenant_id,
                            e.event_id,
                            OutboxRecord(event=e, created_at=now, available_at=now),
                        )
                await c.execute(
                    g.update()
                    .where(g.c.tenant_id == tenant_id, g.c.id == row["id"])
                    .values(admitted_at=now, deadline_at=deadline)
                )
            return len(waiting)

    async def prune_history(self, *, older_than_days, limit=1000):
        """Delete expired replay/notification history in bounded batches.

        Only rows whose purpose has passed are removed: published notifications,
        operation-ledger entries (lease-keyed replays happen within seconds;
        client idempotency keys are honored for older_than_days), quarantine
        evidence, and part observations of uploads that are no longer open.
        Jobs, generations, documents and cleanup history are kept.
        """
        if type(older_than_days) is not int or older_than_days < 1:
            raise ValueError("older_than_days must be a positive integer")
        self._limit(limit)
        statements = (
            "DELETE FROM outbox WHERE ctid IN (SELECT ctid FROM outbox"
            " WHERE published_at < :cutoff LIMIT :limit)",
            "DELETE FROM operations WHERE ctid IN (SELECT ctid FROM operations"
            " WHERE created_at < :cutoff LIMIT :limit)",
            "DELETE FROM quarantine WHERE ctid IN (SELECT ctid FROM quarantine"
            " WHERE created_at < :cutoff LIMIT :limit)",
            "DELETE FROM upload_parts WHERE ctid IN (SELECT p.ctid FROM upload_parts p"
            " JOIN uploads u ON u.tenant_id = p.tenant_id AND u.id = p.upload_id"
            " WHERE u.status <> 'uploading' LIMIT :limit)",
            "DELETE FROM recognition_jobs WHERE ctid IN (SELECT ctid FROM"
            " recognition_jobs WHERE state <> 'pending' AND finished_at < :cutoff"
            " LIMIT :limit)",
            "DELETE FROM recognition_usage WHERE ctid IN (SELECT ctid FROM"
            " recognition_usage WHERE day < CAST(:cutoff AS date) LIMIT :limit)",
        )
        deleted = 0
        async with self.db.transaction(None, lock=False) as (c, now):
            args = {"cutoff": now - timedelta(days=older_than_days), "limit": limit}
            for statement in statements:
                deleted += (await c.execute(sa.text(statement), args)).rowcount
        return deleted

    async def scheduling_generations(self, tenant_id, limit=100):
        """Return bounded sealed generations still needing fan-out scheduling."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            return tuple(
                (
                    await c.execute(
                        sa.select(s.generations.c.id)
                        .select_from(
                            s.generations.join(
                                s.documents,
                                sa.and_(
                                    s.generations.c.tenant_id
                                    == s.documents.c.tenant_id,
                                    s.generations.c.document_id == s.documents.c.id,
                                ),
                            )
                        )
                        .where(
                            s.generations.c.tenant_id == tenant_id,
                            s.generations.c.inventory.is_not(None),
                            s.generations.c.status == "processing",
                            s.generations.c.scheduling_done.is_(False),
                            s.generations.c.deadline_at > now,
                            s.documents.c.deleted_at.is_(None),
                            s.documents.c.desired_id == s.generations.c.id,
                        )
                        .order_by(s.generations.c.id)
                        .limit(limit)
                    )
                ).scalars()
            )

    async def reprocess_document(
        self,
        *,
        tenant_id,
        document_id,
        request_id,
        policy,
        max_generation_seconds,
        pipeline=None,
    ):
        """Explicitly start a fresh generation from the last accepted source.

        This authorized operation resets a budget deliberately, unlike queue
        redelivery. request_id keeps repeated API requests from creating builds.
        """
        from rag_ingest.domain.documents import ProcessingGeneration

        args = (document_id, policy, max_generation_seconds, pipeline)
        async with self.db.transaction(tenant_id) as (c, now):
            old = await self._replay(c, tenant_id, request_id, "reprocess", args)
            if old is not _MISSING:
                return old
            doc = await self._get(c, s.documents, tenant_id, document_id)
            if not doc or doc.deleted_at or not doc.desired_generation_id:
                raise ValueError("No accepted source to reprocess")
            prev = await self._row(
                c, s.generations, tenant_id, doc.desired_generation_id
            )
            recipe = pipeline or decode(prev["pipeline"])
            previous = decode(prev["payload"])
            gen = ProcessingGeneration(
                tenant_id=tenant_id,
                document_id=document_id,
                source_version_id=previous.source_version_id,
                generation_id=uuid4(),
                pipeline_fingerprint=recipe.fingerprint,
                created_at=now,
            )
            deadline = now + timedelta(seconds=max_generation_seconds)
            budget = ExecutionBudget(
                policy=policy,
                starts_at=now,
                deadline_at=min(
                    deadline, now + timedelta(seconds=policy.max_job_seconds)
                ),
                generation_deadline_at=deadline,
            )
            job = Job(
                tenant_id=tenant_id,
                document_id=document_id,
                generation_id=gen.generation_id,
                job_id=uuid5(gen.generation_id, "extract:0"),
                stage=ST.EXTRACT,
                status=JS.QUEUED,
                created_at=now,
                budget=budget,
            )
            event = JobAvailable(
                event_id=uuid5(job.job_id, "initial"),
                tenant_id=tenant_id,
                job_id=job.job_id,
            )
            await self._save(
                c,
                s.generations,
                tenant_id,
                gen.generation_id,
                gen,
                pipeline=encode(recipe),
                deadline_at=deadline,
            )
            await self._insert_job(
                c, job, OutboxRecord(event=event, created_at=now, available_at=now)
            )
            await self._save(
                c,
                s.documents,
                tenant_id,
                document_id,
                replace(doc, desired_generation_id=gen.generation_id),
            )
            await self._remember(c, tenant_id, request_id, "reprocess", args, gen)
            return gen

    async def cleanup_candidates(self, tenant_id, limit=10):
        """Read pending cleanup intent; the janitor rechecks authority before acting."""
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(s.cleanup)
                        .where(
                            s.cleanup.c.tenant_id == tenant_id,
                            s.cleanup.c.status == "pending",
                            s.cleanup.c.payload["after"].astext <= now.isoformat(),
                        )
                        .order_by(s.cleanup.c.id)
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            return tuple(
                dict(row)
                for row in rows
                if decode(row["payload"])["after"] <= now.isoformat()
            )

    async def quarantine_delivery(self, delivery, reason):
        """Durably retain malformed queue diagnostics before acknowledging delivery."""
        from uuid import UUID

        tenant = UUID(int=0)
        id = uuid5(tenant, delivery.message_id + reason)
        async with self.db.transaction(tenant) as (c, now):
            if not await self._row(c, s.quarantine, tenant, id):
                await self._save(
                    c,
                    s.quarantine,
                    tenant,
                    id,
                    {
                        "message_id": delivery.message_id,
                        "body": delivery.body[:65536],
                        "sha256": hashlib.sha256(delivery.body.encode()).hexdigest(),
                        "received_at": now.isoformat(),
                    },
                    reason=reason,
                )

    async def job_report(self, tenant, job_id):
        """Return safe operational state without lease tokens or source contents."""
        async with self.db.transaction(tenant, lock=False) as (c, now):
            row = await self._row(c, s.jobs, tenant, job_id)
            return _report(row) if row else None

    async def list_job_reports(self, tenant, generation_id):
        """Return one generation's safe job reports in a single bounded query."""
        async with self.db.transaction(tenant, lock=False) as (c, now):
            rows = (
                (
                    await c.execute(
                        sa.select(s.jobs.c.payload, s.jobs.c.failure_code)
                        .where(
                            s.jobs.c.tenant_id == tenant,
                            s.jobs.c.generation_id == generation_id,
                        )
                        .order_by(s.jobs.c.shard)
                        .limit(MAX_GENERATION_JOBS)
                    )
                )
                .mappings()
                .all()
            )
            return [_report(row) for row in rows]

    async def authorize_cleanup(self, tenant, cleanup_id):
        """Freeze a reference-safe cleanup plan before any external deletion.

        Artifacts being deleted cannot accept new references until cleanup
        finishes. Historical rows remain; a later upload can rehydrate content
        under the same artifact identity with newly verified retained bytes.
        """
        from rag_ingest.contracts.storage import ObjectRef

        async with self.db.transaction(tenant) as (c, now):
            task = await self._row(c, s.cleanup, tenant, cleanup_id)
            if not task or task["status"] != "pending":
                return None
            payload = decode(task["payload"])
            if payload["after"] > now.isoformat():
                return None
            if "plan" in payload:
                return payload["plan"]
            doc = await self._get(c, s.documents, tenant, task["document_id"])
            target = task["generation_id"]
            if target and target in (
                doc.active_generation_id,
                doc.desired_generation_id,
            ):
                return None
            if target is None and not doc.deleted_at:
                return None
            active = (
                await c.execute(
                    sa.select(sa.func.count())
                    .select_from(s.jobs)
                    .where(
                        s.jobs.c.tenant_id == tenant,
                        s.jobs.c.document_id == doc.document_id,
                        s.jobs.c.status == "running",
                        s.jobs.c.lease_expires_at > now,
                    )
                )
            ).scalar_one()
            if active:
                return None
            q = sa.select(s.generations.c.id).where(
                s.generations.c.tenant_id == tenant,
                s.generations.c.document_id == doc.document_id,
            )
            if target:
                q = q.where(s.generations.c.id == target)
            generations = tuple((await c.execute(q)).scalars())
            refs = []
            artifact_ids = []
            if doc.deleted_at and target is None:
                rows = (
                    (
                        await c.execute(
                            sa.select(s.sources).where(
                                s.sources.c.tenant_id == tenant,
                                s.sources.c.document_id == doc.document_id,
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
                for row in rows:
                    source = decode(row["payload"])
                    arow = await self._row(c, s.artifacts, tenant, source.artifact_id)
                    artifact = decode(arow["payload"])
                    live = (
                        await c.execute(
                            sa.select(sa.func.count())
                            .select_from(
                                s.sources.join(
                                    s.documents,
                                    sa.and_(
                                        s.sources.c.tenant_id
                                        == s.documents.c.tenant_id,
                                        s.sources.c.document_id == s.documents.c.id,
                                    ),
                                )
                            )
                            .where(
                                s.sources.c.tenant_id == tenant,
                                s.sources.c.artifact_id == source.artifact_id,
                                s.documents.c.deleted_at.is_(None),
                            )
                        )
                    ).scalar_one()
                    own = ObjectRef(
                        bucket=source.bucket,
                        key=source.object_key,
                        version_id=source.object_version_id,
                    )
                    canonical = ObjectRef(
                        bucket=artifact.bucket,
                        key=artifact.object_key,
                        version_id=artifact.object_version_id,
                    )
                    if not live:
                        await c.execute(
                            s.artifacts.update()
                            .where(
                                s.artifacts.c.tenant_id == tenant,
                                s.artifacts.c.id == source.artifact_id,
                            )
                            .values(state="deleting")
                        )
                        artifact_ids.append(source.artifact_id)
                        refs.extend((own, canonical))
                    elif (own.bucket, own.key) != (canonical.bucket, canonical.key):
                        refs.append(own)
            plan = {
                "generations": generations,
                "sources": tuple(dict.fromkeys(refs)),
                "artifacts": tuple(set(artifact_ids)),
            }
            payload["plan"] = plan
            await self._save(c, s.cleanup, tenant, cleanup_id, payload)
            return plan

    async def finish_cleanup_plan(self, tenant, cleanup_id, plan):
        """Finalize only the durable plan actually authorized before remote erasure."""
        async with self.db.transaction(tenant) as (c, now):
            task = await self._row(c, s.cleanup, tenant, cleanup_id)
            if not task:
                return
            if decode(task["payload"]).get("plan") != plan:
                raise ValueError("Cleanup plan mismatch")
            for id in plan["artifacts"]:
                await c.execute(
                    s.artifacts.update()
                    .where(
                        s.artifacts.c.tenant_id == tenant,
                        s.artifacts.c.id == id,
                        s.artifacts.c.state == "deleting",
                    )
                    .values(state="deleted")
                )
            await c.execute(
                s.cleanup.update()
                .where(s.cleanup.c.tenant_id == tenant, s.cleanup.c.id == cleanup_id)
                .values(status="complete")
            )
