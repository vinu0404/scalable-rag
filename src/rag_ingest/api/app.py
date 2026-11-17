"""Expose upload and document operations without accepting tenant IDs from clients."""

import asyncio
import hmac
import math
import inspect
import logging
from contextlib import asynccontextmanager
from typing import Literal
from datetime import datetime
from uuid import UUID, uuid5
from fastapi import FastAPI, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from rag_ingest.contracts.errors import (
    DuplicateContentError,
    IngestionError,
    OperatorActionRequiredError,
    RetryableError,
    UncertainOutcomeError,
    LeaseLostError,
)
from rag_ingest.domain.search import SearchFilters
from rag_ingest.runtime.search import result_json, run as run_search
from rag_ingest.domain.limits import MAX_FILENAME_CHARS, MAX_REQUEST_KEY_CHARS
from rag_ingest.logs import log_event


class UploadRequest(BaseModel):
    """Describe the file; bytes go directly to the signed storage destination."""

    model_config = ConfigDict(extra="forbid")
    filename: str = Field(min_length=1, max_length=MAX_FILENAME_CHARS)
    size_bytes: int = Field(gt=0, strict=True)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    document_id: UUID | None = None
    allow_duplicate: bool = False
    # Labels for search filters; only for a new document (use PATCH later).
    tags: list[str] | None = None
    metadata: dict[str, str | int | float | bool] | None = None


class PartRequest(BaseModel):
    """Optionally bind a part grant to an expected checksum."""

    model_config = ConfigDict(extra="forbid")
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class SearchRequest(BaseModel):
    """Bound query input before spending an embedding request."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=100)
    filters: "SearchFiltersBody | None" = None
    # vector, keyword or hybrid; default from INGEST_SEARCH_HYBRID.
    mode: Literal["vector", "keyword", "hybrid"] | None = None
    # Rerank the top candidates; default from INGEST_SEARCH_RERANK.
    rerank: bool | None = None


class SearchFiltersBody(BaseModel):
    """Every given condition must match; see domain.search.SearchFilters."""

    model_config = ConfigDict(extra="forbid")
    document_ids: list[UUID] = Field(default_factory=list, max_length=1000)
    tags_any: list[str] = Field(default_factory=list)
    tags_all: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    created_after: datetime | None = None
    created_before: datetime | None = None

    def to_filters(self):
        return SearchFilters.build(
            document_ids=self.document_ids,
            tags_any=self.tags_any,
            tags_all=self.tags_all,
            metadata=self.metadata,
            created_after=self.created_after,
            created_before=self.created_before,
        )


class LabelsRequest(BaseModel):
    """Replace a document's tags and/or metadata (omitted fields stay as they are)."""

    model_config = ConfigDict(extra="forbid")
    tags: list[str] | None = None
    metadata: dict[str, str | int | float | bool] | None = None


SearchRequest.model_rebuild()


def upload_status(upload):
    """Return safe progress fields without storage credentials or provider handles."""
    return dict(
        upload_id=str(upload.upload_id),
        document_id=str(upload.document_id),
        status=upload.status.value,
        size_bytes=upload.expected_size_bytes,
        expires_at=upload.expires_at.isoformat(),
        duplicate_of=str(upload.duplicate_of) if upload.duplicate_of else None,
    )


def grant_json(grant):
    """Expose the short-lived grant only in the response that requested it."""
    return dict(
        url=grant.url,
        headers=dict(grant.headers),
        expires_at=grant.expires_at.isoformat(),
    )


async def enforce_rate(engine, tenant, operation, per_minute):
    """Refuse with 429 once a tenant exceeds per_minute calls of an operation.

    The token bucket lives in PostgreSQL, so every API process shares one
    limit per tenant; nothing is kept in process memory. It holds up to one
    minute of calls and refills continuously. 0 means unlimited.
    """
    if not per_minute:
        return
    wait = await engine.metadata.acquire_rate(
        key=f"api:{operation}:{tenant}", rate=per_minute / 60, burst=per_minute
    )
    if wait > 0:
        raise HTTPException(
            429,
            "Too many requests for this tenant; retry later",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )


async def search_chunks(engine, tenant, query, limit, filters=None, mode=None, rerank=None):
    """Search this tenant's live documents; see runtime.search for the steps."""
    outcome = await run_search(
        engine, tenant, query, limit=limit, filters=filters, mode=mode, rerank=rerank
    )
    return {
        "results": [result_json(r) for r in outcome.results],
        "mode": outcome.mode,
        "degraded": list(outcome.degraded),
    }


def create_app(engine=None, *, settings=None, authenticate=None, background=False):
    """Build the API; optionally run all three background roles for local use.

    authenticate may be an async callable receiving the bearer token and
    returning a trusted tenant UUID. The bundled policy grants tenant-wide
    document access. Never derive a tenant from a request body or query string.
    """

    @asynccontextmanager
    async def lifespan(app):
        """Own the engine and stop background work before closing its connections."""
        nonlocal engine, settings
        from rag_ingest.runtime.settings import ServiceSettings
        from rag_ingest.runtime.factory import build_engine
        from rag_ingest.runtime.runners import run_publisher, run_scheduler, run_workers

        settings = settings or (
            engine.settings if engine else ServiceSettings.from_env()
        )
        if authenticate is None and not settings.api_tokens:
            raise ValueError("Configure INGEST_API_TOKENS or provide an authenticator")
        engine = engine or build_engine(settings)
        app.state.engine = engine
        app.state.failed = False
        stop = asyncio.Event()
        tasks = []

        def finished(task):
            """Make failed background roles visible to readiness checks."""
            if not task.cancelled() and task.exception() is not None:
                app.state.failed = True
                log_event(
                    logging.getLogger(__name__),
                    "background_role_failed",
                    logging.ERROR,
                    task.exception(),
                    category=type(task.exception()).__name__,
                )

        if background:
            tasks = [
                asyncio.create_task(role(engine, stop))
                for role in (run_publisher, run_scheduler, run_workers)
            ]
            for task in tasks:
                task.add_done_callback(finished)
        try:
            yield
        finally:
            stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await engine.aclose()

    configured = settings or (engine.settings if engine else None)
    docs = bool(getattr(configured, "api_docs", False))
    app = FastAPI(
        title="Recoverable document ingestion",
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs and the schema are opt-in (INGEST_API_DOCS=true).
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )

    async def tenant(authorization: str | None = Header(default=None)):
        """Resolve credentials on the server; invalid credentials reveal no tenant."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Bearer token required")
        token = authorization[7:]
        result = None
        if authenticate:
            result = authenticate(token)
            if inspect.isawaitable(result):
                result = await result
        else:
            for secret, candidate in settings.api_tokens.items():
                if hmac.compare_digest(token.encode(), secret.encode()):
                    result = candidate
        if not isinstance(result, UUID):
            raise HTTPException(401, "Invalid credentials")
        return result

    @app.exception_handler(DuplicateContentError)
    async def duplicate(request, error):
        """Name the existing document so the client can link to it instead."""
        return JSONResponse(
            status_code=409,
            content={
                "error": "duplicate_content",
                "detail": "A live document already has identical content; "
                "send allow_duplicate to keep a second copy.",
                "duplicate_of": {
                    "document_id": str(error.document_id),
                    "filename": error.filename,
                },
            },
        )

    @app.exception_handler(IngestionError)
    async def ingestion_error(request, error):
        """Report safe categories instead of provider messages or connection strings."""
        status = (
            503
            if isinstance(error, (RetryableError, OperatorActionRequiredError))
            else 409
            if isinstance(error, (UncertainOutcomeError, LeaseLostError))
            else 422
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": type(error).__name__,
                "detail": "Operation could not complete; inspect status before retrying.",
            },
        )

    @app.exception_handler(ValueError)
    async def bad_input(request, error):
        """Keep validation failures separate from unexpected server exceptions."""
        return JSONResponse(
            status_code=400,
            content={
                "error": "InvalidOperation",
                "detail": "Invalid input or operation unavailable in current state.",
            },
        )

    async def live_document(t, id):
        """Apply the bundled tenant-wide ACL before exposing a document operation."""
        doc = await engine.metadata.get_document(tenant_id=t, document_id=id)
        if not doc or doc.deleted_at or doc.acl_ref != "tenant":
            raise HTTPException(404, "Document not found")
        return doc

    @app.get("/healthz")
    async def health():
        """Report process liveness without exposing configuration."""
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready(t: UUID = Depends(tenant)):
        """Check SQL reachability and whether a bundled background role has failed."""
        if app.state.failed:
            raise HTTPException(503, "Background role failed")
        await engine.metadata.now(t)
        return {"status": "ready"}

    @app.post("/v1/uploads", status_code=201)
    async def initiate(
        body: UploadRequest,
        t: UUID = Depends(tenant),
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=1, max_length=MAX_REQUEST_KEY_CHARS
        ),
    ):
        """Register a resumable upload before issuing a storage grant."""
        await enforce_rate(
            engine, t, "uploads", engine.settings.api_uploads_per_minute
        )
        if body.document_id:
            await live_document(t, body.document_id)
        result = await engine.initiate_upload(
            t,
            filename=body.filename,
            size_bytes=body.size_bytes,
            sha256=body.sha256,
            document_id=body.document_id,
            request_key=idempotency_key,
            allow_duplicate=body.allow_duplicate,
            tags=body.tags,
            metadata=body.metadata,
        )
        answer = upload_status(result["upload"])
        answer["grant"] = grant_json(result["grant"]) if result["grant"] else None
        plan = result["plan"]
        answer["multipart"] = (
            dict(part_size_bytes=plan.part_size_bytes, total_parts=plan.total_parts)
            if plan
            else None
        )
        return answer

    @app.get("/v1/uploads/{upload_id}")
    async def upload(upload_id: UUID, t: UUID = Depends(tenant)):
        """Read progress using the authenticated tenant and the returned upload ID."""
        value = await engine.metadata.get_upload(tenant_id=t, upload_id=upload_id)
        if not value:
            raise HTTPException(404, "Upload not found")
        await live_document(t, value.document_id)
        return upload_status(value)

    @app.post("/v1/uploads/{upload_id}/parts/{number}")
    async def part(
        upload_id: UUID, number: int, body: PartRequest, t: UUID = Depends(tenant)
    ):
        """Issue a fresh grant for one planned part; retry the same part number."""
        return grant_json(await engine.part_url(t, upload_id, number, body.sha256))

    @app.get("/v1/uploads/{upload_id}/parts")
    async def parts(
        upload_id: UUID,
        after: int | None = Query(default=None, ge=1),
        limit: int = Query(default=1000, ge=1, le=1000),
        t: UUID = Depends(tenant),
    ):
        """Read provider-confirmed parts so interrupted transfers can resume."""
        page = await engine.list_parts(t, upload_id, after, limit)
        return {
            "parts": [
                dict(part_number=p.part_number, etag=p.etag, size_bytes=p.size_bytes)
                for p in page.parts
            ],
            "next_after": page.next_after,
        }

    @app.post("/v1/uploads/{upload_id}/complete", status_code=202)
    async def complete(upload_id: UUID, t: UUID = Depends(tenant)):
        """Complete storage transfer; the scheduler performs full verification later."""
        return upload_status(await engine.complete_upload(t, upload_id))

    @app.get("/v1/documents/{document_id}")
    async def document(document_id: UUID, t: UUID = Depends(tenant)):
        """Distinguish currently searchable data from the desired replacement build."""
        doc = await live_document(t, document_id)
        jobs = (
            await engine.metadata.list_job_reports(t, doc.desired_generation_id)
            if doc.desired_generation_id
            else []
        )
        return dict(
            document_id=str(doc.document_id),
            filename=doc.display_name,
            active_generation_id=str(doc.active_generation_id)
            if doc.active_generation_id
            else None,
            desired_generation_id=str(doc.desired_generation_id)
            if doc.desired_generation_id
            else None,
            created_at=doc.created_at.isoformat(),
            tags=list(doc.tags),
            metadata=doc.metadata_dict,
            jobs=jobs,
        )

    @app.patch("/v1/documents/{document_id}")
    async def labels(document_id: UUID, body: LabelsRequest, t: UUID = Depends(tenant)):
        """Replace tags and/or metadata used by search filters; takes effect at once."""
        await live_document(t, document_id)
        doc = await engine.metadata.update_document_labels(
            tenant_id=t, document_id=document_id, tags=body.tags, metadata=body.metadata
        )
        if doc is None:
            raise HTTPException(404, "Document not found")
        return dict(
            document_id=str(doc.document_id), tags=list(doc.tags), metadata=doc.metadata_dict
        )

    @app.get("/v1/jobs/{job_id}")
    async def job(job_id: UUID, t: UUID = Depends(tenant)):
        """Expose durable failure codes without internal lease tokens."""
        j = await engine.metadata.get_job(tenant_id=t, job_id=job_id)
        if not j:
            raise HTTPException(404, "Job not found")
        await live_document(t, j.document_id)
        return await engine.metadata.job_report(t, job_id)

    @app.get("/v1/documents/{document_id}/view")
    async def view(
        document_id: UUID, download: bool = False, t: UUID = Depends(tenant)
    ):
        """Sign an exact original source version after checking document access."""
        await live_document(t, document_id)
        result = await engine.view_file(t, document_id, download=download)
        return {"url": result.url, "expires_at": result.expires_at.isoformat()}

    def operation_id(t, kind, key):
        """Derive a stable operation identity from a bounded caller idempotency key."""
        return uuid5(t, kind + ":" + key)

    @app.delete("/v1/documents/{document_id}", status_code=202)
    async def delete(
        document_id: UUID,
        t: UUID = Depends(tenant),
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=1, max_length=MAX_REQUEST_KEY_CHARS
        ),
    ):
        """Tombstone immediately; durable cleanup follows after the safety grace."""
        existing = await engine.metadata.get_document(
            tenant_id=t, document_id=document_id
        )
        if not existing or existing.acl_ref != "tenant":
            raise HTTPException(404, "Document not found")
        await engine.metadata.tombstone_document(
            tenant_id=t,
            document_id=document_id,
            deletion_id=operation_id(t, "delete", idempotency_key),
        )
        return {"document_id": str(document_id), "status": "deleted"}

    @app.post("/v1/documents/{document_id}/reprocess", status_code=202)
    async def reprocess(
        document_id: UUID,
        t: UUID = Depends(tenant),
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=1, max_length=MAX_REQUEST_KEY_CHARS
        ),
    ):
        """Explicitly authorize a fresh build and budget using the current recipe."""
        await live_document(t, document_id)
        gen = await engine.metadata.reprocess_document(
            tenant_id=t,
            document_id=document_id,
            request_id=operation_id(t, "reprocess", idempotency_key),
            policy=settings.core.execution_policy,
            max_generation_seconds=settings.core.max_generation_seconds,
            pipeline=settings.pipeline(),
        )
        return {"generation_id": str(gen.generation_id)}

    @app.post("/v1/search")
    async def search(body: SearchRequest, t: UUID = Depends(tenant)):
        """Search this tenant's live documents, optionally filtered (see runtime.search)."""
        await enforce_rate(
            engine, t, "searches", engine.settings.api_searches_per_minute
        )
        filters = body.filters.to_filters() if body.filters else None
        return await search_chunks(
            engine, t, body.query, body.limit, filters, body.mode, body.rerank
        )

    return app
