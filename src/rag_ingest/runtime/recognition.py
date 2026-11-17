"""Recognize the pages a document recipe marks, through any injected Recognizer.

The coordinator owns everything generic so adapters stay small: per-build
page cache, planning to the tool's declared capabilities (page images, whole
document, page selection or page subsets, split to fit limits), the most
private accepted input form, tenant budgets, a shared rate limiter, deadline
awareness, partial results, and job-style submit/record/wait/poll.

Guarantees:
- Every requested page ends as recognized text or an explicit failure.
- A finished page is cached before any error is raised, so retries never
  re-bill it; with racing writers the first cached result wins.
- A provider job is recorded durably right after submission; only expired or
  retryable failures are resubmitted, at most MAX_SUBMISSIONS times per page.
"""

import asyncio
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag_ingest.adapters.processing import pdf_tools
from rag_ingest.contracts.errors import (
    AwaitingRecognitionError,
    OperatorActionRequiredError,
    PermanentInputError,
    RetryableError,
    UncertainOutcomeError,
    with_code,
)
from rag_ingest.contracts.recognition import (
    DocumentInput,
    JobPoll,
    RecognitionRequest,
    RecognitionResult,
    RecognizedPage,
    SubmittedJob,
)
from rag_ingest.contracts.storage import ObjectRef
from rag_ingest.domain.limits import MAX_PAGE_CHARACTERS
from rag_ingest.logs import log_event

log = logging.getLogger(__name__)

MAX_SUBMISSIONS = 3
RENDER_BATCH = 16
INPUT_ORDER = ("path", "bytes", "storage_ref", "signed_url")
SEVERITY = (
    OperatorActionRequiredError,
    PermanentInputError,
    UncertainOutcomeError,
    RetryableError,
)


def category(error):
    """Map a contract error to a stored failure category."""
    if isinstance(error, PermanentInputError):
        return "permanent"
    if isinstance(error, UncertainOutcomeError):
        return "uncertain"
    return "retryable"


def _prefix(tenant_id, generation_id):
    return f"tenants/{tenant_id}/generations/{generation_id}/recognition"


def max_page_characters(settings):
    return min(settings.max_extracted_characters, MAX_PAGE_CHARACTERS)


async def cache_get(engine, tenant_id, generation_id, fingerprint, page_number):
    """Return the cached page of this build, or None; reject foreign entries."""
    key = f"{_prefix(tenant_id, generation_id)}/pages/{page_number:06d}.json"
    info = await engine.objects.stat(ObjectRef(bucket=engine.settings.bucket, key=key))
    if info is None:
        return None
    data = await engine.read_bytes(
        info.ref, max_page_characters(engine.settings) * 6 + 4096
    )
    item = json.loads(data)
    if (
        item.get("schema_version") != 1
        or item.get("page_number") != page_number
        or item.get("recipe") != fingerprint
    ):
        raise PermanentInputError("Recognition cache entry does not match this build")
    return RecognizedPage(
        page_number=page_number, text=item["text"], confidence=item["confidence"]
    )


async def cache_put(engine, tenant_id, generation_id, fingerprint, page):
    """Cache one recognized page; if another writer won, return its entry."""
    key = f"{_prefix(tenant_id, generation_id)}/pages/{page.page_number:06d}.json"
    data = json.dumps(
        {
            "schema_version": 1,
            "page_number": page.page_number,
            "text": page.text,
            "confidence": page.confidence,
            "recipe": fingerprint,
        },
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    try:
        await engine.store_bytes(key, data, "recognized_page", "recognized-page")
    except ValueError:  # create-only key already holds a different result
        existing = await cache_get(
            engine, tenant_id, generation_id, fingerprint, page.page_number
        )
        if existing is None:
            raise
        return existing
    return page


class SourceFile:
    """Download and verify the pinned source only when something needs it."""

    def __init__(self, engine, source, path):
        self.engine, self.source, self._path, self._ready = engine, source, path, False

    async def path(self):
        if not self._ready:
            await self.engine.download_verified(self.source, self._path)
            self._ready = True
        return self._path

    @property
    def ref(self):
        return ObjectRef(
            bucket=self.source.bucket,
            key=self.source.object_key,
            version_id=self.source.object_version_id,
        )


@dataclass
class _Part:
    """One planned request: its pages and how its content is reached."""

    pages: tuple
    kind: str  # image | whole | select | part
    path: Path | None = None
    media_type: str = "application/pdf"
    size: int = 0
    ref: ObjectRef | None = None


class Recognition:
    """Recognize the marked pages of one extract job."""

    def __init__(
        self,
        engine,
        *,
        job,
        source,
        pipeline,
        options,
        page_count,
        local,
        workdir,
        deadline,
    ):
        self.engine = engine
        self.recognizer = engine.recognizer
        self.caps = engine.recognizer.capabilities
        self.settings = engine.settings
        self.job, self.source, self.pipeline = job, source, pipeline
        self.options, self.page_count = options, page_count
        self.local, self.workdir, self.deadline = local, Path(workdir), deadline
        self.loop = asyncio.get_running_loop()

    # -- public ---------------------------------------------------------------

    async def run(self, pages):
        """Return {page_number: RecognizedPage} for every page, or raise."""
        pages = sorted(pages)
        if len(pages) > self.settings.max_recognized_pages:
            raise with_code(
                PermanentInputError(
                    "Document needs more recognized pages than allowed"
                ),
                "recognition_page_budget_exceeded",
            )
        results = {}
        for number in pages:
            cached = await self._cache_get(number)
            if cached is not None:
                results[number] = cached
        remaining = [p for p in pages if p not in results]
        if remaining:
            if self.caps.call_style == "job":
                await self._run_jobs(remaining)  # always raises (waits or fails)
            results.update(await self._run_sync(remaining))
        floor = self.options["min_confidence"]
        if floor is not None and any(
            r.confidence is not None and r.confidence < floor for r in results.values()
        ):
            raise with_code(
                PermanentInputError("Recognized text is below the minimum confidence"),
                "recognition_low_confidence",
            )
        return results

    # -- cache ------------------------------------------------------------------

    async def _cache_get(self, number):
        return await cache_get(
            self.engine,
            self.job.tenant_id,
            self.job.generation_id,
            self.pipeline.fingerprint,
            number,
        )

    async def _cache_put(self, page):
        return await cache_put(
            self.engine,
            self.job.tenant_id,
            self.job.generation_id,
            self.pipeline.fingerprint,
            page,
        )

    # -- planning -----------------------------------------------------------------

    def _chunks(self, pages):
        size = self.caps.max_pages_per_request or len(pages)
        return [tuple(pages[i : i + size]) for i in range(0, len(pages), size)]

    def _fits(self, size):
        limit = self.caps.max_bytes_per_request
        return limit is None or size <= limit

    async def _document_parts(self, pages):
        """Plan document requests: page selection, whole file, or page subsets."""
        whole = self.source.size_bytes
        if self.caps.page_selection and self._fits(whole):
            return [
                _Part(pages=g, kind="select", size=whole) for g in self._chunks(pages)
            ]
        if (
            list(pages) == list(range(1, self.page_count + 1))
            and len(pages) <= (self.caps.max_pages_per_request or len(pages))
            and self._fits(whole)
        ):
            return [_Part(pages=tuple(pages), kind="whole", size=whole)]
        parts = []
        for group in self._chunks(pages):
            parts.extend(await self._split(group))
        return parts

    async def _split(self, group):
        """Build a page-subset PDF; halve groups until each part fits."""
        digest = hashlib.sha256(",".join(map(str, group)).encode()).hexdigest()[:16]
        path = self.workdir / f"part-{digest}.pdf"
        await pdf_tools.split(
            await self.local.path(),
            path,
            pages=group,
            strict_pdf=self.options["strict_pdf"],
            timeout_seconds=self._call_timeout(),
        )
        size = path.stat().st_size
        if self._fits(size):
            return [_Part(pages=group, kind="part", path=path, size=size)]
        if len(group) == 1:
            raise with_code(
                PermanentInputError(
                    "A single page exceeds the recognizer's size limit"
                ),
                "recognition_input_too_large",
            )
        middle = len(group) // 2
        return await self._split(group[:middle]) + await self._split(group[middle:])

    async def _render(self, pages):
        """Render pages to images that fit the tool's pixel/byte/type limits."""
        directory = self.workdir / f"render-{pages[0]:06d}"
        directory.mkdir(exist_ok=True)
        caps = self.caps
        max_pixels = min(
            self.options["max_render_pixels"], caps.max_image_pixels or 10**12
        )
        entries = await pdf_tools.render(
            await self.local.path(),
            directory,
            pages=pages,
            dpi=self.options["render_dpi"],
            max_pixels=max_pixels,
            max_bytes=caps.max_bytes_per_request,
            media_types=caps.image_media_types,
            timeout_seconds=self._call_timeout(),
        )
        parts = []
        for number in pages:
            entry = entries[number]
            if "error" in entry:
                raise with_code(
                    PermanentInputError(
                        "A page image cannot fit the recognizer's limits"
                    ),
                    "recognition_input_too_large",
                )
            parts.append(
                _Part(
                    pages=(number,),
                    kind="image",
                    path=directory / entry["file"],
                    media_type=entry["media_type"],
                    size=entry["size"],
                )
            )
        return parts

    # -- requests -----------------------------------------------------------------

    def _input_kind(self, kind):
        allowed = [
            k
            for k in INPUT_ORDER
            if k in self.caps.inputs
            and (k != "signed_url" or self.options["allow_signed_url"])
            and not (kind == "image" and k in ("storage_ref", "signed_url"))
        ]
        if not allowed:
            raise OperatorActionRequiredError(
                "The recognizer accepts only signed_url inputs; set "
                "INGEST_ALLOW_SIGNED_URL=true to allow short-lived read links"
            )
        return allowed[0]

    def _call_timeout(self):
        return max(5.0, min(120.0, self.deadline - self.loop.time()))

    def _request_id(self, pages, submission):
        raw = "|".join(
            (
                str(self.job.generation_id),
                self.pipeline.fingerprint,
                ",".join(map(str, pages)),
                str(submission),
            )
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:40]

    async def _upload_part(self, part, request_id):
        """Store a page-subset PDF so storage-reading tools can access it."""
        key = f"{_prefix(self.job.tenant_id, self.job.generation_id)}/parts/{request_id}.pdf"
        data = part.path.read_bytes()
        info = await self.engine.store_bytes(key, data, "recognition_part", "pdf")
        return ObjectRef(
            bucket=info.bucket, key=info.object_key, version_id=info.object_version_id
        )

    async def _request(self, part, submission):
        request_id = self._request_id(part.pages, submission)
        kind = self._input_kind(part.kind)
        if part.kind in ("whole", "select"):
            ref, path = self.local.ref, None
        else:
            ref, path = None, part.path
        if kind == "path":
            source = DocumentInput(
                kind="path", path=path or await self.local.path(), size_bytes=part.size
            )
        elif kind == "bytes":
            source = DocumentInput(
                kind="bytes",
                data=(path or await self.local.path()).read_bytes(),
                size_bytes=part.size,
            )
        else:
            ref = ref or await self._upload_part(part, request_id)
            if kind == "storage_ref":
                source = DocumentInput(
                    kind="storage_ref", ref=ref, size_bytes=part.size
                )
            else:
                grant = await self.engine.objects.presign_read(
                    ref, expires_seconds=min(3600, self.settings.presign_seconds)
                )
                source = DocumentInput(
                    kind="signed_url",
                    url=grant.url,
                    url_expires_at=grant.expires_at,
                    size_bytes=part.size,
                )
        return RecognitionRequest(
            tenant_id=self.job.tenant_id,
            document_id=self.job.document_id,
            generation_id=self.job.generation_id,
            request_id=request_id,
            input=source,
            media_type=part.media_type,
            page_numbers=part.pages,
            select_pages=part.kind == "select",
            timeout_seconds=self._call_timeout(),
        )

    # -- budgets and rate -----------------------------------------------------------

    async def _reserve(self, request):
        limit = self.settings.tenant_daily_recognized_pages
        pages = len(request.page_numbers)
        if limit and pages > limit:
            raise with_code(
                PermanentInputError(
                    "One recognition request exceeds the daily page budget"
                ),
                "recognition_budget_too_small",
            )
        wait = await self.engine.metadata.reserve_recognition_pages(
            tenant_id=self.job.tenant_id,
            request_id=request.request_id,
            pages=pages,
            daily_limit=limit,
        )
        if wait:
            now = await self.engine.metadata.now(self.job.tenant_id)
            if (self.job.budget.deadline_at - now).total_seconds() <= wait:
                # The budget resets after this job's deadline; say so plainly
                # instead of a generic deadline failure. Reprocess after reset.
                raise with_code(
                    PermanentInputError("Tenant daily recognition budget is used up"),
                    "recognition_budget_exhausted",
                )
            raise with_code(
                RetryableError(
                    "Tenant daily recognition budget reached", retry_after_seconds=wait
                ),
                "recognition_budget_wait",
            )

    async def _throttle(self):
        await throttle(self.engine, self.deadline)

    # -- sync tools ------------------------------------------------------------------

    def _past_deadline(self):
        return self.loop.time() >= self.deadline

    async def _run_sync(self, pages):
        results, errors, deferred = {}, [], []
        semaphore = asyncio.Semaphore(self.settings.recognition_concurrency)

        async def one(part):
            async with semaphore:
                if self._past_deadline():
                    deferred.append(part)
                    return
                request = await self._request(part, 0)
                await self._reserve(request)
                await self._throttle()
                async with asyncio.timeout(request.timeout_seconds + 10):
                    result = await self.recognizer.recognize(request)
                try:
                    if not isinstance(result, RecognitionResult):
                        raise ValueError("recognize() must return a RecognitionResult")
                    result.validate_for(
                        request, max_page_characters=max_page_characters(self.settings)
                    )
                except ValueError:
                    raise with_code(
                        PermanentInputError(
                            "Recognizer returned results that do not match the request"
                        ),
                        "recognizer_contract_violation",
                    ) from None
                for page in result.pages:
                    results[page.page_number] = await self._cache_put(page)
                errors.extend(f.error for f in result.failures)

        async def run(parts):
            outcomes = await asyncio.gather(
                *(one(p) for p in parts), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    if isinstance(outcome, asyncio.CancelledError):
                        raise outcome
                    errors.append(outcome)

        if self.caps.unit == "page_image":
            for i in range(0, len(pages), RENDER_BATCH):
                if self._past_deadline():
                    deferred.append(None)
                    break
                await run(await self._render(pages[i : i + RENDER_BATCH]))
        else:
            await run(await self._document_parts(pages))
        if errors:
            raise _most_severe(errors)
        if deferred or set(results) != set(pages):
            raise with_code(
                RetryableError(
                    "Recognition progress saved; continuing next attempt",
                    retry_after_seconds=0,
                ),
                "recognition_in_progress",
            )
        return results

    # -- job tools -------------------------------------------------------------------

    async def _run_jobs(self, pages):
        metadata = self.engine.metadata
        records = await metadata.list_recognition_jobs(
            tenant_id=self.job.tenant_id, job_id=self.job.job_id
        )
        remaining = set(pages)
        counts = Counter(p for r in records for p in r["page_numbers"])
        pending = set()
        for r in records:
            covered = set(r["page_numbers"]) & remaining
            if not covered:
                continue
            if r["state"] == "pending":
                pending |= covered
            elif r["state"] == "done" and any(
                r["failures"].get(p) == "permanent" for p in covered
            ):
                raise with_code(
                    PermanentInputError("The recognizer could not read a page"),
                    "recognition_failed",
                )
            elif r["state"] == "failed" and r["failure_code"] == "permanent":
                raise with_code(
                    PermanentInputError("The recognizer rejected the document"),
                    "recognition_failed",
                )
        to_submit = sorted(remaining - pending)
        if any(counts[p] >= MAX_SUBMISSIONS for p in to_submit):
            raise with_code(
                PermanentInputError(
                    "Recognition did not complete after repeated submissions"
                ),
                "recognition_resubmit_limit",
            )
        if to_submit:
            if self.caps.unit == "page_image":
                parts = []
                for i in range(0, len(to_submit), RENDER_BATCH):
                    parts.extend(await self._render(to_submit[i : i + RENDER_BATCH]))
            else:
                parts = await self._document_parts(to_submit)
            for part in parts:
                submission = max(counts[p] for p in part.pages)
                request = await self._request(part, submission)
                await self._reserve(request)
                await self._throttle()
                async with asyncio.timeout(request.timeout_seconds + 10):
                    provider_job_id = await self.recognizer.submit(request)
                await metadata.record_recognition_job(
                    tenant_id=self.job.tenant_id,
                    job_id=self.job.job_id,
                    worker_id=self.job.lease.worker_id,
                    lease_id=self.job.lease.lease_id,
                    request_id=request.request_id,
                    submission=submission,
                    provider_job_id=provider_job_id,
                    page_numbers=part.pages,
                    poll_after_seconds=self.settings.recognition_poll_seconds,
                    select_pages=part.kind == "select",
                )
                log_event(
                    log,
                    "recognition_submitted",
                    job_id=self.job.job_id,
                    pages=len(part.pages),
                    submission=submission,
                )
        raise AwaitingRecognitionError("Waiting for the recognizer to finish")


def _most_severe(errors):
    """Pick the error that decides the job's outcome; keep unexpected ones visible."""
    for kind in SEVERITY:
        for error in errors:
            if isinstance(error, kind):
                if kind is PermanentInputError and not hasattr(error, "code"):
                    with_code(error, "recognition_failed")
                return error
    return errors[0]


async def throttle(engine, deadline):
    """Wait for the shared per-tool rate limit, never past the deadline."""
    rate = engine.recognizer.capabilities.max_requests_per_second
    if not rate:
        return
    loop = asyncio.get_running_loop()
    key = "recognizer:" + engine.recognizer.spec.driver
    while True:
        wait = await engine.metadata.acquire_rate(
            key=key, rate=rate, burst=max(1.0, rate)
        )
        if wait <= 0:
            return
        if loop.time() + wait > deadline:
            raise with_code(
                RetryableError(
                    "Recognition rate limit wait exceeds this attempt",
                    retry_after_seconds=wait,
                ),
                "recognition_rate_limited",
            )
        await asyncio.sleep(wait)


# -- scheduler-side polling for job-style tools -------------------------------------


async def poll_once(engine):
    """Poll due provider jobs, cache finished pages, wake waiting jobs."""
    recognizer = engine.recognizer
    if recognizer is None or recognizer.capabilities.call_style != "job":
        return 0
    polled = 0
    semaphore = asyncio.Semaphore(engine.settings.recognition_concurrency)
    for tenant in await engine.metadata.tenants():
        await engine.metadata.release_waiting_jobs(tenant_id=tenant)
        due = await engine.metadata.due_recognition_polls(tenant_id=tenant, limit=50)

        async def one(row, tenant=tenant):
            async with semaphore:
                await _poll_one(engine, tenant, row)

        await asyncio.gather(*(one(r) for r in due))
        polled += len(due)
    return polled


async def _poll_one(engine, tenant, row):
    settings = engine.settings
    backoff = min(300, settings.recognition_poll_seconds * 2 ** min(row["polls"], 4))

    async def finish(**kwargs):
        """Record the poll outcome; log every terminal state for operators."""
        changed = await engine.metadata.finish_recognition_poll(**kwargs)
        if changed and kwargs["state"] != "pending":
            log_event(
                log,
                "recognition_finished",
                logging.INFO if kwargs["state"] == "done" else logging.WARNING,
                recognition_id=kwargs["recognition_id"],
                state=kwargs["state"],
                pages=len(row["page_numbers"]),
                polls=row["polls"] + 1,
                failed_pages=len(kwargs.get("failures") or {}),
                code=kwargs.get("failure_code") or "",
            )
        return changed

    try:
        await throttle(engine, asyncio.get_running_loop().time() + 60)
        async with asyncio.timeout(60):
            outcome = await engine.recognizer.poll(
                SubmittedJob(
                    job_id=row["provider_job_id"],
                    request_id=row["request_id"],
                    page_numbers=tuple(row["page_numbers"]),
                    select_pages=row["select_pages"],
                )
            )
    except PermanentInputError:
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="failed",
            failure_code="permanent",
        )
        return
    except RetryableError as error:
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="pending",
            retry_after_seconds=max(error.retry_after_seconds or 0, backoff),
        )
        return
    except (UncertainOutcomeError, TimeoutError):
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="pending",
            retry_after_seconds=backoff,
        )
        return
    except OperatorActionRequiredError as error:
        log_event(
            log,
            "recognition_poll_blocked",
            logging.ERROR,
            error,
            recognition_id=row["id"],
        )
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="pending",
            retry_after_seconds=300,
        )
        return
    except Exception as error:
        log_event(
            log,
            "recognition_poll_error",
            logging.ERROR,
            error,
            recognition_id=row["id"],
        )
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="pending",
            retry_after_seconds=backoff,
        )
        return
    if not isinstance(outcome, JobPoll):
        log_event(
            log, "recognition_poll_invalid", logging.ERROR, recognition_id=row["id"]
        )
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="failed",
            failure_code="permanent",
        )
        return
    if outcome.state == "pending":
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="pending",
            retry_after_seconds=max(
                outcome.retry_after_seconds or 0, settings.recognition_poll_seconds
            ),
        )
    elif outcome.state == "expired":
        await finish(tenant_id=tenant, recognition_id=row["id"], state="expired")
    elif outcome.state == "failed":
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="failed",
            failure_code="permanent"
            if category(outcome.error) == "permanent"
            else "retryable",
        )
    else:
        result = outcome.result
        requested = sorted(row["page_numbers"])
        covered = sorted(
            [p.page_number for p in result.pages]
            + [f.page_number for f in result.failures]
        )
        limit = max_page_characters(settings)
        if covered != requested or any(len(p.text) > limit for p in result.pages):
            log_event(
                log,
                "recognition_result_invalid",
                logging.ERROR,
                recognition_id=row["id"],
            )
            await finish(
                tenant_id=tenant,
                recognition_id=row["id"],
                state="failed",
                failure_code="permanent",
            )
            return
        generation = await engine.metadata.get_generation(
            tenant_id=tenant, generation_id=row["generation_id"]
        )
        for page in result.pages:  # cache before marking done
            await cache_put(
                engine,
                tenant,
                generation.generation_id,
                generation.pipeline_fingerprint,
                page,
            )
        await finish(
            tenant_id=tenant,
            recognition_id=row["id"],
            state="done",
            failures={f.page_number: category(f.error) for f in result.failures},
        )
