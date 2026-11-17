"""Run publisher, scheduler, and workers with bounded concurrency and shutdown."""

import asyncio
from rag_ingest.logs import log_event
import logging
import random
from uuid import uuid4, uuid5
from rag_ingest.domain.events import JobAvailable
from rag_ingest.domain.enums import JobStatus as JS
from rag_ingest.contracts.errors import (
    RetryableError,
    UncertainOutcomeError,
    OperatorActionRequiredError,
    RecordVersionError,
    PermanentInputError,
    LeaseLostError,
    DuplicateContentError,
    AwaitingRecognitionError,
)
from rag_ingest.runtime import recognition
from rag_ingest.contracts.storage import CleanupObjectStore, ObjectRef
from rag_ingest.contracts.uploads import MultipartTransfer
from rag_ingest.contracts.vectors import VectorScope
from rag_ingest.retry_decision import decide_job_retry, RetryAction

log = logging.getLogger(__name__)
WARNING, ERROR, DEBUG = logging.WARNING, logging.ERROR, logging.DEBUG


def event(name, level=logging.INFO, exc_info=None, **fields):
    """Write structured diagnostics with identities, never document bodies or secrets."""
    log_event(log, name, level, exc_info, **fields)


# Outbox records sent per tenant per pass: one SQS batch call (its maximum).
OUTBOX_BATCH = 10
# Loop timings and per-pass bounds. Internal tuning, not settings: they trade a
# little latency against database and queue load, and are safe as they are.
SCHEDULER_INTERVAL_SECONDS = 2  # pause between scheduler and poller passes
VERIFIER_INTERVAL_SECONDS = 1  # pause between upload-verification passes
PUBLISHER_IDLE_SECONDS = 0.5  # pause when no notification was due
BACKOFF_SECONDS = 2  # pause after a dependency or unexpected error
HOUSEKEEPING_INTERVAL_SECONDS = 60  # history pruning, abandoned-upload cleanup
PASS_LIMIT = 100  # records per tenant per scheduler pass
OUTBOX_CLAIM_SECONDS = 120  # how long a publisher owns a claimed notification
QUEUE_SEND_TIMEOUT_SECONDS = 60  # bound on one queue send, including retries
SEND_FAILURES = (
    RetryableError,
    UncertainOutcomeError,
    TimeoutError,
    PermanentInputError,
)


def why(error):
    """Which dependency failed and how, for a pending warning.

    The messages are the package's own fixed texts; provider details are
    never chained into them (errors are raised `from None`).
    """
    return {"category": type(error).__name__, "reason": str(error)[:200]}


async def _send(queue, events):
    """One outcome per event (None = accepted), batched when the queue can."""
    batch = getattr(queue, "publish_batch", None)
    if callable(batch):
        outcomes = tuple(await batch(tuple(events)))
        if len(outcomes) != len(events):
            raise UncertainOutcomeError("Queue returned incomplete batch outcomes")
        return outcomes
    outcomes = []
    for item in events:
        try:
            await queue.publish(item)
            outcomes.append(None)
        except SEND_FAILURES as error:
            outcomes.append(error)
    return tuple(outcomes)


async def publish_once(engine, publisher_id):
    """Claim and send up to OUTBOX_BATCH notifications per tenant, in one queue call.

    No SQL lock is held over the queue call. Each record keeps its own claim:
    accepted ones are marked published, the others are rescheduled with
    backoff. Delivery stays at-least-once; workers are idempotent.
    """
    sent = 0
    for tenant in await engine.metadata.tenants():
        records = []
        for _ in range(OUTBOX_BATCH):
            record = await engine.metadata.claim_next_outbox(
                tenant_id=tenant,
                publisher_id=publisher_id,
                claim_id=uuid4(),
                claim_seconds=OUTBOX_CLAIM_SECONDS,
            )
            if not record:
                break
            records.append(record)
        if not records:
            continue
        try:
            # SDK calls have finite transport bounds; do not renew a stuck send forever.
            async with asyncio.timeout(QUEUE_SEND_TIMEOUT_SECONDS):
                outcomes = await _send(engine.queue, [r.event for r in records])
        except SEND_FAILURES as error:
            outcomes = (error,) * len(records)
        for record, outcome in zip(records, outcomes, strict=True):
            claim = record.claim
            if outcome is None:
                if await engine.metadata.mark_outbox_published(
                    tenant_id=tenant,
                    event_id=record.event.event_id,
                    publisher_id=publisher_id,
                    claim_id=claim.claim_id,
                ):
                    sent += 1
                continue
            try:
                await engine.metadata.reschedule_outbox_send(
                    tenant_id=tenant,
                    event_id=record.event.event_id,
                    publisher_id=publisher_id,
                    claim_id=claim.claim_id,
                    retry_id=uuid5(claim.claim_id, "reschedule"),
                    delay_seconds=max(
                        getattr(outcome, "retry_after_seconds", None) or 0,
                        min(60, 2 ** min(record.publish_attempts, 6))
                        * random.random(),  # noqa: S311  retry jitter, not crypto
                    ),
                    failure_code="queue_send_failed",
                )
            except LeaseLostError:
                pass
            event(
                "outbox_send_pending",
                WARNING,
                event_id=record.event.event_id,
                category=type(outcome).__name__,
            )
    return sent


async def _verify(engine, tenant, upload):
    """Verify one upload; only operator-level errors escape."""
    try:
        now = await engine.metadata.now(tenant)
        async with asyncio.timeout(max(1, (upload.expires_at - now).total_seconds())):
            await engine.verify_upload(tenant, upload.upload_id)
        event("upload_accepted", upload_id=upload.upload_id)
    except DuplicateContentError as error:
        event(
            "upload_duplicate",
            upload_id=upload.upload_id,
            duplicate_of=error.document_id,
        )
    except (PermanentInputError, ValueError):
        event("upload_verification_failed", WARNING, upload_id=upload.upload_id)
    except (RetryableError, UncertainOutcomeError, TimeoutError) as error:
        event(
            "upload_verification_pending",
            WARNING,
            upload_id=upload.upload_id,
            **why(error),
        )
    except OperatorActionRequiredError:
        raise
    except Exception as error:
        event("upload_verification_error", ERROR, error, upload_id=upload.upload_id)


async def verify_pass(engine, inflight, start=0):
    """Start verifications not already running, up to verify_concurrency.

    Hashing a large upload can take minutes, so it never runs inline in the
    scheduler loop; finished tasks are reaped here and new ones started.
    """
    for key, task in list(inflight.items()):
        if task.done():
            del inflight[key]
            task.result()  # re-raise operator errors; others were handled
    limit = engine.settings.verify_concurrency
    tenants = list(await engine.metadata.tenants())
    if tenants:  # rotate so early tenants cannot take every slot
        tenants = tenants[start % len(tenants) :] + tenants[: start % len(tenants)]
    for tenant in tenants:
        if len(inflight) >= limit:
            break
        for upload in await engine.metadata.list_verifying_uploads(tenant, limit=limit):
            key = (tenant, upload.upload_id)
            if len(inflight) >= limit:
                break
            if key not in inflight:
                inflight[key] = asyncio.create_task(_verify(engine, tenant, upload))


async def verify_once(engine):
    """Run one verification pass and wait for it (tests and custom loops)."""
    inflight = {}
    await verify_pass(engine, inflight)
    await asyncio.gather(*inflight.values())


async def run_verifier(engine, stop):
    """Verify completed uploads in a bounded pool beside the scheduler loop."""
    inflight, start = {}, 0
    try:
        while not stop.is_set():
            try:
                await verify_pass(engine, inflight, start)
            except (RetryableError, UncertainOutcomeError) as error:
                event("verifier_dependency_pending", WARNING, **why(error))
            start += 1
            await asyncio.sleep(VERIFIER_INTERVAL_SECONDS)
    finally:
        for task in inflight.values():
            task.cancel()
        await asyncio.gather(*inflight.values(), return_exceptions=True)


async def scheduler_once(engine):
    """Release due retries, schedule sealed batches and reconcile; never hashes."""
    for tenant in await engine.metadata.tenants():
        await engine.metadata.admit_generations(
            tenant_id=tenant, max_active=engine.settings.max_active_generations
        )
        await engine.metadata.release_due_retries(tenant_id=tenant, limit=PASS_LIMIT)
        for gen in await engine.metadata.scheduling_generations(tenant):
            await engine.metadata.release_inventory_batches(
                tenant_id=tenant, generation_id=gen, limit=PASS_LIMIT
            )
        await engine.metadata.reconcile(tenant, limit=PASS_LIMIT)


async def _heartbeat(engine, job, delivery, confirmed_at):
    """Renew ownership and queue visibility independently while processing runs.

    confirmed_at is a loop time no later than the last granted lease. A lease
    never outlives lease_seconds from its grant, so a failed renewal is
    tolerated until that bound passes; SQL fencing still guards every write.
    """
    loop = asyncio.get_running_loop()
    lease_seconds = job.budget.policy.lease_seconds
    while True:
        await asyncio.sleep(min(job.budget.policy.heartbeat_seconds, 5))
        sent_at = loop.time()
        try:
            lease = await engine.metadata.renew_job_lease(
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=job.lease.worker_id,
                lease_id=job.lease.lease_id,
                lease_seconds=lease_seconds,
            )
        except (RetryableError, UncertainOutcomeError):
            if loop.time() - confirmed_at >= lease_seconds:
                raise LeaseLostError(
                    "Lease may have expired while renewal failed"
                ) from None
            event("lease_renewal_pending", WARNING, job_id=job.job_id)
            continue
        if lease is None:
            raise LeaseLostError("Heartbeat could not confirm current ownership")
        confirmed_at = sent_at
        try:
            await engine.queue.extend_visibility(
                delivery, seconds=engine.settings.queue_visibility_seconds
            )
        except (RetryableError, UncertainOutcomeError, PermanentInputError):
            # SQL ownership prevents competing execution even if transport redelivers.
            event("queue_visibility_uncertain", WARNING, job_id=job.job_id)


async def _retry_or_fail(engine, job, worker_id, error, failure_code):
    """Apply the saved retry budget; return whether a durable outcome was recorded."""
    current = await engine.metadata.get_job(tenant_id=job.tenant_id, job_id=job.job_id)
    now = await engine.metadata.now(job.tenant_id)
    decision = decide_job_retry(
        current,
        error,
        worker_id=worker_id,
        lease_id=job.lease.lease_id,
        now=now,
        jitter_fraction=random.random(),  # noqa: S311  retry jitter, not crypto
    )
    event(
        "job_retry" if decision.action == RetryAction.RETRY else "job_retry_exhausted",
        WARNING,
        job_id=job.job_id,
        stage=job.stage,
        attempt=job.attempt,
        code=failure_code,
        category=type(error).__name__,
        reason=decision.reason_code,
        detail=str(error)[:200],  # the package's own message; provider text is never chained in
    )
    try:
        if decision.action == RetryAction.RETRY:
            await engine.metadata.schedule_job_retry(
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=worker_id,
                lease_id=job.lease.lease_id,
                retry_id=uuid5(job.lease.lease_id, "retry"),
                delay_seconds=decision.delay_seconds,
                failure_code=failure_code,
            )
            return True
        if decision.action == RetryAction.FAIL:
            await engine.metadata.fail_job(
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=worker_id,
                lease_id=job.lease.lease_id,
                failure_id=uuid5(job.lease.lease_id, "failed"),
                failure_code=decision.reason_code,
            )
            return True
    except LeaseLostError:
        pass
    return False


async def _fail(engine, job, worker_id, code):
    """Record a terminal failure under the current lease; return whether it held."""
    recorded = True
    try:
        await engine.metadata.fail_job(
            tenant_id=job.tenant_id,
            job_id=job.job_id,
            worker_id=worker_id,
            lease_id=job.lease.lease_id,
            failure_id=uuid5(job.lease.lease_id, "failed"),
            failure_code=code,
        )
    except LeaseLostError:
        recorded = False
    event("job_failed", WARNING, job_id=job.job_id, code=code)
    return recorded


async def worker_once(engine, worker_id, *, wait_seconds=10):
    """Receive one delivery, claim SQL authority, execute, and acknowledge durably."""
    deliveries = await engine.queue.receive(
        max_messages=1,
        wait_seconds=wait_seconds,
        visibility_seconds=engine.settings.queue_visibility_seconds,
    )
    if not deliveries:
        return False
    delivery = deliveries[0]
    try:
        notification = JobAvailable.from_json(delivery.body)
    except ValueError:
        await engine.metadata.quarantine_delivery(delivery, "malformed_event")
        await engine.queue.acknowledge(delivery)
        return True
    candidate = await engine.metadata.get_job(
        tenant_id=notification.tenant_id, job_id=notification.job_id
    )
    claimed_at = asyncio.get_running_loop().time()
    job = await engine.metadata.claim_job(
        tenant_id=notification.tenant_id,
        job_id=notification.job_id,
        worker_id=worker_id,
        lease_id=uuid4(),
        lease_seconds=candidate.budget.policy.lease_seconds
        if candidate
        else engine.settings.core.execution_policy.lease_seconds,
    )
    if job is None:
        current = await engine.metadata.get_job(
            tenant_id=notification.tenant_id, job_id=notification.job_id
        )
        if not current:
            await engine.metadata.quarantine_delivery(delivery, "missing_job")
            await engine.queue.acknowledge(delivery)
        elif current.status in (
            JS.SUCCEEDED,
            JS.FAILED,
            JS.CANCELLED,
            JS.RETRY_WAIT,
            JS.RUNNING,
            JS.QUEUED,  # waiting for admission: admit_generations re-notifies
        ):
            # Any crashed owner/missing continuation is repaired from durable SQL state.
            await engine.queue.acknowledge(delivery)
        return True
    event("job_started", job_id=job.job_id, stage=job.stage, attempt=job.attempt)
    started = asyncio.get_running_loop().time()
    work = asyncio.create_task(engine.execute_job(job))
    heart = asyncio.create_task(_heartbeat(engine, job, delivery, claimed_at))
    acknowledged = False
    try:
        now = await engine.metadata.now(job.tenant_id)
        remaining = max(0.001, (job.lease.attempt_deadline_at - now).total_seconds())
        # Leave a short interval to record a terminal timeout while ownership
        # is still valid. If SQL is unavailable, normal lease recovery applies.
        work_seconds = max(0.001, remaining - min(5.0, remaining * 0.1))
        async with asyncio.timeout(work_seconds):
            done, _ = await asyncio.wait(
                (work, heart), return_when=asyncio.FIRST_COMPLETED
            )
            if heart in done:
                await heart
            await work
        acknowledged = True
        event(
            "job_succeeded",
            job_id=job.job_id,
            stage=job.stage,
            attempt=job.attempt,
            seconds=round(asyncio.get_running_loop().time() - started, 2),
        )
    except LeaseLostError:
        event("job_ownership_lost", WARNING, job_id=job.job_id)
    except AwaitingRecognitionError:
        # Not a failure: park without a lease or a spent attempt; the scheduler
        # re-queues the job once every recorded provider job has finished.
        try:
            parked = await engine.metadata.park_job_waiting(
                tenant_id=job.tenant_id,
                job_id=job.job_id,
                worker_id=worker_id,
                lease_id=job.lease.lease_id,
                wait_id=uuid5(job.lease.lease_id, "wait"),
            )
            acknowledged = True
            event("job_waiting", job_id=job.job_id, status=parked.status)
        except LeaseLostError:
            pass
    except RetryableError as error:
        acknowledged = await _retry_or_fail(
            engine,
            job,
            worker_id,
            error,
            getattr(error, "code", None) or "temporary_dependency_failure",
        )
    except (UncertainOutcomeError, TimeoutError) as error:
        # Stage writes are lease-scoped, create-only or ledger-guarded, so a
        # repeat cannot duplicate state; its only cost is a possibly re-billed
        # model call. Repeats are bounded, counted, then need inspection.
        if job.attempt <= engine.settings.max_uncertain_repeats:
            event(
                "uncertain_outcome_repeat",
                WARNING,
                job_id=job.job_id,
                stage=job.stage,
                attempt=job.attempt,
                category=type(error).__name__,
            )
            acknowledged = await _retry_or_fail(
                engine,
                job,
                worker_id,
                RetryableError("Uncertain outcome"),
                "uncertain_outcome_repeat",
            )
        else:
            acknowledged = await _fail(
                engine, job, worker_id, "outcome_requires_inspection"
            )
    except (
        PermanentInputError,
        ValueError,
        OperatorActionRequiredError,
    ) as error:
        if isinstance(error, RecordVersionError):
            raise  # this release is too old; the job is fine for a newer one
        code = getattr(error, "code", None) or (
            "operator_action_required"
            if isinstance(error, OperatorActionRequiredError)
            else "input_or_pipeline_validation_failed"
        )
        acknowledged = await _fail(engine, job, worker_id, code)
        if isinstance(error, OperatorActionRequiredError):
            raise
    except Exception as error:
        # Unclassified errors (bugs, unmapped SDK errors) get a bounded durable
        # retry instead of escaping and cancelling every other worker slot.
        event(
            "job_unexpected_error",
            ERROR,
            error,
            job_id=job.job_id,
            category=type(error).__name__,
        )
        acknowledged = await _retry_or_fail(
            engine,
            job,
            worker_id,
            RetryableError("Unexpected worker error"),
            "unexpected_worker_error",
        )
    finally:
        work.cancel()
        heart.cancel()
        await asyncio.gather(work, heart, return_exceptions=True)
    if acknowledged:
        await engine.queue.acknowledge(delivery)
        event("delivery_acknowledged", DEBUG, job_id=job.job_id)
    return True


async def run_publisher(engine, stop):
    """Drain due outbox records and pause briefly when no work was available."""
    identity = "publisher-" + str(uuid4())
    while not stop.is_set():
        try:
            if not await publish_once(engine, identity):
                await asyncio.sleep(PUBLISHER_IDLE_SECONDS)
        except (RetryableError, UncertainOutcomeError) as error:
            event("publisher_dependency_pending", WARNING, **why(error))
            await asyncio.sleep(BACKOFF_SECONDS)


async def prune_once(engine, *, batch=1000, max_batches=10):
    """Remove expired history in bounded batches; return rows deleted."""
    total = 0
    for _ in range(max_batches):
        deleted = await engine.metadata.prune_history(
            older_than_days=engine.settings.retention_days, limit=batch
        )
        total += deleted
        if deleted < batch:
            break
    if total:
        event("history_pruned", rows=total)
    return total


async def run_recognition_poller(engine, stop):
    """Poll job-style recognizers beside the scheduler loop; never blocks it."""
    while not stop.is_set():
        try:
            await recognition.poll_once(engine)
        except (RetryableError, UncertainOutcomeError) as error:
            event("recognition_poller_pending", WARNING, **why(error))
        except OperatorActionRequiredError:
            raise
        except Exception as error:
            event("recognition_poller_error", ERROR, error)
        await asyncio.sleep(SCHEDULER_INTERVAL_SECONDS)


async def run_scheduler(engine, stop):
    """Keep verification, retry, scheduling, and recovery work moving.

    Verification runs as its own task so a long hash never delays recovery.
    """

    async def loop():
        clock = asyncio.get_running_loop()
        next_prune = 0.0
        while not stop.is_set():
            try:
                await scheduler_once(engine)
                await cleanup_once(engine)
                if clock.time() >= next_prune:
                    await prune_once(engine)
                    await purge_abandoned_uploads(engine)
                    next_prune = clock.time() + HOUSEKEEPING_INTERVAL_SECONDS
            except (RetryableError, UncertainOutcomeError) as error:
                event("scheduler_dependency_pending", WARNING, **why(error))
            await asyncio.sleep(SCHEDULER_INTERVAL_SECONDS)

    async with asyncio.TaskGroup() as group:
        group.create_task(loop())
        group.create_task(run_verifier(engine, stop))
        recognizer = getattr(engine, "recognizer", None)
        if recognizer is not None and recognizer.capabilities.call_style == "job":
            group.create_task(run_recognition_poller(engine, stop))


async def run_workers(engine, stop):
    """Run a fixed number of polling workers instead of unbounded task fan-out."""

    async def worker():
        """Keep one delivery in flight per configured worker slot."""
        id = "worker-" + str(uuid4())
        while not stop.is_set():
            try:
                await worker_once(engine, id)
            except (RetryableError, UncertainOutcomeError) as error:
                event("worker_dependency_pending", WARNING, worker_id=id, **why(error))
                await asyncio.sleep(BACKOFF_SECONDS)
            except OperatorActionRequiredError:
                raise
            except Exception as error:
                event(
                    "worker_unexpected_error",
                    ERROR,
                    error,
                    worker_id=id,
                    category=type(error).__name__,
                )
                await asyncio.sleep(BACKOFF_SECONDS)

    async with asyncio.TaskGroup() as group:
        for _ in range(engine.settings.worker_concurrency):
            group.create_task(worker())


async def purge_abandoned_uploads(engine, *, limit=PASS_LIMIT):
    """Erase files left by uploads that were never accepted.

    A refused duplicate, a failed verification or an expired upload can leave
    bytes (or unfinished multipart parts) under that upload's own unique key.
    No source version ever references such a key, so it is erased once the
    session expired more than the grace period ago, then marked so it is done
    once. Storage failures propagate; the next pass retries.
    """
    grace = engine.settings.abandoned_upload_grace_seconds
    removed = 0
    for tenant in await engine.metadata.tenants():
        for upload, plan in await engine.metadata.abandoned_uploads(
            tenant_id=tenant, grace_seconds=grace, limit=limit
        ):
            ref = ObjectRef(bucket=upload.bucket, key=upload.object_key)
            if plan:  # stored parts of an unfinished multipart upload are billed too
                await engine.uploads.abort_multipart(
                    MultipartTransfer(target=ref, provider_upload_id=plan.provider_upload_id)
                )
            await engine.objects.delete_all_key_versions(ref)
            if await engine.metadata.mark_upload_files_removed(
                tenant_id=tenant, upload_id=upload.upload_id
            ):
                removed += 1
                event(
                    "upload_files_removed",
                    upload_id=upload.upload_id,
                    status=upload.status.value,
                    duplicate=bool(upload.duplicate_of),
                )
    return removed


async def cleanup_once(engine):
    """Remove retired generation outputs only after durable authorization and grace.

    Original source erasure is performed through an explicit, reference-safe
    cleanup plan. This loop does not guess whether shared artifacts are unused.
    """
    supported = isinstance(engine.objects, CleanupObjectStore)
    for tenant in await engine.metadata.tenants():
        for task in await engine.metadata.cleanup_candidates(tenant):
            if not supported:
                # Never authorize: that marks artifacts "deleting" and would block
                # re-uploads of the same bytes until a cleanup that cannot run.
                event(
                    "cleanup_blocked_unsupported_store",
                    ERROR,
                    document_id=task["document_id"],
                )
                continue
            plan = await engine.metadata.authorize_cleanup(tenant, task["id"])
            if plan is None:
                continue
            for gen in plan["generations"]:
                scope = VectorScope(
                    tenant_id=tenant, document_id=task["document_id"], generation_id=gen
                )
                await engine.vectors.delete_generation(scope)
                if engine.keyword is not None:
                    await engine.keyword.delete_generation(scope)
                await engine.metadata.delete_search_chunks(
                    tenant_id=tenant, generation_id=gen
                )
                await engine.objects.delete_prefix_versions(
                    engine.settings.bucket, f"tenants/{tenant}/generations/{gen}/"
                )
            for ref in plan["sources"]:
                await engine.objects.delete_all_key_versions(ref)
            await engine.metadata.finish_cleanup_plan(tenant, task["id"], plan)
            event("cleanup_completed", document_id=task["document_id"])
