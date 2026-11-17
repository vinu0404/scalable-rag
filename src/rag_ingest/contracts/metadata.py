"""Define atomic metadata operations, starting with upload intake.

These are database contracts, not SQL implementations. Implementations must
enforce tenant boundaries, constraints, and expected-state checks inside their
transactions. Never hold a database transaction open during storage calls.
Retrieval must enforce current document permissions, tombstones, and active
generation pointers; storing those fields alone does not enforce visibility.
"""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from rag_ingest.contracts.storage import ObjectRef
from rag_ingest.domain.artifacts import ContentArtifact
from rag_ingest.domain.checkpoints import StageCheckpoint
from rag_ingest.domain.documents import Document, ProcessingGeneration, SourceVersion
from rag_ingest.domain.inventory import BatchPlan, GenerationInventory
from rag_ingest.domain.jobs import Job, JobLease
from rag_ingest.domain.manifests import ManifestOutput
from rag_ingest.domain.multipart import MultipartPlan, ObservedUploadPart
from rag_ingest.domain.pipeline import PipelineConfig
from rag_ingest.domain.outbox import OutboxRecord, PublisherClaim
from rag_ingest.domain.uploads import UploadSession


class MetadataStore(Protocol):
    """Specify durable operations without tying the core to an ORM.

    The service supplies trusted tenant context after authentication. Adapters
    must still scope every operation by tenant and check record relationships.
    Use database time for deadlines. Map known dependency failures to shared
    error categories; never report a successful commit when its outcome is
    uncertain. Concurrent calls must obey each operation's atomic guarantees.
    """

    async def initiate_upload(
        self,
        *,
        upload: UploadSession,
        pipeline: PipelineConfig,
        request_fingerprint: str,
        expected_desired_generation_id: UUID | None,
        new_document: Document | None = None,
    ) -> UploadSession:
        """Save intake state and its processing recipe in one transaction.

        For a new document, save new_document with the session. Otherwise
        require an existing, nondeleted document in this tenant. Do not create
        an ingestion job or outbox notification before file verification.

        Compare the document's desired generation with the supplied expected
        value and save that baseline with intake. A new document requires None.
        On a mismatch raise ValueError; do not silently rebase a replacement.
        A matching idempotent retry returns its saved session and baseline.

        Enforce uniqueness on (tenant_id, request_key). Persist a SHA-256
        request_fingerprint covering client-intended file details, target
        document choice, and pipeline config. Exclude newly generated IDs,
        timestamps, and storage keys from that fingerprint.

        On a matching retry, return the original session without creating new
        records, changing its expiry, or replacing its recipe. On incompatible
        key reuse, raise ValueError with a safe message. A terminal session
        stays terminal; starting another upload requires another request key.
        """
        ...

    async def get_upload(
        self, *, tenant_id: UUID, upload_id: UUID
    ) -> UploadSession | None:
        """Read the current session in this tenant, or None if it is absent.

        This is a snapshot, not permission to mutate it. Later writes must
        check their required current state inside the database operation.
        """
        ...

    async def get_upload_pipeline(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
    ) -> PipelineConfig | None:
        """Read the complete processing recipe saved with this upload.

        Return None for an absent session. Restore the saved configuration
        schema explicitly; never substitute today's defaults when resuming.
        """
        ...

    async def attach_multipart_plan(self, plan: MultipartPlan) -> bool:
        """Atomically attach a provider transfer and fixed plan to its session.

        Require UPLOADING state, an unexpired session, a nondeleted document,
        matching expected file size, and no different attached transfer/plan.
        Save the plan and session.multipart_upload_id together. Validate tested
        provider limits in the service before calling this operation.

        Return True for a new attachment or an identical existing plan while
        the session is still eligible. Return False for absence or state/plan
        conflict. Never replace the winning transfer. The caller must arrange
        cleanup of any losing provider transfer created outside this transaction.
        """
        ...

    async def get_multipart_plan(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
    ) -> MultipartPlan | None:
        """Read this session's attached plan for resume or cleanup decisions.

        Returning a plan does not establish that its session is still eligible
        for new part URLs. The signing service must check current intake state.
        """
        ...

    async def record_observed_part(self, part: ObservedUploadPart) -> bool:
        """Save provider-reported part progress against the current plan.

        Require an eligible UPLOADING session and call validate_against on
        its attached plan. Isolate rows by tenant, upload, provider transfer,
        and part number. Identical repeated observations are harmless.

        Reject older observations and conflicting observations with equal
        timestamps; return False for stale progress or state mismatch. Invalid
        plan/size data raises ValueError. Return True when saved or already
        recorded identically. Timestamps cannot establish storage truth: even
        cached newer observations must be reconciled before completion.
        """
        ...

    async def list_observed_parts(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
        provider_upload_id: str,
        after_part_number: int | None,
        limit: int,
    ) -> tuple[ObservedUploadPart, ...]:
        """Read a bounded ordered page of recorded progress for one transfer.

        Require a positive bounded limit. Return only part numbers greater
        than after_part_number, or start at the beginning when it is None.
        Use the last returned number to request another page. This is cached
        progress, not a replacement for the provider's list_parts operation.
        """
        ...

    async def begin_verification(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
        ref: ObjectRef,
    ) -> bool:
        """Freeze the completed object reference and mark the session VERIFYING.

        The service must first confirm completion and resolve a pinned version
        or establish write-once protection. Require the assigned bucket/key,
        an unexpired UPLOADING session, and a nondeleted document. Persist ref
        and the status change together, without creating processing work.

        Return True for that transition or an identical VERIFYING retry. Return
        False for absence, expiry, other states, or a different frozen reference.
        Never replace an already frozen reference. Checksums are computed
        outside this database transaction over exactly those protected bytes.
        """
        ...

    async def get_verification_ref(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
    ) -> ObjectRef | None:
        """Read the exact object reference saved for verification and recovery.

        Return None when no reference was registered in this tenant. Keeping
        a reference does not establish that verification or acceptance passed.
        """
        ...

    async def find_content_artifact(
        self,
        *,
        tenant_id: UUID,
        sha256: str,
    ) -> ContentArtifact | None:
        """Look up verified reusable content with a tenant-scoped indexed hash.

        Do not scan stored files. The caller must use a trusted computed hash
        and compare size. This snapshot is not a reservation against cleanup;
        acceptance must atomically resolve deduplication and attach references.
        """
        ...

    async def accept_verified_upload(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
        verified_ref: ObjectRef,
        artifact_candidate: ContentArtifact,
        source: SourceVersion,
        generation: ProcessingGeneration,
        first_job: Job,
        outbox: OutboxRecord,
    ) -> UploadSession | None:
        """Register verified content and its first processing work atomically.

        The service calls only after checking actual size, trusted SHA-256,
        supported content, and intake policy. This operation does not read S3.
        Require the frozen reference, VERIFYING state, unexpired intake, a
        nondeleted document, and the desired-generation baseline saved at
        initiation. Reject a stale baseline instead of overwriting newer work.

        Validate every tenant/document/source/generation/job/event relationship.
        Require source storage to match verified_ref and source size/hash to
        match verification expectations and artifact_candidate. The generation
        fingerprint must match the full recipe saved at initiation. Require a
        fresh queued EXTRACT job and an unpublished, unclaimed outbox event
        for that job. Invalid candidate data raises ValueError safely.

        Resolve UNIQUE (tenant_id, sha256) races inside this transaction. If
        identical content already exists, check size, ensure its retained
        bytes are not being deleted, and attach that artifact's canonical ID
        to the stored source rather than persisting a losing candidate ID.
        Coordinate new references with cleanup; keep document ACLs separate.

        In the same commit: save the source and generation, reference the
        artifact, save the first job and outbox event, set desired_generation,
        and mark the session ACCEPTED with source_version_id. Preserve the
        existing active_generation until safe publication later.

        Return the accepted session. Repeated acceptance of the same frozen
        verified content returns the original accepted session without making
        more work or resetting desired_generation. Resolve original IDs using
        the upload's durable registration, not freshly generated retry IDs.
        Return None for absence or current-state conflicts; never partially
        commit registration. Surface uncertain commits for reconciliation.
        """
        ...

    async def reject_verified_upload(
        self,
        *,
        tenant_id: UUID,
        upload_id: UUID,
        verified_ref: ObjectRef,
        reason_code: str,
    ) -> bool:
        """Record permanent verification failure for the same frozen object.

        Use a safe, bounded reason code rather than file text or credentials.
        Atomically change a matching VERIFYING session to FAILED and save the
        reason. Return True for the change or the identical recorded failure;
        False for absence or state/reference mismatch. Never overwrite an
        accepted session, enqueue processing, or delete S3 objects here.
        """
        ...

    async def get_job(self, *, tenant_id: UUID, job_id: UUID) -> Job | None:
        """Read a job's current details in this tenant, or None if absent.

        Useful for status reporting and inspecting a duplicate notification.
        Reading RUNNING or QUEUED does not grant ownership. Never implement a
        claim as this read followed by an unconditional update.
        """
        ...

    async def claim_job(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        lease_seconds: int,
    ) -> Job | None:
        """Atomically acquire eligible work and return its job with a lease.

        The runner generates a fresh UUID for a new acquisition request. If
        the database response is lost, reuse that UUID only to resolve the
        same request, not to create another acquisition. Never reuse an old
        lease ID for a later ownership attempt.

        Check the tenant, job/document/generation relationships, nondeleted
        document, and still-desired generation in the same transaction. Accept
        QUEUED work or RUNNING work with an expired lease. RETRY_WAIT work
        must first pass through release_due_retries, even when its time has
        arrived. Check the job's durable attempt budget and execution deadline.
        Do not acquire terminal jobs, waiting retries, or another unexpired lease.

        On a new claim, set RUNNING, clear next_attempt_at, increment attempt,
        and persist worker_id, lease_id, and database-time expiry together.
        Bound positive lease_seconds by configured limits and the job deadline.
        Concurrent claimers must not both win for the same current interval.

        An identical replay of the current unexpired acquisition returns that
        job without incrementing attempts or extending expiry. Replays of an
        expired or replaced acquisition return None; retain sufficient durable
        acquisition history to prevent old IDs becoming new ownership.
        Return None for absence or ineligibility. Return only committed state;
        raise UncertainOutcomeError if commit cannot be determined.

        None does not mean 'completed' and does not itself authorize deleting
        the queue message. The runner must follow durable recovery policy.
        """
        ...

    async def renew_job_lease(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        lease_seconds: int,
    ) -> JobLease | None:
        """Extend only this attempt's current, unexpired database ownership.

        Atomically require RUNNING status, matching worker and lease IDs, an
        unexpired lease, an undeleted document, the still-desired generation,
        and an unexpired execution deadline. Compute expiry from database time
        with configured duration limits. Never revive an expired lease, change
        its ID, or increment the attempt count during renewal.

        Return the committed lease, or None if ownership is no longer valid.
        Surface database failures separately; an uncertain response is not a
        confirmed renewal. The runner must pause authoritative writes unless
        it can establish current ownership. Recheck ownership inside every
        progress or completion write even after a successful heartbeat.

        This does not extend SQS visibility. The runner must maintain queue
        delivery separately and enforce stage and overall processing deadlines.
        """
        ...

    async def get_checkpoint(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
    ) -> StageCheckpoint | None:
        """Read the completed stage/batch checkpoint for this job, if present.

        This record is a reuse candidate, not proof its files remain intact.
        The worker must match source, exact stage input, pipeline settings,
        work identity, and manifest integrity before using saved outputs.
        """
        ...

    async def save_checkpoint(
        self,
        *,
        checkpoint: StageCheckpoint,
        worker_id: str,
        lease_id: UUID,
    ) -> StageCheckpoint:
        """Save completed output metadata only while this attempt owns the job.

        The service first saves and verifies the immutable manifest and its
        outputs outside the transaction. Inside the transaction, require the
        current unexpired lease, RUNNING status, valid deadline, nondeleted
        document, and still-desired generation. Raise LeaseLostError if this
        attempt no longer has authority; never save stale progress anyway.

        Check checkpoint.producer_lease_id equals lease_id, plus all tenant,
        document, generation, stage, shard, source, input, and pipeline links
        against registered work. Enforce one completed checkpoint per job.
        Invalid relationships or incompatible existing output raise ValueError.

        An equivalent existing checkpoint returns the original record without
        replacing its immutable output references or producer identity. Its
        producer can be an earlier lease; current ownership still governs this
        call. Return only committed state, and surface uncertain commits.

        This does not finish the job, create next work, activate search data,
        or authorize queue acknowledgment. A crash after saving is recoverable
        by validating this checkpoint and completing the stage under a new lease.
        """
        ...

    async def complete_stage(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        checkpoint_id: UUID,
        completion_id: UUID,
        next_jobs: tuple[Job, ...],
        next_outbox: tuple[OutboxRecord, ...],
    ) -> Job:
        """Finish one stage and register its required continuation atomically.

        completion_id identifies this exact completion request. Persist its
        request identity and payload binding with the result. On a lost-response
        retry, the same ID and equivalent payload return the recorded success
        without creating more work, even though that completed job has released
        its lease. Reusing the ID with different data raises ValueError. This
        replay only resolves committed work; it must never revive or republish it.

        For a new completion, require the current unexpired matching lease,
        RUNNING status, valid deadline, nondeleted document, and still-desired
        generation. Require a saved validated checkpoint for this exact job;
        it may have been produced by an earlier attempt. Raise LeaseLostError
        on lost authority. PUBLISH jobs use a separate publication operation.

        Validate next_jobs and next_outbox against the saved dependency plan,
        not just caller-supplied lists. New jobs must be queued with matching
        tenant/document/generation and stable stage/shard identities. Each new
        runnable job needs its matching unpublished, unclaimed outbox event.
        Enforce job uniqueness and prerequisite readiness inside this transaction.

        For parallel embedding batches, record which expected batches finished
        and release the indexing continuation only when all prerequisites are
        complete. Concurrent final batches must not create duplicate downstream
        jobs. Never treat an empty continuation list as proof the whole pipeline
        finished. Bound fan-out and reject incomplete scheduling plans rather
        than silently dropping required work.

        For CHUNK completion, a sealed inventory with durable batch-scheduling
        intent can represent the continuation instead of inserting every batch
        job here. Validate that intent before allowing empty next_jobs/outbox.
        The batch scheduler must wait until this chunking job is SUCCEEDED.

        Commit job SUCCEEDED with its lease cleared, completion history,
        dependency progress, next jobs, and outbox records together. Do not
        activate the generation or delete queue messages here. A worker may
        acknowledge its delivery only after this durable outcome is established.
        Surface uncertain commits so the runner resolves completion_id before
        deciding whether to repeat work or acknowledge the delivery.
        """
        ...

    async def schedule_job_retry(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        retry_id: UUID,
        delay_seconds: float,
        failure_code: str,
    ) -> Job:
        """Durably schedule another attempt after a known temporary failure.

        The core chooses a finite nonnegative delay using bounded backoff,
        jitter, and any provider minimum delay. Validate against the saved
        retry policy. Do not silently shorten provider guidance; if no allowed
        retry fits the execution deadline, record terminal failure instead.

        For a new request, atomically require this worker's matching unexpired
        lease, RUNNING status, a nondeleted document, and the still-desired
        generation. Raise LeaseLostError on lost authority. Recheck the saved
        attempt budget and overall deadline using database time.

        If another attempt is permitted, set RETRY_WAIT, clear the lease, save
        next_attempt_at from database time plus the chosen delay, and append
        safe failure/attempt history. Do not increment attempt here; claiming
        the next execution increments it. Preserve valid checkpoints.

        If the budget is exhausted or no retry fits the deadline, set FAILED,
        clear lease and next_attempt_at, and save the exhaustion reason and
        required terminal disposition instead. Block dependent work from
        being released; preserve any previous active document generation.

        Commit retry_id, its payload binding, and the resulting transition in
        the same transaction. An identical replay resolves the stored outcome
        without postponing the retry, overwriting newer attempts, or incrementing
        counters. Reject incompatible ID reuse with ValueError. Return that
        committed transition snapshot, which may no longer be current state.
        Raise UncertainOutcomeError if the commit cannot be established.

        The database scheduler owns delayed retries. It will atomically create
        the due attempt's notification through the outbox. Do not also enqueue
        a delayed queue retry or sleep inside this transaction. A runner may
        acknowledge the old delivery only after its durable disposition is
        established. This method does not call SQS.

        Do not use this as automatic handling of UncertainOutcomeError: first
        verify the remote outcome or establish that repeating is safe. Use a
        bounded safe failure_code, never raw file text or provider credentials.
        """
        ...

    async def fail_job(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        failure_id: UUID,
        failure_code: str,
    ) -> Job:
        """Record a terminal failure for work that should not retry unchanged.

        For a new request, atomically require RUNNING status, the exact worker
        and unexpired lease, a nondeleted document, and its desired generation.
        Raise LeaseLostError when this attempt no longer has authority. The
        execution deadline may already be exceeded; current lease ownership
        still controls this write. An expired owner cannot fail someone else's
        work even when it discovers an actual processing error.

        Set FAILED, clear lease and next_attempt_at, and persist bounded safe
        failure_code, attempt history, and required terminal disposition in
        one commit. Prevent unfinished dependencies from advancing; preserve
        completed artifacts for the configured recovery/retention policy and
        keep any old active generation available. Do not publish a partial build.

        Save failure_id and its payload binding with the committed outcome.
        Identical retries resolve that outcome without changing later state;
        incompatible ID reuse raises ValueError. Return the recorded transition
        snapshot, not a promise that it is still current. Surface uncertain
        commits for resolution before queue acknowledgment.

        Use for permanent input errors or deliberate terminal policy outcomes.
        Missing credentials require operational intervention, not declaring
        every input corrupt. LeaseLostError must stop the attempt, not trigger
        another failure write. Reconciliation handles abandoned or obsolete jobs
        through separate conditional operations. This method neither deletes
        queue messages nor deletes stored files.
        """
        ...

    async def release_due_retries(
        self,
        *,
        tenant_id: UUID,
        limit: int,
    ) -> tuple[OutboxRecord, ...]:
        """Make a bounded batch of due retries runnable and save their notifications.

        The scheduler calls this repeatedly for authorized tenants. Use a
        positive limit bounded by deployment policy. Select at most that many
        RETRY_WAIT jobs with next_attempt_at at or before database time, using
        the due-time index and safe concurrent row claims. Never scan all jobs
        into application memory or hold locks while talking to SQS.

        In the same transaction, recheck each selected job's state, tenant,
        document, generation, attempt budget, and execution deadline. Eligible
        jobs become QUEUED with next_attempt_at cleared and no lease. Preserve
        attempt count; only the subsequent worker acquisition increments it.

        For each released job, create one fresh JobAvailable event in the
        outbox. Bind the notification uniquely to that job's durable retry_id
        from schedule_job_retry. A later retry cycle gets another identity;
        concurrent schedulers and ambiguous-commit retries must not create
        multiple notifications for the same release cycle. A published event
        can still be delivered repeatedly by the transport.

        Commit QUEUED state, outbox intent, and release history together. If
        the document was deleted or the generation superseded, conditionally
        cancel this waiting work instead. If its budget or deadline is spent,
        record FAILED and terminal disposition. Do not create runnable work
        for those cases or change any active document generation.

        Return committed outbox records for jobs actually released. This is
        informational: the normal publisher reads the outbox independently.
        Empty output does not prove no backlog; jobs may have been cancelled,
        failed, or handled concurrently. A lost response can be followed by
        another bounded call because saved state and retry-cycle uniqueness
        prevent repeating a release. Surface uncertain commits rather than
        pretending they failed, and never send notifications inside this method.
        """
        ...

    async def get_outbox_record(
        self,
        *,
        tenant_id: UUID,
        event_id: UUID,
    ) -> OutboxRecord | None:
        """Read current delivery intent to inspect or resolve an uncertain outcome.

        Return None only for an absent record in this tenant. A snapshot does
        not grant publishing ownership. published_at records queue acceptance,
        not document processing success.
        """
        ...

    async def claim_next_outbox(
        self,
        *,
        tenant_id: UUID,
        publisher_id: str,
        claim_id: UUID,
        claim_seconds: int,
    ) -> OutboxRecord | None:
        """Atomically acquire one due notification for this publisher attempt.

        Use the indexed ready set: published_at is NULL, available_at is due
        by database time, and no unexpired publisher claim exists. Select and
        lock safely so concurrent publishers cannot acquire the same current
        claim. Bound work and never load the whole outbox into memory.

        For a new acquisition, save publisher_id, fresh claim_id, bounded
        positive database-time expiry, and publish_attempts + 1 together.
        Persist successful acquisition identity so a lost-response retry with
        the same claim_id returns its original still-valid claim rather than
        acquiring another event. Expired or replaced acquisition replays return
        None; they cannot reacquire work using their old ID. The caller uses
        a fresh UUID for each new acquisition request.

        Return the committed claimed record or None if no claim was obtained.
        Sending happens after the transaction ends. Do not mark published,
        alter job status, or assume queue delivery is exclusive. Even an event
        whose job became obsolete can be delivered safely because workers
        must validate authoritative job and document state.
        """
        ...

    async def renew_outbox_claim(
        self,
        *,
        tenant_id: UUID,
        event_id: UUID,
        publisher_id: str,
        claim_id: UUID,
        claim_seconds: int,
    ) -> PublisherClaim | None:
        """Extend only the matching current, unexpired publishing claim.

        Require an unpublished record and compute expiry from database time
        with configured duration limits. Do not increment publish_attempts or
        revive expired ownership. Return the committed claim or None on lost
        ownership. Surface dependency failures; uncertainty is not confirmation.
        Bound total send-attempt duration so heartbeats cannot hide stuck work.
        This does not renew a processing job lease or SQS message visibility.
        """
        ...

    async def mark_outbox_published(
        self,
        *,
        tenant_id: UUID,
        event_id: UUID,
        publisher_id: str,
        claim_id: UUID,
    ) -> bool:
        """Record a confirmed queue send under the current publisher claim.

        Call only after Queue.publish confirms acceptance. Atomically require
        an unpublished record with matching publisher, claim ID, and unexpired
        ownership. Set published_at from database time, clear the claim, and
        record the successful claim identity in publication history together.

        Return True for that commit or an identical previously committed
        outcome for this acquisition. Return False for absence or lost/replaced
        authority. A successful send followed by a rejected publication update
        can still be sent again later; never bypass ownership to suppress it.
        Raise UncertainOutcomeError when the database commit is unresolved.

        Neither this transaction nor a rollback can undo a queue send. A crash
        after sending but before this commit leaves recoverable intent and may
        cause a duplicate notification. Worker idempotency remains required.
        """
        ...

    async def reschedule_outbox_send(
        self,
        *,
        tenant_id: UUID,
        event_id: UUID,
        publisher_id: str,
        claim_id: UUID,
        retry_id: UUID,
        delay_seconds: float,
        failure_code: str,
    ) -> OutboxRecord:
        """Release a failed send attempt and durably schedule another attempt.

        Require the matching current unexpired publisher claim and an
        unpublished record; raise LeaseLostError on lost publishing authority.
        Validate a finite nonnegative bounded delay and safe failure code.
        Atomically clear the claim, set available_at using database time plus
        delay, and save retry_id, payload binding, and failure history. Preserve
        the event ID, body, and publish_attempts count. The next successful
        acquisition increments that count.

        An identical retry_id resolves the recorded transition without moving
        its retry time or overwriting newer state. Reject conflicting ID reuse.
        Return the recorded snapshot; it need not still be the current record.
        Surface uncertain database commits for resolution.

        A queue send with unknown outcome can be retried with the same event
        identity because delivery is designed to tolerate duplicates. Record
        that uncertainty; never label it confirmed nondelivery or generate a
        new event ID to retry. Do not alter the ingestion job retry counter.

        Apply bounded per-attempt retries and operational backoff/alerts for
        sustained outages. Retain unresolved delivery intent: retry exhaustion
        or bad credentials must not silently mark it published or discard it.
        Any pause/quarantine and replay policy must stay durable and observable.
        This operation does not call SQS or sleep in the database transaction.
        """
        ...

    async def get_document(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Read document state, including a deletion tombstone, in this tenant.

        Return None for absence, not merely deletion. Status and recovery paths
        need to distinguish deleted records from missing ones. The service
        must enforce the document's access policy before exposing it to a user.
        """
        ...

    async def publish_generation(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        generation_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        publication_id: UUID,
        expected_manifest_sha256: str,
        verified_chunk_count: int,
        verified_chunk_ids_sha256: str,
        vector_target: str,
        verified_at: datetime,
    ) -> Document:
        """Activate a complete, still-desired generation under its publish lease.

        Before calling, the trusted publication service must verify the sealed
        generation manifest, exact expected chunk identities, and queryability
        in the intended vector target. A count alone is insufficient. The ID
        digest must use the versioned canonical encoding from the manifest.
        This operation records that evidence; it does not query a vector store.

        Inside one transaction, require a RUNNING PUBLISH job for this exact
        tenant/document/generation, matching worker and unexpired lease, a valid
        execution deadline, completed durable prerequisites, and a nondeleted
        document whose desired_generation_id still equals generation_id.
        Raise LeaseLostError when the attempt no longer has authority.

        Compare manifest hash, expected count, exact ID-set digest, source and
        pipeline identity, and vector target with the saved sealed build plan.
        Validate verification time and configured freshness policy. Reject
        incomplete or mismatched evidence with ValueError. Outputs must remain
        protected from incompatible mutation between verification and use;
        PostgreSQL cannot make remote storage part of this transaction.

        Atomically set active_generation_id, finish the publication job as
        SUCCEEDED with its lease cleared, save evidence and publication_id with
        payload binding, and register durable retention/cleanup intent for the
        previous generation. Preserve the current desired-generation pointer.

        An identical publication_id replay resolves its committed outcome
        without switching the pointer again, recreating cleanup work, or
        reviving a deleted document. Reject incompatible ID reuse. Return the
        current Document snapshot; a newer generation or deletion may already
        have superseded the original success. Surface unresolved commits.

        Retrieval must enforce active generation, live access rules, and
        deletion on every supported path, including caches. This operation
        alone does not filter vector search or acknowledge queue delivery.
        """
        ...

    async def tombstone_document(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        deletion_id: UUID,
    ) -> Document | None:
        """Logically delete a document before asynchronously removing its data.

        The service must first authorize deletion. In one transaction, lock
        the matching document, set deleted_at from database time, clear active
        and desired generation pointers, and persist deletion_id, cancellation
        intent, and cleanup intent. Keep document identity and deletion history.
        Publication must serialize against this same state so late work cannot
        activate a deleted document. Do not reuse its ID for a new upload.

        Record cancellation intent durably rather than updating an unbounded
        number of jobs in this transaction. Every worker claim, heartbeat,
        upload acceptance, and progress/publication write must independently
        enforce the tombstone while cancellation catches up.

        Return None if absent in this tenant. Repeated deletion returns its
        existing tombstone without resetting deletion time or duplicating
        cleanup. Resolve a repeated deletion_id idempotently; reject reuse for
        a different target. A missing response must not undo logical deletion.

        Do not delete S3 objects or vectors here. Cleanup must honor retention,
        shared-artifact references, concurrent attachments, and backup policy.
        File-view and retrieval services must reject new access using current
        permissions and tombstones. Previously issued presigned URLs require
        the separate delivery/access-removal policy; changing this row does
        not itself invalidate remote grants or erase physical copies.
        """
        ...

    async def seal_generation_inventory(
        self,
        *,
        inventory: GenerationInventory,
        inventory_ref: ManifestOutput,
        chunking_job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        sealing_id: UUID,
    ) -> ManifestOutput:
        """Freeze a verified expected-batch plan under the chunking job's lease.

        The service first saves immutable chunk files, per-batch manifests,
        and the canonical inventory outside the transaction. Validate every
        batch's expected ID range and files, all configured bounds, and the
        generation-wide ID digest. inventory_ref must identify those exact
        bytes in the version-1 generation-inventory format. Do not infer
        readiness from the inventory's chunk count alone.

        Inside one transaction, require this tenant/document/generation's
        RUNNING CHUNK job, exact worker and unexpired lease, valid execution
        deadline, nondeleted document, and still-desired generation. Validate
        source version/hash, saved pipeline, configured dimensions and vector
        target, canonical inventory hash/size, and registered prerequisites.
        Raise LeaseLostError on lost authority or ValueError for invalid data.

        Save the pinned reference, immutable expected batch descriptors, and
        resumable scheduling intent together. Seal exactly one plan per
        generation. An equivalent already sealed plan returns its original
        reference without replacing it; incompatible changes require another
        generation. Keep the original file references even after a new attempt.

        Persist sealing_id and payload binding for uncertain-response recovery.
        Identical request replay resolves the committed seal without reviving
        the job, publishing content, or resetting scheduling progress. Invalid
        request-ID reuse raises ValueError. Surface unresolved database commits.

        This does not finish the chunking job or release embedding jobs. The
        worker must commit its checkpoint and complete_stage; only then may
        release_inventory_batches act on the durable scheduling intent.
        """
        ...

    async def get_generation_inventory_ref(
        self,
        *,
        tenant_id: UUID,
        generation_id: UUID,
    ) -> ManifestOutput | None:
        """Read the sealed inventory reference for recovery and publication.

        Return None when no plan is sealed in this tenant. Reading a reference
        does not authorize retrieval or prove that its files remain valid.
        The service must verify downloaded bytes and current generation state.
        """
        ...

    async def release_inventory_batches(
        self,
        *,
        tenant_id: UUID,
        generation_id: UUID,
        limit: int,
    ) -> tuple[OutboxRecord, ...]:
        """Create a bounded set of embedding jobs from unscheduled sealed batches.

        The scheduler uses a positive policy-bounded limit. Atomically require
        the saved sealed plan, completed chunking prerequisite, nondeleted
        document, still-desired generation, and valid generation execution
        deadline. Never schedule from a caller-supplied replacement inventory.

        Claim at most limit unscheduled descriptors. For each, save a queued
        EMBED job bound to its exact batch inputs, recipe, execution policy,
        and stable shard key, plus its initial outbox notification. Enforce
        uniqueness by tenant/generation/stage/shard and bind each notification
        to its durable job-creation identity. Update scheduling progress in the
        same commit so a crash cannot lose the remaining continuation.

        Concurrent schedulers and lost-response retries must not create more
        jobs for the same batch. Return newly committed outbox records only;
        the publisher sends them independently. Empty output is not proof all
        work is done; inspect durable expected/scheduled/completed batch state.
        Do not increment execution attempts or invent worker leases here.

        The fan-in barrier must compare completed work against every sealed
        expected batch, including those not yet scheduled. A partly scheduled
        generation must never release indexing/publication as if complete.
        For obsolete or deleted generations, stop scheduling and leave durable
        cancellation/cleanup to the reconciler. Do not mutate the sealed plan,
        sleep, contact SQS, or hold a transaction open for vector operations.
        """
        ...

    async def aclose(self) -> None:
        """Release owned database resources safely; repeated calls must be safe."""
        ...


class RuntimeMetadataStore(MetadataStore, Protocol):
    """Add the service reads and recovery operations needed by the bundled runtime.

    Implement these with the same tenant scope and transactional guarantees as
    MetadataStore. Status reads do not grant lease or publication authority.
    Cleanup plan dictionaries are internal typed-codec values, not API input.
    """

    async def now(self, tenant_id: UUID) -> datetime:
        """Return authoritative database time for budget calculations."""
        ...

    async def tenants(self) -> tuple[UUID, ...]:
        """Enumerate service-owned tenants for background processing, without content."""
        ...

    async def get_source_version(
        self, *, tenant_id: UUID, source_version_id: UUID
    ) -> SourceVersion | None:
        """Read the exact original source registered for this tenant."""
        ...

    async def get_generation(
        self, *, tenant_id: UUID, generation_id: UUID
    ) -> ProcessingGeneration | None:
        """Read a build's source and recipe identity without claiming work."""
        ...

    async def get_generation_pipeline(
        self, *, tenant_id: UUID, generation_id: UUID
    ) -> PipelineConfig | None:
        """Restore the accepted recipe so changing defaults cannot change old work."""
        ...

    async def list_generation_jobs(
        self, *, tenant_id: UUID, generation_id: UUID, stage=None
    ) -> tuple[Job, ...]:
        """List bounded generation dependencies; the bundled limit is 4,100 jobs (MAX_GENERATION_JOBS)."""
        ...

    async def get_inventory(
        self, *, tenant_id: UUID, generation_id: UUID
    ) -> GenerationInventory | None:
        """Return the immutable sealed plan used for fan-out and completeness."""
        ...

    async def get_batch_plan(
        self, *, tenant_id: UUID, generation_id: UUID, batch_key: str
    ) -> BatchPlan | None:
        """Return the plan saved when this batch's embedding job was scheduled.

        Written in the same commit as the job, copied from the sealed
        inventory. None only when no such batch is scheduled.
        """
        ...

    async def abandoned_uploads(
        self, *, tenant_id: UUID, grace_seconds: int, limit: int = 100
    ) -> tuple[tuple[UploadSession, MultipartPlan | None], ...]:
        """Failed or expired uploads, never accepted, whose files are not yet
        erased and whose session expired more than grace_seconds ago."""
        ...

    async def mark_upload_files_removed(
        self, *, tenant_id: UUID, upload_id: UUID
    ) -> bool:
        """Record the erasure once; False when not abandoned or already done."""
        ...

    async def list_verifying_uploads(
        self, tenant_id: UUID, limit: int = 10
    ) -> tuple[UploadSession, ...]:
        """Find bounded, unexpired completed transfers still needing verification."""
        ...

    async def admit_generations(self, *, tenant_id: UUID, max_active: int) -> int:
        """Start waiting generations up to max_active; budgets begin here.

        claim_job must refuse jobs of generations not yet admitted, and
        reconcile must neither expire nor re-notify them while they wait.
        """
        ...

    async def prune_history(self, *, older_than_days: int, limit: int) -> int:
        """Delete expired replay/notification history in bounded batches."""
        ...

    # External recognition (see contracts.recognition). All writes that act
    # for a worker require its current lease; parking and polling must be
    # atomic with the pending count so a finished job is never left waiting.

    async def record_recognition_job(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        request_id: str,
        submission: int,
        provider_job_id: str,
        page_numbers: tuple[int, ...],
        poll_after_seconds: float,
        select_pages: bool = False,
    ) -> dict:
        """Durably record an accepted provider job; idempotent per submission."""
        ...

    async def list_recognition_jobs(self, *, tenant_id: UUID, job_id: UUID) -> tuple:
        """All recorded provider jobs for one stage job."""
        ...

    async def park_job_waiting(
        self,
        *,
        tenant_id: UUID,
        job_id: UUID,
        worker_id: str,
        lease_id: UUID,
        wait_id: UUID,
    ) -> Job:
        """WAITING if provider work is pending, otherwise re-queue immediately."""
        ...

    async def due_recognition_polls(self, *, tenant_id: UUID, limit: int) -> tuple:
        """Pending provider jobs whose poll time has come."""
        ...

    async def finish_recognition_poll(
        self,
        *,
        tenant_id: UUID,
        recognition_id: UUID,
        state: str,
        failure_code: str | None = None,
        failures: dict | None = None,
        retry_after_seconds: float = 0,
    ) -> bool:
        """Record a poll outcome and re-queue the waiting job when none pend."""
        ...

    async def release_waiting_jobs(self, *, tenant_id: UUID, limit: int = 100) -> int:
        """Abandon provider jobs of ended work; wake waiters with none pending."""
        ...

    async def acquire_rate(self, *, key: str, rate: float, burst: float) -> float:
        """Shared token bucket; seconds to wait, 0 when a call may proceed."""
        ...

    async def reserve_recognition_pages(
        self, *, tenant_id: UUID, request_id: str, pages: int, daily_limit: int
    ) -> float:
        """Reserve today's billed pages; seconds until it fits, 0 when reserved."""
        ...

    async def scheduling_generations(
        self, tenant_id: UUID, limit: int = 100
    ) -> tuple[UUID, ...]:
        """Return eligible sealed builds with unscheduled work; completed builds must not starve later builds."""
        ...

    async def reconcile(self, tenant_id: UUID, limit: int = 100) -> int:
        """Repair expired/obsolete jobs, missing notifications and expired intake in fair bounded passes."""
        ...

    async def reprocess_document(
        self,
        *,
        tenant_id: UUID,
        document_id: UUID,
        request_id: UUID,
        policy,
        max_generation_seconds: float,
        pipeline: PipelineConfig | None = None,
    ) -> ProcessingGeneration:
        """Idempotently authorize a new generation and fresh fixed budget from an accepted source."""
        ...

    async def job_report(self, tenant: UUID, job_id: UUID) -> dict | None:
        """Expose safe state/failure identifiers without credentials or lease tokens."""
        ...

    async def list_job_reports(self, tenant: UUID, generation_id: UUID) -> list[dict]:
        """Return job_report() shapes for one generation in one bounded read."""
        ...

    async def quarantine_delivery(self, delivery, reason: str) -> None:
        """Durably retain bounded malformed-message evidence before acknowledging it."""
        ...

    async def cleanup_candidates(
        self, tenant_id: UUID, limit: int = 10
    ) -> tuple[dict, ...]:
        """List due cleanup intent without yet authorizing remote erasure."""
        ...

    async def authorize_cleanup(self, tenant: UUID, cleanup_id: UUID) -> dict | None:
        """Freeze a reference-safe erasure plan under the same lock as content acceptance."""
        ...

    async def finish_cleanup_plan(
        self, tenant: UUID, cleanup_id: UUID, plan: dict
    ) -> None:
        """Finalize only the previously frozen plan after its remote erasure succeeds."""
        ...
