"""Define generation-scoped vector writes, verification, and cleanup.

An accepted write is not necessarily searchable yet. The publication service
must verify the exact expected set before activating a generation in metadata.
"""

from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Protocol
from uuid import UUID

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.domain.embeddings import EmbeddedChunk


@dataclass(frozen=True, kw_only=True)
class VectorScope:
    """Identify exactly one tenant, document, and generation for an operation.

    Keep all three when reading, writing, or deleting. A generation ID alone
    must not let a caller access another tenant's data.
    """

    tenant_id: UUID
    document_id: UUID
    generation_id: UUID

    def __post_init__(self) -> None:
        """Require UUID identifiers without claiming the records are authorized."""
        for name in ("tenant_id", "document_id", "generation_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")

    def validate_item(self, item: EmbeddedChunk) -> None:
        """Reject an embedding that does not belong to this exact scope."""
        if not isinstance(item, EmbeddedChunk):
            raise ValueError("item must be an EmbeddedChunk")
        chunk = item.chunk
        if (chunk.tenant_id, chunk.document_id, chunk.generation_id) != (
            self.tenant_id,
            self.document_id,
            self.generation_id,
        ):
            raise ValueError("Embedding does not belong to the vector scope")

    def validate_item_ids(self, vector_id: str, document_id: UUID, generation_id: UUID) -> None:
        """Reject an entry outside this scope (the ID embeds tenant, document, build)."""
        prefix = f"{self.tenant_id}:{self.document_id}:{self.generation_id}:"
        if (document_id, generation_id) != (self.document_id, self.generation_id) or not (
            isinstance(vector_id, str) and vector_id.startswith(prefix)
        ):
            raise ValueError("Entry does not belong to the scope")


@dataclass(frozen=True, kw_only=True)
class VectorWriteResult:
    """Report one accepted write or classified failure for a logical vector ID.

    error=None means write acceptance was confirmed, not that publication is
    allowed. A lost response must use UncertainOutcomeError, not false success.
    """

    vector_id: str
    error: (
        RetryableError
        | PermanentInputError
        | OperatorActionRequiredError
        | UncertainOutcomeError
        | None
    ) = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Check identity and error category; do not inspect a remote database."""
        if not isinstance(self.vector_id, str) or not self.vector_id.strip():
            raise ValueError("vector_id must not be blank")
        if self.error is not None and not isinstance(
            self.error,
            (
                RetryableError,
                PermanentInputError,
                OperatorActionRequiredError,
                UncertainOutcomeError,
            ),
        ):
            raise ValueError("error must be a classified vector-write failure")


@unique
class VectorReadiness(StrEnum):
    """Distinguish matching searchable output from incomplete indexing states."""

    READY = "ready"
    MISSING = "missing"
    MISMATCH = "mismatch"
    NOT_QUERYABLE = "not_queryable"


@dataclass(frozen=True, kw_only=True)
class VectorCheck:
    """Report verification of one expected vector in the intended search target.

    READY requires matching content/metadata and demonstrated index visibility
    under the adapter's supported search semantics. Fetch-by-ID alone is not
    automatically proof of visibility in similarity search.
    """

    vector_id: str
    readiness: VectorReadiness

    def __post_init__(self) -> None:
        """Require a logical vector ID and a recognized readiness state."""
        if not isinstance(self.vector_id, str) or not self.vector_id.strip():
            raise ValueError("vector_id must not be blank")
        if not isinstance(self.readiness, VectorReadiness):
            raise ValueError("readiness must be a VectorReadiness")


class VectorStore(Protocol):
    """Specify replaceable indexing with stable logical identities.

    Bind an adapter instance to a configured search target with a stable
    identity stored in the generation plan. Validate dimensions, metric, and
    supported filtering/verification capabilities at startup. Provisioning or
    changing an index must be explicit, not a hidden side effect of a write.
    Bound batches and network deadlines; keep SDK calls off the event loop.
    """

    @property
    def target_id(self) -> str:
        """Identify the logical database/index/namespace used for this build.

        This is not a credential or a transient connection URL. Changing the
        index or incompatible settings must change its registered identity.
        """
        ...

    async def upsert_batch(
        self,
        scope: VectorScope,
        *,
        items: tuple[EmbeddedChunk, ...],
    ) -> tuple[VectorWriteResult, ...]:
        """Write a bounded batch to stable IDs and report every item exactly once.

        Require unique IDs, matching scope, recipe, and dimensions. Preserve
        source attribution, text hash, pipeline identity, and original logical
        ID in stored metadata. If the provider needs a different ID format,
        use a deterministic collision-safe mapping consistently across retries.

        Repeating the same payload must not append duplicates. Do not replace
        a different logical payload under an existing immutable generation ID;
        the adapter must enforce compatibility under concurrent writes or
        reject the unsupported configuration. Compare stored vectors using
        the declared precision/serialization policy, not an unstated tolerance.

        Return accepted and failed/uncertain outcomes for all requested IDs,
        with no missing, duplicate, or unexpected outcomes. A batch-level error
        is allowed if no individual outcome is known. Never label a timed-out
        write definitely absent. Writing does not activate the generation.
        """
        ...

    async def verify_batch(
        self,
        scope: VectorScope,
        *,
        expected: tuple[EmbeddedChunk, ...],
    ) -> tuple[VectorCheck, ...]:
        """Inspect a bounded expected set and report each ID exactly once.

        Check exact scope, source/text/recipe metadata, dimensions, and stored
        vector payload according to declared precision policy. Distinguish
        missing, incompatible, and not-yet-queryable output. Classify provider
        failures separately; access denial is not MISSING. Do not claim READY
        when the provider cannot establish the required query visibility.

        The service validates result coverage and aggregates all manifest
        batches before publication. Counts alone do not prove completeness.
        This read does not activate the generation or guarantee future service
        availability. Mutations and cleanup must respect the sealed build.
        """
        ...

    async def delete_generation(self, scope: VectorScope) -> None:
        """Remove only this exact generation and confirm the intended deletion.

        The service must authorize cleanup and check retention/current state
        before calling. Handle repeated deletion safely. Do not broaden scope
        to a tenant or collection. If deletion is asynchronous, confirm its
        completion within bounded deadlines or surface a retryable/uncertain
        outcome; acceptance alone is not proof of physical removal. Late worker
        writes require separate quiescence and reconciliation safeguards.
        """
        ...

    async def aclose(self) -> None:
        """Release owned client resources; repeated close calls must be safe."""
        ...


class DigestVerifiableVectorStore(VectorStore, Protocol):
    """Optional: verify a whole generation without re-reading its vectors.

    Adapters that implement both methods let publication compare one digest
    per batch, computed by the database, instead of downloading every
    embedding again and checking it row by row. Adapters without them keep
    the verify_batch path; nothing else changes.

    The digest must cover everything verify_batch compares (stored IDs,
    payload and exact vector), so equal digests mean every row would be
    READY. Any difference is never trusted either way: the service then
    runs verify_batch on that batch to classify missing or mismatched rows.
    """

    def expected_digest(
        self, scope: VectorScope, items: tuple[EmbeddedChunk, ...]
    ) -> str:
        """Digest the sealed build's batch exactly as stored_digests would."""
        ...

    async def stored_digests(
        self, scope: VectorScope, *, batches: tuple[tuple[str, ...], ...]
    ) -> tuple[str | None, ...]:
        """Digest each group of stored IDs in one bounded read (None: no rows)."""
        ...


@dataclass(frozen=True, kw_only=True)
class VectorHit:
    """One candidate from a vector search: identity and similarity only.

    Text, spans and file names are never taken from the vector store; the
    search pipeline reads them from the chunk table after checking that the
    document is live for the tenant.
    """

    vector_id: str
    document_id: UUID
    generation_id: UUID
    score: float


class SearchableVectorStore(VectorStore, Protocol):
    """A vector store the bundled search pipeline can query."""

    async def search_hits(
        self,
        *,
        tenant_id: UUID,
        vector: tuple[float, ...],
        k: int,
        document_ids: tuple[UUID, ...] = (),
    ) -> tuple[VectorHit, ...]:
        """The k most similar vectors of this tenant, best first.

        With document_ids, only those documents. The result may include
        superseded builds; the pipeline drops them. Never return another
        tenant's vectors.
        """
        ...
