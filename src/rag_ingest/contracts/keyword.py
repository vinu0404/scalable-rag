"""Keyword (lexical) search as a replaceable adapter: BM25, full-text, or any engine.

A keyword index is optional and detachable. When configured, the index stage
writes every chunk to it, publication verifies it, cleanup deletes from it,
and hybrid search combines its ranking with the vector ranking.

Like a vector store it returns IDs and scores only. Text, file names and pages
always come from the chunk table after the document is checked to be live for
the tenant, so an engine with a bug or stale data cannot leak another
tenant's, a deleted document's or a superseded build's text.
"""

from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from rag_ingest.contracts.vectors import VectorCheck, VectorScope

SCORING = ("bm25", "tfidf", "other")


@dataclass(frozen=True, kw_only=True)
class KeywordCapabilities:
    """What an engine does, declared once.

    scoring: how it ranks (informational; fusion uses ranks, not scores).
    max_batch_items: largest index_batch it accepts.
    filters: filters it can apply itself during search ("document_ids");
        others are applied afterwards by the pipeline, which fetches more.
    eventually_visible: new entries become searchable only after a refresh
        (OpenSearch style); verification then reports NOT_QUERYABLE until
        they are, and publication waits.
    """

    scoring: str
    max_batch_items: int = 512
    filters: frozenset = field(default_factory=frozenset)
    eventually_visible: bool = False

    def __post_init__(self) -> None:
        if self.scoring not in SCORING:
            raise ValueError(f"scoring must be one of {SCORING}")
        if type(self.max_batch_items) is not int or not 1 <= self.max_batch_items <= 10_000:
            raise ValueError("max_batch_items must be 1..10000")
        if not isinstance(self.filters, frozenset) or self.filters - {"document_ids"}:
            raise ValueError("filters may only contain 'document_ids'")


@dataclass(frozen=True, kw_only=True)
class KeywordDocument:
    """One chunk to index: its stable ID, scope and exact text."""

    vector_id: str
    document_id: UUID
    generation_id: UUID
    text: str
    text_sha256: str


@dataclass(frozen=True, kw_only=True)
class KeywordExpectation:
    """What verification expects for one chunk: present, with exactly this text."""

    vector_id: str
    text_sha256: str


@dataclass(frozen=True, kw_only=True)
class KeywordHit:
    """One keyword match: identity and the engine's score only."""

    vector_id: str
    document_id: UUID
    generation_id: UUID
    score: float


class KeywordIndex(Protocol):
    """Operations every keyword engine adapter implements.

    Rules:
    - Tenant isolation: every call carries the tenant; never return or touch
      another tenant's entries (namespace, routing or a filter).
    - Stable IDs: indexing the same chunk again is a no-op; different text
      under an existing ID is refused (PermanentInputError), never replaced.
    - Query text is plain user text: never pass it to an engine's query
      syntax (no query_string, no to_tsquery on raw input). A query with no
      searchable terms returns no hits rather than an error.
    - Map provider failures to contracts.errors categories; bound every call.
    """

    @property
    def target_id(self) -> str:
        """Identity of the index and its analysis (engine, index, language).

        Changing the analyzer or language must change it: entries indexed
        under another target are not searched until reindexed.
        """
        ...

    @property
    def capabilities(self) -> KeywordCapabilities: ...

    async def index_batch(
        self, scope: VectorScope, *, items: tuple[KeywordDocument, ...]
    ) -> None:
        """Index every item (idempotent); raise if any could not be stored."""
        ...

    async def verify_batch(
        self, scope: VectorScope, *, expected: tuple[KeywordExpectation, ...]
    ) -> tuple[VectorCheck, ...]:
        """One check per expected ID: READY, MISSING, MISMATCH or NOT_QUERYABLE."""
        ...

    async def search(
        self,
        *,
        tenant_id: UUID,
        query: str,
        k: int,
        document_ids: tuple[UUID, ...] = (),
    ) -> tuple[KeywordHit, ...]:
        """Up to k best matches of this tenant, best first."""
        ...

    async def delete_generation(self, scope: VectorScope) -> None:
        """Remove exactly this build's entries; repeating is safe."""
        ...

    async def aclose(self) -> None: ...
