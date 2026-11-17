"""Reranking as a replaceable adapter: any cross-encoder, rerank API or model.

A reranker reads the query together with each candidate passage and scores
how well the passage answers it. That is more accurate than comparing
separately computed embeddings or matching words, and slower, so it only
reorders the top candidates the retrievers found. It is optional: none is
bundled, and search works without one.

The pipeline gives it plain text and IDs, applies a time limit, and on any
failure keeps the fused order (reported as degraded) unless strict.
"""

from dataclasses import dataclass
from typing import Protocol

from rag_ingest.domain.pipeline import ComponentSpec


@dataclass(frozen=True, kw_only=True)
class RerankerCapabilities:
    """Limits the pipeline respects before calling the reranker.

    max_candidates: most passages per call. max_query_characters and
    max_passage_characters: longer text is cut (at a character boundary) by
    the pipeline, so the reranker never receives more than it accepts.
    """

    max_candidates: int = 100
    max_query_characters: int = 2000
    max_passage_characters: int = 4000

    def __post_init__(self) -> None:
        for name, low, high in (
            ("max_candidates", 1, 1000),
            ("max_query_characters", 1, 100_000),
            ("max_passage_characters", 1, 100_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be {low}..{high}")


@dataclass(frozen=True, kw_only=True)
class RerankCandidate:
    """One passage to score; id is opaque to the reranker."""

    id: str
    text: str


@dataclass(frozen=True, kw_only=True)
class RerankScore:
    """The reranker's relevance score for one candidate (higher is better)."""

    id: str
    score: float


class Reranker(Protocol):
    """Operations every reranker adapter implements.

    Rules:
    - Return exactly one finite score per candidate, no other IDs. Order of
      the returned tuple does not matter; the pipeline sorts by score.
    - Scores only need to order this call's candidates; they need not be
      comparable across calls or with retriever scores.
    - Bound the call (the pipeline also enforces its own time limit) and map
      provider failures to contracts.errors categories.
    - Never log or keep query or passage text beyond the call.
    """

    @property
    def spec(self) -> ComponentSpec:
        """Identity of the model and version, for logs and debugging."""
        ...

    @property
    def capabilities(self) -> RerankerCapabilities: ...

    async def rerank(
        self, query: str, candidates: tuple[RerankCandidate, ...]
    ) -> tuple[RerankScore, ...]: ...

    async def aclose(self) -> None: ...
