"""Define bounded embedding batches with explicit results for every chunk.

The service persists known successes before retrying unresolved work. Model
calls and vector-database writes remain separate operations.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.domain.chunks import Chunk
from rag_ingest.domain.embeddings import EmbeddedChunk
from rag_ingest.domain.pipeline import PipelineConfig


@dataclass(frozen=True, kw_only=True)
class EmbeddingLimits:
    """Limit each request's item count, text, token budget, and elapsed time.

    The adapter must enforce provider limits too, including tokens added by
    configured prompts. Token counting must use the appropriate tokenizer.
    Runtime concurrency, total cost, and maximum dimensions are bounded by
    service policy separately. Do not truncate input silently to fit limits.
    """

    max_items: int
    max_input_characters: int
    max_batch_tokens: int
    timeout_seconds: float

    def __post_init__(self) -> None:
        """Require positive limits and a finite positive request timeout."""
        for name in ("max_items", "max_input_characters", "max_batch_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        try:
            valid = (
                type(self.timeout_seconds) in (int, float)
                and math.isfinite(self.timeout_seconds)
                and self.timeout_seconds > 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("timeout_seconds must be finite and positive")


@dataclass(frozen=True, kw_only=True)
class EmbeddingFailure:
    """Report a known failure or uncertain outcome for one requested chunk.

    Use shared categories so the core can distinguish safe retries, permanent
    input errors, operational problems, and requests that may already be billed.
    The error message must be safe; repr=False does not redact direct logging.
    """

    chunk: Chunk
    error: (
        RetryableError
        | PermanentInputError
        | OperatorActionRequiredError
        | UncertainOutcomeError
    ) = field(repr=False)

    def __post_init__(self) -> None:
        """Require the chunk and an allowed processing failure category."""
        if not isinstance(self.chunk, Chunk):
            raise ValueError("chunk must be a Chunk")
        if not isinstance(
            self.error,
            (
                RetryableError,
                PermanentInputError,
                OperatorActionRequiredError,
                UncertainOutcomeError,
            ),
        ):
            raise ValueError("error must be a classified embedding failure")


@dataclass(frozen=True, kw_only=True)
class EmbeddingBatchResult:
    """Return one success or failure per input, even for a partly successful batch."""

    items: tuple[EmbeddedChunk | EmbeddingFailure, ...]

    def __post_init__(self) -> None:
        """Require immutable, distinct chunk outcomes rather than a partial list."""
        if not isinstance(self.items, tuple) or any(
            not isinstance(item, (EmbeddedChunk, EmbeddingFailure))
            for item in self.items
        ):
            raise ValueError("items must be a tuple of embedding outcomes")
        ids = [item.chunk.vector_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "A chunk cannot have multiple outcomes in one batch result"
            )

    def validate_for(
        self,
        chunks: tuple[Chunk, ...],
        *,
        pipeline_fingerprint: str,
        dimensions: int,
    ) -> None:
        """Check exact input coverage and reject mismatched successful results.

        Call before saving results. Order may differ, but each requested chunk
        must appear exactly once with unchanged text, identity, and provenance.
        A missing response is a failure/uncertainty, never an implicit success.
        """
        if not isinstance(chunks, tuple) or any(
            not isinstance(c, Chunk) for c in chunks
        ):
            raise ValueError("chunks must be a tuple of Chunk records")
        expected = {chunk.vector_id: chunk for chunk in chunks}
        if len(expected) != len(chunks):
            raise ValueError("Requested chunks must have distinct identities")
        if {item.chunk.vector_id for item in self.items} != set(expected):
            raise ValueError("Batch outcomes must cover exactly the requested chunks")
        for item in self.items:
            if item.chunk != expected[item.chunk.vector_id]:
                raise ValueError("Batch outcome does not match its requested chunk")
            if isinstance(item, EmbeddedChunk):
                if (
                    item.pipeline_fingerprint != pipeline_fingerprint
                    or item.dimensions != dimensions
                ):
                    raise ValueError(
                        "Embedding output does not match the expected recipe or dimensions"
                    )


class Embedder(Protocol):
    """Specify replaceable embedding without choosing a model provider."""

    async def embed_batch(
        self,
        chunks: tuple[Chunk, ...],
        *,
        pipeline: PipelineConfig,
        dimensions: int,
        limits: EmbeddingLimits,
    ) -> EmbeddingBatchResult:
        """Embed a bounded nonempty batch and preserve known individual outcomes.

        Validate unique chunk identities, one tenant/generation per batch,
        matching saved recipe, expected dimensions, model revision, and limits.
        Account for every output-affecting prompt/transform in the saved recipe.
        Return validated successes and classified failures for all inputs.
        Correlate provider result indices to chunks; never assume responses
        arrive in input order or pair vectors by an unchecked list position.

        If no individual outcomes are known, a classified batch-level error may
        be raised. If some successes are known, return them with explicit
        failures/uncertainties for the rest so the service can save progress.
        Unknown programming errors must remain visible. Do not retry forever
        inside the adapter or multiply SDK and core retry budgets.

        A lost response may mean the model ran and billed the request. Report
        uncertainty honestly; safe vector IDs do not prevent duplicate model
        charges. Keep network calls off the event loop and never log chunk text.
        This method does not save checkpoints, publish data, or call SQS.
        """
        ...

    async def aclose(self) -> None:
        """Release owned model-client resources; repeated calls must be safe."""
        ...
