"""A reproducible, offline Embedder for tests; it never calls a network.

Vectors are derived from each chunk's text (same text, same vector), so
tests run without an API key, cost nothing and never flake. They carry no
meaning: never use this for real search. Inject it explicitly:

    settings = ServiceSettings(..., embedding_driver="deterministic")
    engine = build_engine(settings, embedder=DeterministicEmbedder())
"""

import hashlib
import math

from rag_ingest.contracts.embedding import EmbeddingBatchResult
from rag_ingest.domain.embeddings import EmbeddedChunk


class DeterministicEmbedder:
    """Produce reproducible test vectors; never use this as a semantic model."""

    async def embed_batch(self, chunks, *, pipeline, dimensions, limits):
        """Derive nonzero normalized vectors from text for offline integration tests."""
        if pipeline.embedder.driver != "deterministic":
            raise ValueError("Test embedder must be explicitly selected")
        if not 1 <= len(chunks) <= limits.max_items:
            raise ValueError("Invalid test embedding batch")
        results = []
        for chunk in chunks:
            seed = hashlib.sha256(chunk.text.encode()).digest()
            values = [(seed[i % 32] - 127.5) / 127.5 for i in range(dimensions)]
            length = math.sqrt(sum(v * v for v in values))
            results.append(
                EmbeddedChunk(
                    chunk=chunk,
                    pipeline_fingerprint=pipeline.fingerprint,
                    dimensions=dimensions,
                    vector=tuple(v / length for v in values),
                )
            )
        return EmbeddingBatchResult(items=tuple(results))

    async def aclose(self):
        """No remote resources are used by this test implementation."""
