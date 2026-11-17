"""Example custom adapter for a hypothetical internal company embedding gateway.

This is NOT a wrapper around a built-in provider. The example gateway accepts
POST /encode with {model, texts, dimensions, max_tokens} and returns
{vectors: [[...], ...]} in the same order. Its documented tokenizer uses one
UTF-8 byte per token. Adapt these assumptions to your real provider contract.
"""

import asyncio
import json
import httpx
from rag_ingest.contracts.embedding import EmbeddingBatchResult
from rag_ingest.contracts.errors import (
    RetryableError,
    RateLimitedError,
    PermanentInputError,
    OperatorActionRequiredError,
    UncertainOutcomeError,
)
from rag_ingest.domain.embeddings import EmbeddedChunk


class CompanyEmbedder:
    """Translate a private gateway into the package's embedding contract."""

    def __init__(self, *, base_url, token, transport=None):
        """Keep credentials out of the pipeline fingerprint and saved artifacts."""
        if not token:
            raise ValueError("Company gateway token required")
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + token},
            transport=transport,
            limits=httpx.Limits(max_connections=8),
            timeout=httpx.Timeout(20, connect=5),
        )

    async def embed_batch(self, chunks, *, pipeline, dimensions, limits):
        """Enforce bounds, call the gateway once, and validate every returned vector.

        The core persists successful output and decides whether a classified
        failure is eligible for a later attempt. Never mark an unknown billed
        outcome as safe to repeat merely because an HTTP timeout occurred.
        """
        spec = pipeline.embedder
        options = json.loads(spec.options_json)
        if spec.driver != "company_http" or spec.implementation_version != "1":
            raise ValueError("Wrong adapter recipe")
        if (
            set(options) != {"model", "dimensions", "encoding"}
            or options["encoding"] != "utf8-bytes"
        ):
            raise ValueError("Unsupported company recipe")
        if dimensions != options["dimensions"] or not 1 <= len(chunks) <= min(
            limits.max_items, 128
        ):
            raise ValueError("Invalid company batch")
        if (
            len({c.vector_id for c in chunks}) != len(chunks)
            or len({(c.tenant_id, c.generation_id) for c in chunks}) != 1
        ):
            raise ValueError("Mixed embedding scope")
        if any(len(c.text) > limits.max_input_characters for c in chunks):
            raise PermanentInputError("Company input too long")
        tokens = sum(len(c.text.encode("utf-8")) for c in chunks)
        if tokens > limits.max_batch_tokens:
            raise PermanentInputError("Company token budget exceeded")
        try:
            async with asyncio.timeout(limits.timeout_seconds):
                response = await self.client.post(
                    "encode",
                    json={
                        "model": options["model"],
                        "texts": [c.text for c in chunks],
                        "dimensions": dimensions,
                        "max_tokens": limits.max_batch_tokens,
                    },
                )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise RetryableError("Company gateway connection failed") from None
        except (httpx.TimeoutException, httpx.NetworkError, TimeoutError):
            raise UncertainOutcomeError("Company request may have completed") from None
        if response.status_code == 429:
            raise RateLimitedError("Company gateway throttled request")
        if response.status_code in (401, 403, 404):
            raise OperatorActionRequiredError("Company gateway access rejected")
        if response.status_code >= 500:
            raise UncertainOutcomeError("Company gateway outcome unknown")
        if response.status_code >= 400:
            raise PermanentInputError("Company gateway rejected input")
        try:
            vectors = response.json()["vectors"]
            if not isinstance(vectors, list) or len(vectors) != len(chunks):
                raise ValueError
            if any(
                not isinstance(v, list) or any(type(x) not in (int, float) for x in v)
                for v in vectors
            ):
                raise ValueError
            result = EmbeddingBatchResult(
                items=tuple(
                    EmbeddedChunk(
                        chunk=c,
                        pipeline_fingerprint=pipeline.fingerprint,
                        dimensions=dimensions,
                        vector=tuple(float(x) for x in v),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                )
            )
            result.validate_for(
                chunks, pipeline_fingerprint=pipeline.fingerprint, dimensions=dimensions
            )
        except (ValueError, TypeError, KeyError, OverflowError):
            raise UncertainOutcomeError("Invalid company gateway result") from None
        return result

    async def aclose(self):
        """Release the HTTP pool when the owning ingestion engine shuts down."""
        await self.client.aclose()
