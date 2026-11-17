"""OpenAI-compatible embeddings (the built-in embedder)."""

import asyncio
import json
import httpx
from rag_ingest.domain.limits import MAX_BATCH_ITEMS
from rag_ingest.contracts.embedding import EmbeddingBatchResult, EmbeddingFailure
from rag_ingest.domain.embeddings import EmbeddedChunk
from rag_ingest.contracts.errors import (
    RetryableError,
    RateLimitedError,
    UncertainOutcomeError,
    OperatorActionRequiredError,
    PermanentInputError,
)
from rag_ingest.retry import run_with_retries, CallRetryPolicy


def attempt_seconds(timeout_seconds):
    """Time one HTTP attempt may take: two thirds of the call's whole budget.

    The rest leaves room for one quick retry of a dropped connection. With the
    default 30 s budget an attempt gets 20 s, as before this was configurable.
    """
    return timeout_seconds * 2 / 3


class OpenAIEmbedder:
    """Call a configured model endpoint with explicit limits and no hidden HTTP retries."""

    def __init__(
        self, *, api_key, base_url="https://api.openai.com/v1", transport=None
    ):
        """Receive credentials at construction; never save them in pipeline recipes."""
        if not api_key:
            raise OperatorActionRequiredError("Embedding API key is required")
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + api_key},
            transport=transport,
            timeout=httpx.Timeout(20, connect=5),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def embed_batch(self, chunks, *, pipeline, dimensions, limits):
        """Validate token budgets and map provider indices to exact requested chunks."""
        import tiktoken

        spec = pipeline.embedder
        options = json.loads(spec.options_json)
        if (
            spec.driver != "openai"
            or spec.implementation_version != "1"
            or set(options) != {"model", "dimensions", "encoding"}
        ):
            raise ValueError("Unsupported embedding recipe")
        if dimensions != options["dimensions"] or not 1 <= len(chunks) <= min(
            limits.max_items, MAX_BATCH_ITEMS
        ):
            raise ValueError("Invalid embedding dimensions or batch size")
        if (
            len({c.vector_id for c in chunks}) != len(chunks)
            or len({(c.tenant_id, c.generation_id) for c in chunks}) != 1
        ):
            raise ValueError("Mixed or duplicate embedding scope")
        encoding = tiktoken.get_encoding(options["encoding"])
        tokens = 0
        for chunk in chunks:
            if len(chunk.text) > limits.max_input_characters:
                raise PermanentInputError("Chunk exceeds embedding input policy")
            count = len(encoding.encode(chunk.text, disallowed_special=()))
            if count > 8191:
                raise PermanentInputError("Chunk exceeds built-in model token ceiling")
            tokens += count
        if tokens > limits.max_batch_tokens:
            raise PermanentInputError("Embedding request exceeds token budget")

        per_attempt = attempt_seconds(limits.timeout_seconds)

        async def request():
            """Classify known HTTP outcomes before the short retry helper sees them."""
            try:
                r = await self.client.post(
                    "embeddings",
                    json={
                        "model": options["model"],
                        "input": [c.text for c in chunks],
                        "dimensions": dimensions,
                        "encoding_format": "float",
                    },
                    timeout=httpx.Timeout(per_attempt, connect=5),
                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                raise RetryableError("Embedding connection failed") from None
            except (httpx.TimeoutException, httpx.NetworkError):
                raise UncertainOutcomeError(
                    "Embedding response lost; billing may have occurred"
                ) from None
            if r.status_code == 429:
                try:
                    delay = max(0, float(r.headers.get("retry-after", "0")))
                except ValueError:
                    delay = None
                raise RateLimitedError(
                    "Embedding provider throttled the request",
                    retry_after_seconds=delay,
                )
            if r.status_code in (401, 403, 404):
                raise OperatorActionRequiredError(
                    "Embedding endpoint or credentials rejected"
                )
            if r.status_code >= 500:
                raise UncertainOutcomeError(
                    "Embedding provider failed after accepting the request"
                )
            if r.status_code >= 400:
                raise PermanentInputError(
                    "Embedding provider rejected request parameters or input"
                )
            try:
                return r.json()
            except ValueError:
                raise UncertainOutcomeError(
                    "Embedding response was not valid JSON"
                ) from None

        try:
            result = await run_with_retries(
                request,
                deadline=asyncio.get_running_loop().time() + limits.timeout_seconds,
                policy=CallRetryPolicy(
                    max_elapsed_seconds=limits.timeout_seconds,
                    attempt_timeout_seconds=per_attempt,
                ),
            )
        except TimeoutError:
            raise UncertainOutcomeError(
                "Embedding call deadline elapsed after possible dispatch"
            ) from None
        data = result.get("data", []) if isinstance(result, dict) else []
        if not isinstance(data, list):
            raise UncertainOutcomeError("Embedding response data must be a list")
        by_index = {}
        duplicates = set()
        for row in data:
            if not isinstance(row, dict) or type(row.get("index")) is not int:
                continue
            i = row["index"]
            if not 0 <= i < len(chunks):
                raise UncertainOutcomeError(
                    "Embedding response contains a foreign input index"
                )
            if i in by_index:
                duplicates.add(i)
            by_index[i] = row
        outcomes = []
        for i, chunk in enumerate(chunks):
            try:
                if i in duplicates:
                    raise ValueError
                values = by_index[i]["embedding"]
                if not isinstance(values, list) or any(
                    type(v) not in (int, float) for v in values
                ):
                    raise ValueError
                item = EmbeddedChunk(
                    chunk=chunk,
                    pipeline_fingerprint=pipeline.fingerprint,
                    dimensions=dimensions,
                    vector=tuple(float(v) for v in values),
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                item = EmbeddingFailure(
                    chunk=chunk,
                    error=UncertainOutcomeError("Missing or invalid embedding result"),
                )
            outcomes.append(item)
        result = EmbeddingBatchResult(items=tuple(outcomes))
        result.validate_for(
            chunks, pipeline_fingerprint=pipeline.fingerprint, dimensions=dimensions
        )
        return result

    async def aclose(self):
        """Close HTTP connections explicitly during shutdown."""
        await self.client.aclose()
