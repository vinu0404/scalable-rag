"""Live-test Reranker: an OpenAI chat model scores each passage for the query.

Application code for the live test, not part of the package (like the
Textract adapter). A dedicated cross-encoder (examples/http_reranker.py) is
faster and cheaper per call; this one needs only the OpenAI key the live
test already uses. Passages are labelled 1..n so long chunk IDs never reach
the prompt; the reply must give exactly one 0..10 score per label.
"""

import json
import os

import httpx

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    RateLimitedError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.contracts.rerank import RerankerCapabilities, RerankScore
from rag_ingest.domain.pipeline import ComponentSpec

INSTRUCTIONS = (
    "You rate how well each numbered passage answers the question. "
    "Score 0 (irrelevant) to 10 (directly and fully answers it). "
    'Reply with JSON only: {"scores": {"<number>": <score>, ...}} covering every number.'
)


class OpenAIReranker:
    def __init__(self, *, api_key, model=None, base_url="https://api.openai.com/v1", transport=None):
        self.model = model or os.environ.get("DEMO_RERANK_MODEL", "gpt-4o-mini")
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": "Bearer " + api_key},
            timeout=httpx.Timeout(30, connect=5),
            transport=transport,
        )
        self._capabilities = RerankerCapabilities(
            max_candidates=20, max_query_characters=1000, max_passage_characters=1200
        )

    @property
    def spec(self):
        return ComponentSpec(
            driver="openai-chat-rerank",
            implementation_version="1",
            options_json=json.dumps({"model": self.model}),
        )

    @property
    def capabilities(self):
        return self._capabilities

    async def rerank(self, query, candidates):
        passages = "\n\n".join(f"[{n}] {c.text}" for n, c in enumerate(candidates, start=1))
        try:
            r = await self.client.post(
                "chat/completions",
                json={
                    "model": self.model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": INSTRUCTIONS},
                        {"role": "user", "content": f"Question: {query}\n\nPassages:\n\n{passages}"},
                    ],
                },
            )
        except httpx.TimeoutException:
            raise UncertainOutcomeError("Reranker response lost") from None
        except httpx.HTTPError:
            raise RetryableError("Reranker unreachable") from None
        if r.status_code == 429:
            raise RateLimitedError("Reranker throttled")
        if r.status_code in (401, 403, 404):
            raise OperatorActionRequiredError("Reranker model or key refused")
        if r.status_code >= 400:
            raise RetryableError(f"Reranker failed ({r.status_code})")
        try:
            scores = json.loads(r.json()["choices"][0]["message"]["content"])["scores"]
            return tuple(
                RerankScore(id=c.id, score=float(scores[str(n)]))
                for n, c in enumerate(candidates, start=1)
            )
        except (KeyError, ValueError, TypeError, IndexError):
            raise RetryableError("Reranker reply was not one score per passage") from None

    async def aclose(self):
        await self.client.aclose()
