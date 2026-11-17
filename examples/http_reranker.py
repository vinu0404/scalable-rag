"""Example Reranker for HTTP cross-encoder services.

Two common request shapes:
- api="cohere": Cohere-style  POST {base}/v2/rerank  {model, query, documents}
  -> {"results": [{"index", "relevance_score"}]}  (Cohere, and compatible gateways)
- api="tei": Hugging Face Text Embeddings Inference  POST {base}/rerank
  {query, texts} -> [{"index", "score"}]  (self-hosted bge-reranker, etc.)

Not part of the package and not run against a live service here; its request
handling is checked in tests/test_rerank.py, and a real service can be
checked with rag_ingest.testing.rerank_conformance. Wire it in with:

    reranker = HttpReranker("https://api.cohere.com", api="cohere",
                            model="rerank-v3.5", headers={"Authorization": "Bearer ..."})
    engine = build_engine(settings, reranker=reranker)   # and INGEST_SEARCH_RERANK=true
"""

import json

import httpx

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RateLimitedError,
    RetryableError,
)
from rag_ingest.contracts.rerank import RerankerCapabilities, RerankScore
from rag_ingest.domain.pipeline import ComponentSpec


class HttpReranker:
    def __init__(self, base_url, *, api="tei", model=None, headers=None, transport=None,
                 max_candidates=100, max_passage_characters=4000):
        if api not in ("cohere", "tei"):
            raise ValueError("api must be 'cohere' or 'tei'")
        self.api = api
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers or {},
            timeout=httpx.Timeout(10, connect=5),
            transport=transport,
        )
        self._capabilities = RerankerCapabilities(
            max_candidates=max_candidates, max_passage_characters=max_passage_characters
        )

    @property
    def spec(self):
        return ComponentSpec(
            driver=f"http-{self.api}",
            implementation_version="1",
            options_json=json.dumps({"model": self.model or "server-default"}),
        )

    @property
    def capabilities(self):
        return self._capabilities

    async def rerank(self, query, candidates):
        texts = [c.text for c in candidates]
        if self.api == "cohere":
            path, body = "v2/rerank", {"model": self.model, "query": query,
                                       "documents": texts, "top_n": len(texts)}
        else:
            path, body = "rerank", {"query": query, "texts": texts, "raw_scores": False}
        try:
            r = await self.client.post(path, json=body)
        except httpx.HTTPError:
            raise RetryableError("Reranker unreachable") from None
        if r.status_code == 429:
            raise RateLimitedError("Reranker throttled the request")
        if r.status_code in (401, 403):
            raise OperatorActionRequiredError("Reranker refused the credentials")
        if r.status_code >= 500:
            raise RetryableError("Reranker failed")
        if r.status_code >= 400:
            raise PermanentInputError("Reranker rejected the request")
        rows = r.json()["results"] if self.api == "cohere" else r.json()
        key = "relevance_score" if self.api == "cohere" else "score"
        return tuple(
            RerankScore(id=candidates[row["index"]].id, score=float(row[key]))
            for row in rows
            if 0 <= row["index"] < len(candidates)
        )

    async def aclose(self):
        await self.client.aclose()
