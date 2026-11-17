"""Example KeywordIndex: OpenSearch (or Elasticsearch) BM25 over its REST API.

Not part of the package and not tested against a live cluster here; its
request shapes are checked in tests/test_keyword.py, and a real cluster can
be checked with rag_ingest.testing.keyword_conformance. Wire it in with:

    keyword = OpenSearchKeywordIndex("https://search.internal:9200", index="rag-chunks",
                                     headers={"Authorization": "Basic ..."})
    await keyword.create_index()          # once, explicitly
    engine = build_engine(settings, keyword=keyword)

Rules it follows (see contracts/keyword.py):
- every request is filtered and routed by tenant;
- writes are create-only, so different text under an existing ID is refused;
- user text only ever goes into a `match` query, never `query_string`;
- new entries are searchable after a refresh, so verification reports
  NOT_QUERYABLE until then and publication waits.
"""

import json

import httpx

from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RetryableError,
)
from rag_ingest.contracts.keyword import KeywordCapabilities, KeywordHit
from rag_ingest.contracts.vectors import VectorCheck, VectorReadiness


class OpenSearchKeywordIndex:
    def __init__(self, url, *, index="rag-chunks", analyzer="english", headers=None, transport=None):
        self.index = index
        self.analyzer = analyzer
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/") + "/",
            headers=headers or {},
            timeout=httpx.Timeout(10, connect=5),
            transport=transport,
        )
        self._capabilities = KeywordCapabilities(
            scoring="bm25",
            max_batch_items=500,
            filters=frozenset({"document_ids"}),
            eventually_visible=True,
        )

    @property
    def target_id(self):
        return f"opensearch:{self.index}:{self.analyzer}"

    @property
    def capabilities(self):
        return self._capabilities

    async def _call(self, method, path, **kwargs):
        try:
            r = await self.client.request(method, path, **kwargs)
        except httpx.HTTPError:
            raise RetryableError("Keyword engine unreachable") from None
        if r.status_code in (401, 403):
            raise OperatorActionRequiredError("Keyword engine refused the credentials")
        if r.status_code == 429 or r.status_code >= 500:
            raise RetryableError("Keyword engine is busy or failing")
        if r.status_code >= 400 and r.status_code != 404:
            raise PermanentInputError("Keyword engine rejected the request")
        return r

    async def create_index(self):
        """Create the index with its mapping (explicit setup, never implicit)."""
        await self._call(
            "PUT",
            self.index,
            json={
                "mappings": {
                    "properties": {
                        "tenant_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "generation_id": {"type": "keyword"},
                        "text_sha256": {"type": "keyword"},
                        "text": {"type": "text", "analyzer": self.analyzer},
                    }
                }
            },
        )

    async def index_batch(self, scope, *, items):
        tenant = str(scope.tenant_id)
        lines = []
        for item in items:
            scope.validate_item_ids(item.vector_id, item.document_id, item.generation_id)
            lines.append(json.dumps({"create": {"_index": self.index, "_id": item.vector_id, "routing": tenant}}))
            lines.append(
                json.dumps(
                    {
                        "tenant_id": tenant,
                        "document_id": str(item.document_id),
                        "generation_id": str(item.generation_id),
                        "text": item.text,
                        "text_sha256": item.text_sha256,
                    }
                )
            )
        r = await self._call(
            "POST",
            "_bulk",
            content="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        conflicts = []
        for item, result in zip(items, r.json().get("items", []), strict=True):
            status = result["create"]["status"]
            if status == 409:
                conflicts.append(item)
            elif status >= 300:
                raise RetryableError("Keyword engine did not store every entry")
        if conflicts:  # already there: fine only if it is the same text
            stored = await self._stored(tenant, [c.vector_id for c in conflicts])
            if any(stored.get(c.vector_id) != c.text_sha256 for c in conflicts):
                raise PermanentInputError("Keyword engine holds different text for an ID")

    async def _stored(self, tenant, ids):
        r = await self._call("POST", f"{self.index}/_mget", params={"routing": tenant}, json={"ids": ids})
        return {
            d["_id"]: d["_source"]["text_sha256"]
            for d in r.json().get("docs", [])
            if d.get("found") and d["_source"].get("tenant_id") == tenant
        }

    async def verify_batch(self, scope, *, expected):
        tenant = str(scope.tenant_id)
        ids = [e.vector_id for e in expected]
        stored = await self._stored(tenant, ids)
        r = await self._call(
            "POST",
            f"{self.index}/_search",
            params={"routing": tenant},
            json={
                "size": len(ids),
                "_source": False,
                "query": {"bool": {"filter": [{"ids": {"values": ids}}, {"term": {"tenant_id": tenant}}]}},
            },
        )
        searchable = {h["_id"] for h in r.json()["hits"]["hits"]}
        checks = []
        for e in expected:
            if e.vector_id not in stored:
                readiness = VectorReadiness.MISSING
            elif stored[e.vector_id] != e.text_sha256:
                readiness = VectorReadiness.MISMATCH
            elif e.vector_id not in searchable:
                readiness = VectorReadiness.NOT_QUERYABLE  # stored, not refreshed yet
            else:
                readiness = VectorReadiness.READY
            checks.append(VectorCheck(vector_id=e.vector_id, readiness=readiness))
        return tuple(checks)

    async def search(self, *, tenant_id, query, k, document_ids=()):
        tenant = str(tenant_id)
        filters = [{"term": {"tenant_id": tenant}}]
        if document_ids:
            filters.append({"terms": {"document_id": [str(d) for d in document_ids]}})
        r = await self._call(
            "POST",
            f"{self.index}/_search",
            params={"routing": tenant},
            json={
                "size": k,
                "_source": ["document_id", "generation_id", "tenant_id"],
                # match: the text is analyzed as words, never parsed as syntax.
                "query": {"bool": {"must": [{"match": {"text": {"query": query}}}], "filter": filters}},
            },
        )
        from uuid import UUID

        return tuple(
            KeywordHit(
                vector_id=h["_id"],
                document_id=UUID(h["_source"]["document_id"]),
                generation_id=UUID(h["_source"]["generation_id"]),
                score=float(h["_score"]),
            )
            for h in r.json()["hits"]["hits"]
            if h["_source"].get("tenant_id") == tenant
        )

    async def delete_generation(self, scope):
        tenant = str(scope.tenant_id)
        await self._call(
            "POST",
            f"{self.index}/_delete_by_query",
            params={"routing": tenant, "refresh": "true", "conflicts": "proceed"},
            json={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": tenant}},
                            {"term": {"generation_id": str(scope.generation_id)}},
                        ]
                    }
                }
            },
        )

    async def aclose(self):
        await self.client.aclose()
