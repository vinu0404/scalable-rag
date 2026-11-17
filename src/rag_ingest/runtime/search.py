"""The search pipeline: query -> candidates -> live, filtered, fused results.

Steps:
1. Embed the query.
2. With filters, find matching live documents first. When few match, search
   only their vectors (exact, nothing missed); otherwise search the tenant and
   filter afterwards.
3. Fetch extra candidates, keep hits whose document is live, visible and
   passes the filters, reading their text from the chunk table, and fetch
   more when too many were dropped.
4. Collapse identical passages into one result listing every place it occurs.

Vector stores only return IDs and scores, so any store can serve this, and
none can return text of a deleted, superseded or foreign document.
"""

import asyncio
import hashlib
import logging
import math
from dataclasses import dataclass, replace
from uuid import uuid4

from rag_ingest.contracts.embedding import EmbeddingFailure, EmbeddingLimits
from rag_ingest.contracts.errors import IngestionError, RetryableError, with_code
from rag_ingest.contracts.rerank import RerankCandidate
from rag_ingest.contracts.vectors import VectorHit
from rag_ingest.domain.chunks import Chunk, SourceSpan
from rag_ingest.domain.search import SearchResult
from rag_ingest.logs import log_event

log = logging.getLogger(__name__)

# A filter matching at most this many documents searches only their vectors.
EXACT_FILTER_DOCUMENTS = 500
# Candidates fetched per wanted result, and the most fetched for one search.
CANDIDATE_FACTOR = 4
MAX_CANDIDATES = 1000
MAX_QUERY_CHARACTERS = 4000


async def embed_query(engine, tenant, query):
    """The query's vector, from the same embedding recipe as the documents."""
    id = uuid4()
    chunk = Chunk(
        tenant_id=tenant,
        document_id=id,
        source_version_id=id,
        generation_id=id,
        ordinal=0,
        text=query,
        spans=(
            SourceSpan(
                extraction_sha256=hashlib.sha256(query.encode()).hexdigest(),
                page_number=1,
                start_char=0,
                end_char=len(query),
            ),
        ),
    )
    settings = engine.settings
    recipe = settings.pipeline()
    result = await engine.embedder.embed_batch(
        (chunk,),
        pipeline=recipe,
        dimensions=settings.embedding_dimensions,
        limits=EmbeddingLimits(
            max_items=1,
            max_input_characters=MAX_QUERY_CHARACTERS,
            max_batch_tokens=8192,
            timeout_seconds=settings.embedding_timeout_seconds,
        ),
    )
    result.validate_for(
        (chunk,),
        pipeline_fingerprint=recipe.fingerprint,
        dimensions=settings.embedding_dimensions,
    )
    item = result.items[0]
    if isinstance(item, EmbeddingFailure):
        raise item.error
    return item.vector


def collapse(results):
    """One result per distinct text (best score first), listing the other places."""
    kept, by_text = [], {}
    for r in results:
        first = by_text.get(r.text_sha256)
        if first is None:
            by_text[r.text_sha256] = len(kept)
            kept.append(r)
            continue
        place = (r.document_id, r.filename, r.pages)
        best = kept[first]
        if place not in best.also_in and (r.document_id, r.pages) != (
            best.document_id,
            best.pages,
        ):
            kept[first] = replace(best, also_in=best.also_in + (place,))
    return kept


def _with_text(results):
    """Drop live hits whose chunk text is missing (should not happen; logged)."""
    kept = [r for r in results if r.text is not None]
    if len(kept) != len(results):
        log_event(
            log,
            "search_chunk_text_missing",
            logging.WARNING,
            hits=len(results) - len(kept),
        )
    return kept


MODES = ("vector", "keyword", "hybrid")


@dataclass(frozen=True)
class SearchOutcome:
    """Results plus how they were produced."""

    results: tuple
    mode: str
    degraded: tuple = ()  # retrievers that failed; the other one answered


def fuse(rankings, *, method, rrf_k, weights):
    """One ranking from several, best first: [(hit, score, ranks)].

    rankings: {name: [hit, ...] best first}. "rrf" scores each ID by
    sum(weight / (rrf_k + rank)) over the rankings it appears in, so only
    order matters and score scales never need to agree. "weighted" scales
    each ranking's scores to 0..1 (min-max) and adds them with the weights.
    """
    combined, hits, ranks = {}, {}, {}
    for name, ranking in rankings.items():
        weight = weights[name]
        scores = [h.score for h in ranking]
        low, high = (min(scores), max(scores)) if scores else (0.0, 0.0)
        for rank, hit in enumerate(ranking, start=1):
            if method == "rrf":
                part = weight / (rrf_k + rank)
            else:
                part = weight * ((hit.score - low) / (high - low) if high > low else 1.0)
            combined[hit.vector_id] = combined.get(hit.vector_id, 0.0) + part
            hits.setdefault(hit.vector_id, hit)
            ranks.setdefault(hit.vector_id, {})[name] = rank
    order = sorted(combined, key=lambda v: (-combined[v], v))
    return [(hits[v], combined[v], ranks[v]) for v in order]


async def _retrieve(name, call, timeout):
    """(name, hits) or (name, error); a retriever's failure is reported, not raised."""
    try:
        if timeout is None:
            return name, await call()
        async with asyncio.timeout(timeout):
            return name, await call()
    except (IngestionError, TimeoutError, OSError) as error:
        # Dependency failures only; a bug in an adapter still raises.
        return name, error


async def _rerank(engine, query, results, top_n, timeout_ms):
    """Reorder the first top_n results by the reranker's scores; the rest keep order.

    Text is cut to the reranker's declared limits. The reply must hold exactly
    one finite score per candidate; anything else is treated as a failed call.
    """
    reranker = engine.reranker
    caps = reranker.capabilities
    n = min(top_n, caps.max_candidates, len(results))
    if n < 2:
        return results
    head, tail = results[:n], results[n:]
    candidates = tuple(
        RerankCandidate(id=r.vector_id, text=r.text[: caps.max_passage_characters])
        for r in head
    )
    async with asyncio.timeout(timeout_ms / 1000):
        scores = await reranker.rerank(query[: caps.max_query_characters], candidates)
    by_id = {s.id: s.score for s in scores}
    if (
        len(scores) != n
        or set(by_id) != {c.id for c in candidates}
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in by_id.values())
    ):
        raise with_code(
            RetryableError("Reranker returned an incomplete or invalid response"),
            "rerank_invalid_response",
        )
    order = sorted(range(n), key=lambda i: (-by_id[head[i].vector_id], i))
    reranked = [
        replace(
            head[i],
            score=float(by_id[head[i].vector_id]),
            ranks=head[i].ranks + (("rerank", position),),
        )
        for position, i in enumerate(order, start=1)
    ]
    return reranked + list(tail)


async def run(engine, tenant, query, *, limit=10, filters=None, mode=None, rerank=None):
    """Up to `limit` distinct passages from the tenant's live documents.

    mode: "vector", "keyword" or "hybrid" (default from INGEST_SEARCH_HYBRID).
    Hybrid embeds the query and searches keywords at the same time, fuses
    the two rankings, then keeps only live, permitted, filter-passing hits.
    rerank: reorder the top INGEST_RERANK_TOP_N with the injected reranker
    (default from INGEST_SEARCH_RERANK).
    """
    settings = engine.settings
    if not isinstance(query, str) or not 1 <= len(query) <= MAX_QUERY_CHARACTERS:
        raise ValueError("Query must be 1..4000 characters")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    mode = mode or ("hybrid" if settings.search_hybrid else "vector")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    keyword = getattr(engine, "keyword", None)
    if mode != "vector" and keyword is None:
        raise ValueError("Keyword search is not configured (INGEST_KEYWORD_INDEX)")
    rerank = settings.search_rerank if rerank is None else rerank
    if rerank and getattr(engine, "reranker", None) is None:
        raise ValueError("No reranker is configured")
    # With reranking, gather a pool to reorder, then cut to the limit.
    target = max(limit, settings.rerank_top_n) if rerank else limit
    if filters is not None and not filters.active:
        filters = None
    scope = ()
    if filters is not None:
        documents = await engine.metadata.filter_documents(
            tenant_id=tenant, filters=filters, limit=EXACT_FILTER_DOCUMENTS + 1
        )
        if not documents:
            return SearchOutcome((), mode)
        if len(documents) <= EXACT_FILTER_DOCUMENTS:
            scope = documents
    keyword_scope = (
        scope if keyword is not None and "document_ids" in keyword.capabilities.filters else ()
    )
    wanted = {"vector": ("vector",), "keyword": ("keyword",), "hybrid": ("vector", "keyword")}[mode]
    weights = {"vector": settings.search_vector_weight, "keyword": settings.search_keyword_weight}
    vector, degraded = None, []
    k = min(MAX_CANDIDATES, target * CANDIDATE_FACTOR)
    while True:
        calls = []
        if "vector" in wanted and "vector" not in degraded:

            async def vector_hits(k=k):
                nonlocal vector
                if vector is None:  # embedded once, reused when fetching more
                    vector = await embed_query(engine, tenant, query)
                return await engine.vectors.search_hits(
                    tenant_id=tenant, vector=vector, k=k, document_ids=scope
                )

            calls.append(_retrieve("vector", vector_hits, None))
        if "keyword" in wanted and "keyword" not in degraded:
            calls.append(
                _retrieve(
                    "keyword",
                    lambda k=k: keyword.search(
                        tenant_id=tenant, query=query, k=k, document_ids=keyword_scope
                    ),
                    settings.search_keyword_timeout_seconds,
                )
            )
        rankings, failures = {}, {}
        for name, outcome in await asyncio.gather(*calls):
            if isinstance(outcome, BaseException):
                failures[name] = outcome
            else:
                rankings[name] = list(outcome)
        if failures:
            if settings.search_strict or not rankings:
                raise next(iter(failures.values()))
            for name, error in failures.items():
                degraded.append(name)
                log_event(
                    log,
                    "search_retriever_failed",
                    logging.WARNING,
                    retriever=name,
                    category=type(error).__name__,
                    reason=str(error)[:200],
                )
        if len(rankings) == 1:
            ((name, ranking),) = rankings.items()
            fused = [(h, h.score, {name: n}) for n, h in enumerate(ranking, start=1)]
        else:
            fused = fuse(
                rankings, method=settings.search_fusion, rrf_k=settings.search_rrf_k, weights=weights
            )
        hits = tuple(
            VectorHit(
                vector_id=h.vector_id,
                document_id=h.document_id,
                generation_id=h.generation_id,
                score=score,
            )
            for h, score, _ in fused
        )
        ranks = {h.vector_id: tuple(sorted(r.items())) for h, _, r in fused}
        results = await engine.metadata.hydrate_search_hits(
            tenant_id=tenant, hits=hits, filters=filters
        )
        results = collapse(
            [replace(r, ranks=ranks[r.vector_id]) for r in _with_text(results)]
        )
        # Every retriever returned fewer than k: nothing more exists to fetch.
        exhausted = all(len(r) < k for r in rankings.values())
        if len(results) >= target or exhausted or k >= MAX_CANDIDATES:
            break
        k = min(MAX_CANDIDATES, k * 2)
    results = results[:target]
    if rerank:
        outcome = await _retrieve(
            "rerank",
            lambda: _rerank(engine, query, results, settings.rerank_top_n, settings.rerank_timeout_ms),
            None,
        )
        if isinstance(outcome[1], BaseException):
            if settings.search_strict:
                raise outcome[1]
            degraded.append("rerank")
            log_event(
                log,
                "search_retriever_failed",
                logging.WARNING,
                retriever="rerank",
                category=type(outcome[1]).__name__,
                reason=str(outcome[1])[:200],
            )
        else:
            results = outcome[1]
    return SearchOutcome(tuple(results[:limit]), mode, tuple(degraded))


async def search(engine, tenant, query, *, limit=10, filters=None, mode=None, rerank=None):
    """The results of run() only."""
    outcome = await run(
        engine, tenant, query, limit=limit, filters=filters, mode=mode, rerank=rerank
    )
    return outcome.results


def result_json(r: SearchResult):
    """API shape of one result (earlier fields kept, new ones added)."""
    return dict(
        document_id=str(r.document_id),
        generation_id=str(r.generation_id),
        vector_id=r.vector_id,
        filename=r.filename,
        text=r.text,
        score=r.score,
        pages=list(r.pages),
        spans=[dict(s) for s in r.spans],
        also_in=[
            dict(document_id=str(d), filename=name, pages=list(pages))
            for d, name, pages in r.also_in
        ],
        # Rank in each retriever that found it, e.g. {"vector": 3, "keyword": 1}.
        ranks=dict(r.ranks),
    )
