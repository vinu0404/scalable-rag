"""Measure the vector index against exact search: recall@k and latency.

Stored vectors are sampled as queries (they follow the real data's
distribution). For each one, the exact top-k (every vector of the tenant,
sorted) is compared with the index's top-k at each ef_search value:

    recall@k = |index top-k ∩ exact top-k| / |exact top-k|

averaged over the samples, with p50/p95 latency of each path. Use it to pick
INGEST_HNSW_EF_SEARCH: the smallest value whose recall meets your target.
Read-only: it never writes.
"""

import json
import statistics
import time

import sqlalchemy as sa


async def sample_queries(vectors, samples):
    """Random stored vectors (tenant, vector) of this target, at most `samples`."""
    async with vectors.db.engine.connect() as c:
        rows = await c.execute(
            sa.text(
                "SELECT tenant_id, embedding::text FROM rag_vectors WHERE target = :target"
                " ORDER BY random() LIMIT :n"
            ),
            dict(target=vectors.target_id, n=samples),
        )
        return [(tenant, tuple(json.loads(v))) for tenant, v in rows]


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


async def evaluate(vectors, *, samples=100, k=10, ef_values=(40, 100, 200)):
    """{'samples', 'k', 'exact': {...}, 'index': {ef: {...}}} with recall and latency."""
    queries = await sample_queries(vectors, samples)
    if not queries:
        raise ValueError("No vectors to evaluate yet")
    exact_ms, truth = [], []
    for tenant, vector in queries:
        start = time.perf_counter()
        hits = await vectors.search_hits(tenant_id=tenant, vector=vector, k=k, exact=True)
        exact_ms.append((time.perf_counter() - start) * 1000)
        truth.append({h.vector_id for h in hits})
    report = {
        "samples": len(queries),
        "k": k,
        "exact": {"p50_ms": _percentile(exact_ms, 0.5), "p95_ms": _percentile(exact_ms, 0.95)},
        "index": {},
    }
    for ef in ef_values:
        recalls, ms = [], []
        for (tenant, vector), expected in zip(queries, truth, strict=True):
            start = time.perf_counter()
            hits = await vectors.search_hits(
                tenant_id=tenant, vector=vector, k=k, ef_search=ef
            )
            ms.append((time.perf_counter() - start) * 1000)
            found = {h.vector_id for h in hits}
            recalls.append(len(found & expected) / len(expected) if expected else 1.0)
        report["index"][ef] = {
            "recall": statistics.fmean(recalls),
            "min_recall": min(recalls),
            # Queries whose index results share nothing with the exact ones: the
            # graph walk ended in the wrong region (raise ef_search; see README).
            "missed": sum(1 for r in recalls if r == 0),
            "p50_ms": _percentile(ms, 0.5),
            "p95_ms": _percentile(ms, 0.95),
        }
    return report


def format_report(report):
    lines = [
        f"{report['samples']} sampled queries, k={report['k']}",
        f"exact search      p50 {report['exact']['p50_ms']:7.1f} ms   p95 {report['exact']['p95_ms']:7.1f} ms",
    ]
    for ef, r in report["index"].items():
        lines.append(
            f"index ef_search={ef:<4} recall {r['recall']:.3f} (min {r['min_recall']:.2f},"
            f" {r['missed']} missed entirely)"
            f"   p50 {r['p50_ms']:7.1f} ms   p95 {r['p95_ms']:7.1f} ms"
        )
    return "\n".join(lines)
