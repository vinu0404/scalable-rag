"""Search benchmark at scale: load synthetic data, then measure the real code paths.

Not part of the package. Run against a DISPOSABLE database only (it drops
indexes and bulk-loads rows that bypass the ingestion pipeline):

    python bench/search_scale.py load    --dsn postgresql://.../ingestion_bench
    python bench/search_scale.py measure --dsn postgresql://.../ingestion_bench

Data (why synthetic and what it models):
- Vectors: 1536 dims, hierarchical clusters (a shared "common direction",
  200 topics, one centroid per document, noise per chunk), so chunks of one
  document are close, one topic is related, other topics are far, and every
  tenant's vectors sit in the same regions of space (the hard case for a
  tenant-filtered ANN index). Uniform random vectors would be unrealistic.
- Text: ~120 pseudo-words per chunk from a 30k-word vocabulary with a
  Zipf-like (log-uniform rank) distribution, plus words of the chunk's topic.
- Tenants: skewed, one large, a few medium, many small.
- Queries: a stored chunk's vector plus noise (not the stored vector itself,
  which would trivially find itself), and words from that chunk's text.

Measured through the package: PgVectorStore.search_hits (HNSW + iterative
scan + completeness guard), PostgresKeywordIndex.search, and the whole
runtime.search.run pipeline (retrieval, fusion, liveness/filter hydration,
collapsing), with only the query embedding call replaced.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import random
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import psycopg

DIMS = 1536
TOPICS = 200
VOCAB = 30_000
WORDS_PER_CHUNK = 120
CHUNKS_PER_DOC = 50
# (class, tenants, documents per tenant): 150k + 100k + 50k = 300k chunks.
TENANTS = (("large", 1, 3000), ("medium", 5, 400), ("small", 50, 20))
TARGET = "pgvector-bench"
TAG_DEPTS, TAG_PROJECTS = 10, 300
SEED = 7

log = logging.getLogger("bench")


def tenant_plan():
    rnd = random.Random(SEED)
    plan = []
    for cls, count, docs in TENANTS:
        for _ in range(count):
            plan.append((cls, uuid.UUID(int=rnd.getrandbits(128), version=4), docs))
    return plan


SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def word(i):
    """Deterministic pronounceable pseudo-word for vocabulary rank i."""
    out, n = [], i + 1
    while n:
        n, r = divmod(n, len(SYLLABLES))
        out.append(SYLLABLES[r])
    return "".join(out) + ("x" if len(out) == 1 else "")


VOCABULARY = [word(i) for i in range(VOCAB)]
# Each topic owns 15 mid-frequency words (ranks 2000..), used often in its chunks.
RANK = {w: i for i, w in enumerate(VOCABULARY)}
TOPIC_WORDS = [VOCABULARY[2000 + t * 15 : 2000 + (t + 1) * 15] for t in range(TOPICS)]


def chunk_text(rnd, topic):
    words = []
    for _ in range(WORDS_PER_CHUNK):
        if rnd.random() < 0.15:
            words.append(rnd.choice(TOPIC_WORDS[topic]))
        else:
            words.append(VOCABULARY[int(VOCAB ** rnd.random()) - 1])
    return " ".join(words)


# ---- load -----------------------------------------------------------------------------------


def connect(dsn, **kw):
    return psycopg.connect(dsn, autocommit=True, **kw)


def timed(label, fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    seconds = time.perf_counter() - start
    print(f"{label:<44} {seconds:8.1f} s", flush=True)
    return seconds, result


def migrate(dsn):
    from alembic import command
    from alembic.config import Config

    import rag_ingest.cli as cli
    from pathlib import Path

    config = Config()
    config.set_main_option("script_location", str(Path(cli.__file__).parent / "migrations"))
    config.attributes["dsn"] = dsn
    command.upgrade(config, "head")

    async def provision():
        from rag_ingest.adapters.postgres.connection import Database
        from rag_ingest.adapters.postgres.vectors import PgVectorStore

        store = PgVectorStore(Database(dsn), dimensions=DIMS, target_id=TARGET, index="none")
        try:
            await store.provision()
        finally:
            await store.aclose()

    asyncio.run(provision())
    with connect(dsn) as c:
        # A fresh start each run; bulk load first, build indexes after (timed separately).
        c.execute("TRUNCATE rag_vectors, search_chunks, documents CASCADE")
        c.execute("DROP TABLE IF EXISTS bench_docs, bench_topics, bench_common")
        c.execute("DROP INDEX IF EXISTS ix_search_chunks_keyword")
        c.execute("DROP INDEX IF EXISTS rag_vectors_hnsw")


def load_text(dsn, plan):
    """Documents and chunk rows (text only) by COPY; FK triggers off for the bulk load."""
    rnd = random.Random(SEED + 1)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    docs = []  # (tenant, doc, gen, topic)
    with connect(dsn) as c:
        c.execute("SET session_replication_role = replica")
        with c.cursor() as cur:
            with cur.copy(
                "COPY documents (tenant_id, id, payload, active_id, tags, metadata, created_at)"
                " FROM STDIN"
            ) as cp:
                for cls, tenant, count in plan:
                    for n in range(count):
                        doc, gen = uuid.uuid4(), uuid.uuid4()
                        topic = rnd.randrange(TOPICS)
                        docs.append((tenant, doc, gen, topic))
                        created = base + timedelta(minutes=rnd.randrange(60 * 24 * 365))
                        payload = {"fields": {"display_name": f"{cls}-{n}.pdf", "acl_ref": "tenant"}}
                        tags = "{dept-%d,project-%d}" % (
                            rnd.randrange(TAG_DEPTS), rnd.randrange(TAG_PROJECTS)
                        )
                        cp.write_row((
                            tenant, doc, json.dumps(payload), gen, tags,
                            json.dumps({"year": created.year, "topic": topic}), created,
                        ))
            spans = json.dumps([{"page_number": 1, "start_char": 0, "end_char": 1}])
            with cur.copy(
                "COPY search_chunks (tenant_id, vector_id, document_id, generation_id, ordinal,"
                " content, spans, text_sha256) FROM STDIN"
            ) as cp:
                for tenant, doc, gen, topic in docs:
                    for o in range(CHUNKS_PER_DOC):
                        text = chunk_text(rnd, topic)
                        cp.write_row((
                            tenant, f"{doc}:{o}", doc, gen, o, text, spans,
                            hashlib.sha256(text.encode()).hexdigest(),
                        ))
        c.execute("CREATE TABLE IF NOT EXISTS bench_docs (doc uuid PRIMARY KEY, topic int)")
        with c.cursor() as cur, cur.copy("COPY bench_docs FROM STDIN") as cp:
            for _, doc, _, topic in docs:
                cp.write_row((doc, topic))
        c.execute("ANALYZE")
    return len(docs) * CHUNKS_PER_DOC


def noise(norm):
    """SQL for a random vector of about this L2 norm (correlated on the row, so per row)."""
    sigma = norm / DIMS**0.5
    return (
        f"(SELECT array_agg(random_normal(0, {sigma}))::vector({DIMS})"
        " FROM generate_series(1, %d) WHERE g.k IS NOT NULL)" % DIMS
    )


def make_centroids(dsn):
    with connect(dsn) as c:
        c.execute("SELECT setseed(0.42)")
        c.execute(f"CREATE TABLE IF NOT EXISTS bench_common (v vector({DIMS}))")
        c.execute(f"INSERT INTO bench_common SELECT {noise(1.0)} FROM (SELECT 1 AS k) g")
        c.execute(f"CREATE TABLE IF NOT EXISTS bench_topics (topic int PRIMARY KEY, c vector({DIMS}))")
        # Anisotropy like real embeddings: every topic shares a common direction.
        c.execute(
            "INSERT INTO bench_topics SELECT g.k, l2_normalize((SELECT v FROM bench_common)"
            f" + {noise(2.0)}) FROM generate_series(0, {TOPICS - 1}) AS g(k)"
        )
        c.execute(f"ALTER TABLE bench_docs ADD COLUMN IF NOT EXISTS c vector({DIMS})")
        c.execute(
            "UPDATE bench_docs d SET c = l2_normalize(t.c + "
            + noise(0.8).replace("g.k", "d.topic")
            + ") FROM bench_topics t WHERE t.topic = d.topic"
        )


def make_vectors_part(dsn, tenant, part, parts):
    """One slice of a tenant's chunks: document centroid + per-chunk noise."""
    with connect(dsn) as c:
        c.execute("SET synchronous_commit = off")  # bulk load only
        c.execute(
            f"""INSERT INTO rag_vectors (target, tenant_id, document_id, generation_id, vector_id,
                payload_hash, payload, embedding)
            SELECT %s, sc.tenant_id, sc.document_id, sc.generation_id, sc.vector_id,
                sc.text_sha256, jsonb_build_object('vector_id', sc.vector_id,
                    'document_id', sc.document_id, 'generation_id', sc.generation_id,
                    'ordinal', sc.ordinal),
                l2_normalize(d.c + {noise(0.8).replace("g.k", "sc.ordinal")})
            FROM search_chunks sc JOIN bench_docs d ON d.doc = sc.document_id
            WHERE sc.tenant_id = %s AND abs(hashtext(sc.document_id::text)) %% %s = %s""",
            (TARGET, tenant, parts, part),
        )


def make_vectors(dsn, plan, workers=8):
    jobs = []
    for _, tenant, docs in plan:
        parts = max(1, docs // 100)
        jobs += [(dsn, tenant, p, parts) for p in range(parts)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda j: make_vectors_part(*j), jobs))
    with connect(dsn) as c:
        c.execute("ANALYZE rag_vectors")


def make_tsvectors(dsn, plan, workers=8):
    """The same statement PostgresKeywordIndex.index_batch runs, over every row."""

    def one(tenant):
        with connect(dsn) as c:
            c.execute(
                "UPDATE search_chunks SET search_tsv = to_tsvector('english', content),"
                " keyword_target = 'postgres-fts:english' WHERE tenant_id = %s",
                (tenant,),
            )

    with ThreadPoolExecutor(workers) as pool:
        # The large tenant is split by its own parallelism inside one statement; fine here.
        list(pool.map(one, [t for _, t, _ in plan]))


def build_gin(dsn):
    with connect(dsn) as c:
        c.execute("SET maintenance_work_mem = '1GB'")
        c.execute(
            "CREATE INDEX ix_search_chunks_keyword ON search_chunks USING gin (tenant_id, search_tsv)"
        )


def paradedb_example():
    """The ParadeDB KeywordIndex from examples/ (it is not part of the package)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples" / "paradedb_keyword.py"
    spec = importlib.util.spec_from_file_location("paradedb_keyword", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ParadeDBKeywordIndex


def build_bm25(dsn):
    async def run():
        from rag_ingest.adapters.postgres.connection import Database
        ParadeDBKeywordIndex = paradedb_example()

        index = ParadeDBKeywordIndex(Database(dsn))
        try:
            await index.provision()
        finally:
            await index.db.engine.dispose()

    asyncio.run(run())


def build_hnsw_with_package(dsn):
    """The package's own provisioning path (CREATE INDEX CONCURRENTLY via PgVectorStore)."""

    async def run():
        from rag_ingest.adapters.postgres.connection import Database
        from rag_ingest.adapters.postgres.vectors import PgVectorStore

        store = PgVectorStore(Database(dsn), dimensions=DIMS, target_id=TARGET)
        try:
            await store._provision_index()
        finally:
            await store.aclose()

    asyncio.run(run())


def sizes(dsn):
    with connect(dsn) as c:
        for name in ("rag_vectors", "rag_vectors_hnsw", "search_chunks", "ix_search_chunks_keyword"):
            size = c.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (name,)).fetchone()
            print(f"  size {name:<28} {size[0]}")


def incremental_inserts(dsn, rows=5000, batch=512):
    """Insert rows with the HNSW index live (what ingestion does), one commit per batch.

    The new vectors are generated first, so only the insert (heap, TOAST, the
    scope btree and the HNSW graph) is timed.
    """
    with connect(dsn) as c:
        tenant = c.execute(
            "SELECT tenant_id FROM rag_vectors GROUP BY tenant_id ORDER BY count(*) DESC LIMIT 1"
        ).fetchone()[0]
        c.execute("DROP TABLE IF EXISTS bench_new")
        c.execute(
            f"""CREATE TABLE bench_new AS SELECT row_number() OVER () AS n, v.tenant_id,
                v.document_id, v.generation_id, 'inc:' || v.vector_id AS vector_id,
                v.payload_hash, v.payload,
                l2_normalize(v.embedding + {noise(0.3).replace("g.k", "v.vector_id")}) AS embedding
            FROM rag_vectors v WHERE v.target = %s AND v.tenant_id = %s LIMIT %s""",
            (TARGET, tenant, rows),
        )
        start = time.perf_counter()
        for low in range(0, rows, batch):
            c.execute(
                """INSERT INTO rag_vectors (target, tenant_id, document_id, generation_id,
                    vector_id, payload_hash, payload, embedding)
                SELECT %s, tenant_id, document_id, generation_id, vector_id, payload_hash,
                    payload, embedding FROM bench_new WHERE n > %s AND n <= %s""",
                (TARGET, low, low + batch),
            )
        seconds = time.perf_counter() - start
        c.execute("DELETE FROM rag_vectors WHERE vector_id LIKE 'inc:%%'")
        c.execute("DROP TABLE bench_new")
    print(f"insert with HNSW live: {rows} rows in {seconds:.1f} s = {rows / seconds:,.0f} rows/s"
          f" (batches of {batch}, synchronous commit)")


def load(dsn):
    plan = tenant_plan()
    timed("migrate + provision (no indexes)", migrate, dsn)
    secs, chunks = timed("generate + COPY documents and chunk text", load_text, dsn, plan)
    print(f"  {chunks:,} chunks, {sum(d for *_, d in plan):,} documents, {len(plan)} tenants")
    timed("topic and document centroids", make_centroids, dsn)
    secs, _ = timed("vectors (generated in SQL, 8 workers)", make_vectors, dsn, plan)
    secs, _ = timed("keyword tsvectors (8 workers)", make_tsvectors, dsn, plan)
    print(f"  = {chunks / secs:,.0f} chunks/s")
    timed("GIN keyword index build", build_gin, dsn)
    with connect(dsn) as c:
        has_bm25 = c.execute(
            "SELECT count(*) FROM pg_available_extensions WHERE name = 'pg_search'"
        ).fetchone()[0]
    if has_bm25:
        timed("BM25 index build (ParadeDBKeywordIndex.provision)", build_bm25, dsn)
    try:
        secs, _ = timed("HNSW build via PgVectorStore (m=16, efc=64)", build_hnsw_with_package, dsn)
        print(f"  = {chunks / secs:,.0f} vectors/s")
    except Exception as error:  # report the package's behaviour, then build it directly
        print(f"  package provisioning FAILED: {type(error).__name__}: {str(error)[:200]}")
        with connect(dsn) as c:
            c.execute("DROP INDEX IF EXISTS rag_vectors_hnsw")
            c.execute("SET maintenance_work_mem = '2GB'")
            secs, _ = timed(
                "HNSW build direct (m=16, efc=64)",
                c.execute,
                "CREATE INDEX rag_vectors_hnsw ON rag_vectors USING hnsw"
                " (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
            )
            print(f"  = {chunks / secs:,.0f} vectors/s")
    with connect(dsn) as c:
        c.execute("VACUUM ANALYZE")
    sizes(dsn)
    incremental_inserts(dsn)


# ---- measure --------------------------------------------------------------------------------


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summary(ms):
    return f"p50 {pct(ms, .5):6.1f}  p95 {pct(ms, .95):6.1f}  p99 {pct(ms, .99):6.1f} ms"


class Counter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = {}

    def emit(self, record):
        name = getattr(record, "event", None) or record.getMessage()[:60]
        self.events[name] = self.events.get(name, 0) + 1


def make_queries(dsn, per_class, seed=11):
    """Per tenant class: (tenant, query vector, query text, source vector_id)."""
    rnd = random.Random(seed)
    out = {}
    with connect(dsn) as c:
        c.execute(f"SELECT setseed({seed / 100})")
        for cls, *_ in TENANTS:
            tenants = [t for k, t, _ in tenant_plan() if k == cls]
            rows = []
            for i in range(per_class):
                tenant = tenants[i % len(tenants)]
                vid, text, vec = c.execute(
                    f"""SELECT sc.vector_id, sc.content,
                        l2_normalize(v.embedding + {noise(0.6).replace("g.k", "v.vector_id")})::text
                    FROM search_chunks sc JOIN rag_vectors v ON v.tenant_id = sc.tenant_id
                        AND v.vector_id = sc.vector_id AND v.target = %s
                    WHERE sc.tenant_id = %s ORDER BY random() LIMIT 1""",
                    (TARGET, tenant),
                ).fetchone()
                words = text.split()
                # A realistic short query: 2 distinctive words of the chunk + 2 common ones.
                rare = sorted(set(words), key=lambda w: -RANK[w])[:2]
                query = " ".join(rare + rnd.sample(words, 2))
                rows.append((tenant, tuple(json.loads(vec)), query, vid))
            out[cls] = rows
    return out


async def measure(dsn, per_class, concurrency, keyword_engine="postgres", skip_vector=False):

    from rag_ingest.adapters.postgres.connection import Database
    from rag_ingest.adapters.postgres.keyword import PostgresKeywordIndex
    from rag_ingest.adapters.postgres.metadata import PostgresMetadataStore
    from rag_ingest.adapters.postgres.vectors import PgVectorStore
    from rag_ingest.config import Settings
    from rag_ingest.domain.search import SearchFilters
    from rag_ingest.runtime import search as pipeline
    from rag_ingest.runtime.settings import ServiceSettings

    counter = Counter()
    logging.getLogger("rag_ingest").addHandler(counter)
    logging.getLogger("rag_ingest").setLevel(logging.WARNING)
    queries = make_queries(dsn, per_class)
    db = Database(dsn, pool_size=max(5, concurrency))
    vectors = PgVectorStore(db, dimensions=DIMS, target_id=TARGET)
    if keyword_engine == "paradedb":
        ParadeDBKeywordIndex = paradedb_example()

        keyword = ParadeDBKeywordIndex(db)
    else:
        keyword = PostgresKeywordIndex(db)
    # One keyword engine is active at a time (rows carry its target): make it this one.
    with connect(dsn) as c:
        c.execute(
            "UPDATE search_chunks SET keyword_target = %s WHERE keyword_target IS DISTINCT FROM %s",
            (keyword.target_id, keyword.target_id),
        )
        c.execute("VACUUM ANALYZE search_chunks")
    metadata = PostgresMetadataStore(db)
    vector_of = {q: v for rows in queries.values() for _, v, q, _ in rows}

    async def fake_embed(engine, tenant, query):
        return vector_of[query]

    pipeline.embed_query = fake_embed  # the only replaced step: no embedding API calls

    class Engine:
        pass

    engine = Engine()
    engine.settings = ServiceSettings(
        core=Settings(metadata_dsn=dsn, s3_bucket="b",
                      sqs_queue_url="https://sqs.us-east-1.amazonaws.com/1/q"),
        vector_dsn=dsn, bucket="b", embedding_driver="deterministic",
        search_hybrid=True, keyword_index="postgres",  # the engine object is injected below
    )
    engine.vectors, engine.keyword, engine.metadata, engine.reranker = vectors, keyword, metadata, None

    # Warm the caches once (numbers below are warm-cache numbers).
    for rows in queries.values():
        for tenant, vec, _, _ in rows[:20]:
            await vectors.search_hits(tenant_id=tenant, vector=vec, k=10)

    print(f"\n== vector search (PgVectorStore.search_hits), {per_class} queries per class ==")
    for cls, rows in ({} if skip_vector else queries).items():
        for k in (10, 40):
            truth, exact_ms = [], []
            for tenant, vec, _, _ in rows:
                s = time.perf_counter()
                hits = await vectors.search_hits(tenant_id=tenant, vector=vec, k=k, exact=True)
                exact_ms.append((time.perf_counter() - s) * 1000)
                truth.append({h.vector_id for h in hits})
            print(f"{cls:<6} k={k:<3} exact            {summary(exact_ms)}")
            for ef in (40, 100, 200):
                before = dict(counter.events)
                recalls, ms = [], []
                for (tenant, vec, _, _), expected in zip(rows, truth, strict=True):
                    s = time.perf_counter()
                    hits = await vectors.search_hits(tenant_id=tenant, vector=vec, k=k, ef_search=ef)
                    ms.append((time.perf_counter() - s) * 1000)
                    recalls.append(len({h.vector_id for h in hits} & expected) / len(expected))
                guard = counter.events.get("vector_search_scan_limit_reached", 0) - before.get(
                    "vector_search_scan_limit_reached", 0
                )
                print(
                    f"{cls:<6} k={k:<3} hnsw ef={ef:<4} recall {statistics.fmean(recalls):.3f}"
                    f" (min {min(recalls):.2f})  {summary(ms)}"
                    + (f"  scan-limit fallbacks {guard}" if guard else "")
                )

    print(f"\n== keyword search ({type(keyword).__name__}.search, k=40) ==")
    for cls, rows in queries.items():
        ms, found = [], 0
        for tenant, _, q, vid in rows:
            s = time.perf_counter()
            hits = await keyword.search(tenant_id=tenant, query=q, k=40)
            ms.append((time.perf_counter() - s) * 1000)
            found += vid in {h.vector_id for h in hits[:10]}
        print(f"{cls:<6} {summary(ms)}   source chunk in top 10: {found / len(rows):.0%}")

    async def pipeline_run(label, rows, **kw):
        ms, found = [], 0
        for tenant, _, q, vid in rows:
            s = time.perf_counter()
            outcome = await pipeline.run(engine, tenant, q, limit=10, **kw)
            ms.append((time.perf_counter() - s) * 1000)
            found += vid in {r.vector_id for r in outcome.results}
        print(f"{label:<34} {summary(ms)}   source chunk in top 10: {found / len(rows):.0%}")

    print("\n== whole search pipeline (runtime.search.run, limit=10) ==")
    for cls, rows in queries.items():
        for mode in ("vector", "keyword", "hybrid"):
            await pipeline_run(f"{cls:<6} {mode}", rows, mode=mode)

    large = queries["large"]
    print("\n== filters on the large tenant (hybrid) ==")
    async with db.engine.connect() as c:
        import sqlalchemy as sa

        tenant = large[0][0]
        counts = dict(
            (await c.execute(sa.text(
                "SELECT t, count(*) FROM documents, unnest(tags) t WHERE tenant_id = :t"
                " AND t IN ('dept-3', 'project-42') GROUP BY t"), dict(t=tenant))).all()
        )
    for label, filters in (
        (f"narrow tag ({counts.get('project-42', 0)} docs, exact path)",
         SearchFilters.build(tags_any=["project-42"])),
        (f"broad tag ({counts.get('dept-3', 0)} docs, index + post-filter)",
         SearchFilters.build(tags_any=["dept-3"])),
        ("metadata year=2024 (all docs)", SearchFilters.build(metadata={"year": 2024})),
    ):
        ms = []
        for tenant, _, q, _ in large:
            s = time.perf_counter()
            await pipeline.run(engine, tenant, q, limit=10, mode="hybrid", filters=filters)
            ms.append((time.perf_counter() - s) * 1000)
        print(f"{label:<46} {summary(ms)}")

    print(f"\n== concurrency: {concurrency} clients, hybrid, mixed tenants ==")
    mixed = [r for rows in queries.values() for r in rows]
    random.Random(3).shuffle(mixed)
    lat = []

    async def client(items):
        for tenant, _, q, _ in items:
            s = time.perf_counter()
            await pipeline.run(engine, tenant, q, limit=10, mode="hybrid")
            lat.append((time.perf_counter() - s) * 1000)

    s = time.perf_counter()
    await asyncio.gather(*(client(mixed[i::concurrency]) for i in range(concurrency)))
    wall = time.perf_counter() - s
    print(f"{len(mixed)} queries in {wall:.1f} s = {len(mixed) / wall:.0f} queries/s   {summary(lat)}")
    if counter.events:
        print("warnings logged:", counter.events)
    await db.engine.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("load", "measure"))
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--queries", type=int, default=100, help="queries per tenant class")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--keyword", choices=("postgres", "paradedb"), default="postgres")
    parser.add_argument("--skip-vector", action="store_true", help="skip the vector recall section")
    args = parser.parse_args()
    if "bench" not in args.dsn.rsplit("/", 1)[-1]:
        sys.exit("Refusing: the database name must contain 'bench' (this drops indexes).")
    if args.command == "load":
        load(args.dsn)
    else:
        asyncio.run(measure(args.dsn, args.queries, args.concurrency, args.keyword, args.skip_vector))


if __name__ == "__main__":
    main()
