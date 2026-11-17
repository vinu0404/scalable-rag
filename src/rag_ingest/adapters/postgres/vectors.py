"""Store generation-scoped immutable embeddings in pgvector.

The default implementation uses exact cosine search. Add ANN indexes only
through a reviewed migration with measured recall and filtering behavior.
"""

import hashlib
import json
import logging
import struct
import sqlalchemy as sa
from .metadata import binding
from rag_ingest.logs import log_event
from rag_ingest.domain.limits import MAX_BATCH_ITEMS, MAX_DIGEST_IDS
from rag_ingest.contracts.vectors import (
    VectorCheck,
    VectorHit,
    VectorReadiness,
    VectorWriteResult,
)
from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    UncertainOutcomeError,
)


log = logging.getLogger(__name__)
INDEX_NAME = "rag_vectors_hnsw"
# pgvector can index `vector` up to 2,000 dimensions and `halfvec` up to 4,000.
VECTOR_INDEX_MAX_DIMENSIONS = 2000
HALFVEC_INDEX_MAX_DIMENSIONS = 4000


class PgVectorStore:
    """Bind one configured model dimension to a concrete logical target.

    Search uses an HNSW index (approximate nearest neighbours) with pgvector's
    iterative scans, so tenant and document filters cannot starve results;
    see search_hits. index="none" keeps exact search only.
    """

    def __init__(
        self,
        database,
        *,
        dimensions=1536,
        target_id="pgvector-default",
        embedding_spec=None,
        index="hnsw",
        hnsw_m=16,
        hnsw_ef_construction=64,
        hnsw_ef_search=200,
        hnsw_max_scan_tuples=20_000,
    ):
        """Validate SQL-shape inputs; connection lifecycle stays explicit."""
        if type(dimensions) is not int or not 1 <= dimensions <= 16000:
            raise ValueError("Unsupported vector dimensions")
        if index not in ("hnsw", "none"):
            raise ValueError("index must be 'hnsw' or 'none'")
        for name, value, low, high in (
            ("hnsw_m", hnsw_m, 2, 100),
            ("hnsw_ef_construction", hnsw_ef_construction, 4, 1000),
            ("hnsw_ef_search", hnsw_ef_search, 1, 1000),
            ("hnsw_max_scan_tuples", hnsw_max_scan_tuples, 1000, 100_000_000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be {low}..{high}")
        if hnsw_ef_construction < 2 * hnsw_m:
            raise ValueError("hnsw_ef_construction must be at least 2 * hnsw_m")
        self.db = database
        self.dimensions = dimensions
        self._target = target_id
        self.expected_embedding = (
            binding(embedding_spec.to_dict()) if embedding_spec is not None else None
        )
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_max_scan_tuples = hnsw_max_scan_tuples
        # What the index covers and how a query must be written to use it.
        if index == "none" or dimensions > HALFVEC_INDEX_MAX_DIMENSIONS:
            self.index = None
            self._key, self._query = "v.embedding", "CAST(:vector AS vector)"
        elif dimensions > VECTOR_INDEX_MAX_DIMENSIONS:
            # Half precision for ranking only; verification keeps exact float32.
            self.index = (f"(embedding::halfvec({dimensions}))", "halfvec_cosine_ops")
            self._key = f"(v.embedding::halfvec({dimensions}))"
            self._query = f"CAST(:vector AS halfvec({dimensions}))"
        else:
            self.index = ("embedding", "vector_cosine_ops")
            self._key, self._query = "v.embedding", "CAST(:vector AS vector)"

    @property
    def target_id(self):
        """Return the durable logical index identity, without credentials."""
        return self._target

    async def provision(self):
        """Explicit setup only: create extension and the vector table if absent."""
        async with self.db.engine.begin() as c:
            await c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            await c.execute(
                sa.text(
                    "CREATE TABLE IF NOT EXISTS rag_vector_targets (target text PRIMARY KEY, embedding_recipe text NOT NULL)"
                )
            )
            if self.expected_embedding:
                await c.execute(
                    sa.text(
                        "INSERT INTO rag_vector_targets(target,embedding_recipe) VALUES (:target,:recipe) ON CONFLICT DO NOTHING"
                    ),
                    dict(target=self.target_id, recipe=self.expected_embedding),
                )
                await self._ensure_target(c)
            await c.execute(
                sa.text(f"""CREATE TABLE IF NOT EXISTS rag_vectors (
                target text NOT NULL, tenant_id uuid NOT NULL, document_id uuid NOT NULL,
                generation_id uuid NOT NULL, vector_id text NOT NULL, payload_hash text NOT NULL,
                payload jsonb NOT NULL, embedding vector({self.dimensions}) NOT NULL,
                PRIMARY KEY(target,tenant_id,vector_id))""")
            )
            await c.execute(
                sa.text(
                    "CREATE INDEX IF NOT EXISTS rag_vectors_scope ON rag_vectors(target,tenant_id,document_id,generation_id)"
                )
            )
        await self._provision_index()
        await self.doctor()

    async def _index_state(self, c):
        """(exists, valid, reloptions) of the HNSW index."""
        row = (
            await c.execute(
                sa.text(
                    "SELECT i.indisvalid, cl.reloptions FROM pg_class cl"
                    " JOIN pg_index i ON i.indexrelid = cl.oid WHERE cl.relname = :name"
                ),
                dict(name=INDEX_NAME),
            )
        ).first()
        return (False, False, None) if row is None else (True, row[0], row[1] or [])

    async def _provision_index(self):
        """Build the HNSW index CONCURRENTLY (writes continue during the build).

        A failed concurrent build leaves an invalid index that IF NOT EXISTS
        would skip forever, so an invalid one is dropped and rebuilt. Changing
        m or ef_construction needs a deliberate rebuild (DROP INDEX CONCURRENTLY
        rag_vectors_hnsw, then migrate); a mismatch is reported, not applied.
        """
        if self.index is None:
            log_event(
                log,
                "vector_index_disabled",
                logging.WARNING,
                dimensions=self.dimensions,
                hint="search is exact; every query reads all of a tenant's vectors",
            )
            return
        expression, ops = self.index
        async with self.db.engine.connect() as c:
            c = await c.execution_options(isolation_level="AUTOCOMMIT")
            try:
                await self._build_index(c, expression, ops)
            finally:
                # The session timeout was lifted for the build: never reuse this connection.
                await c.invalidate()

    async def _build_index(self, c, expression, ops):
        # A build over existing rows takes minutes (300k x 1536 dims: ~95 s with
        # 2 GB maintenance_work_mem), far beyond the pool's 15 s statement limit.
        await c.execute(sa.text("SET statement_timeout = 0"))
        exists, valid, options = await self._index_state(c)
        if exists and not valid:
            await c.execute(sa.text(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}"))
            exists = False
        if not exists:
            await c.execute(
                sa.text(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} ON rag_vectors"
                    f" USING hnsw ({expression} {ops})"
                    f" WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})"
                )
            )
            return
        wanted = {f"m={self.hnsw_m}", f"ef_construction={self.hnsw_ef_construction}"}
        if set(options) != wanted:
            log_event(
                log,
                "vector_index_settings_differ",
                logging.WARNING,
                built=",".join(sorted(options)),
                configured=",".join(sorted(wanted)),
                hint="DROP INDEX CONCURRENTLY rag_vectors_hnsw, then run migrate to rebuild",
            )

    async def doctor(self):
        """Reject a table whose vector dimensions differ from configuration."""
        async with self.db.engine.connect() as c:
            await self._ensure_target(c)
            value = (
                await c.execute(
                    sa.text(
                        "SELECT format_type(atttypid,atttypmod) FROM pg_attribute WHERE attrelid='rag_vectors'::regclass AND attname='embedding'"
                    )
                )
            ).scalar_one()
            if value != f"vector({self.dimensions})":
                raise OperatorActionRequiredError(
                    "pgvector dimensions do not match configuration"
                )

    async def _ensure_target(self, connection):
        """Reject model/recipe drift inside one logical vector target."""
        if self.expected_embedding is None:
            return
        recipe = (
            await connection.execute(
                sa.text(
                    "SELECT embedding_recipe FROM rag_vector_targets WHERE target=:target"
                ),
                dict(target=self.target_id),
            )
        ).scalar_one_or_none()
        if recipe != self.expected_embedding:
            raise OperatorActionRequiredError(
                "Vector target embedding recipe changed; provision a distinct target/database"
            )

    def _batch(self, scope, items):
        """Bound requests and validate scope before constructing vector SQL values."""
        if not isinstance(items, tuple) or not 1 <= len(items) <= MAX_BATCH_ITEMS:
            raise ValueError(f"Vector batch must contain 1..{MAX_BATCH_ITEMS} items")
        if len({i.chunk.vector_id for i in items}) != len(items):
            raise ValueError("Duplicate vector identities")
        for item in items:
            scope.validate_item(item)
            if item.dimensions != self.dimensions:
                raise ValueError("Vector dimension mismatch")
            try:
                quantized = tuple(
                    struct.unpack("!f", struct.pack("!f", v))[0] for v in item.vector
                )
            except (OverflowError, struct.error):
                raise PermanentInputError("Vector exceeds float32 range") from None
            if not any(quantized):
                raise PermanentInputError(
                    "Cosine vectors must be nonzero at float32 precision"
                )

    async def upsert_batch(self, scope, *, items):
        """Create stable IDs; reject different payloads instead of overwriting them."""
        self._batch(scope, items)
        results = []
        async with self.db.transaction(scope.tenant_id, lock=False) as (c, now):
            await self._ensure_target(c)
            rows = [
                (
                    item.chunk.vector_id,
                    binding(item),
                    # Identity only: the vector lives in `embedding`, the text in
                    # the chunk table. payload_hash still covers the full item.
                    json.dumps(
                        {
                            "vector_id": item.chunk.vector_id,
                            "document_id": str(item.chunk.document_id),
                            "generation_id": str(item.chunk.generation_id),
                            "ordinal": item.chunk.ordinal,
                        }
                    ),
                    json.dumps(item.vector, allow_nan=False),
                )
                for item in items
            ]
            ids = [r[0] for r in rows]
            scope_args = dict(
                target=self.target_id,
                tenant=str(scope.tenant_id),
                doc=str(scope.document_id),
                gen=str(scope.generation_id),
            )
            # One statement for the batch; existing rows are kept, never replaced.
            await c.execute(
                sa.text("""INSERT INTO rag_vectors(target,tenant_id,document_id,generation_id,vector_id,payload_hash,payload,embedding)
                SELECT :target,CAST(:tenant AS uuid),CAST(:doc AS uuid),CAST(:gen AS uuid),
                    r.id,r.hash,CAST(r.payload AS jsonb),CAST(r.vector AS vector)
                FROM unnest(CAST(:ids AS text[]),CAST(:hashes AS text[]),
                    CAST(:payloads AS text[]),CAST(:vectors AS text[])) AS r(id,hash,payload,vector)
                ON CONFLICT (target,tenant_id,vector_id) DO NOTHING"""),
                dict(
                    scope_args,
                    ids=ids,
                    hashes=[r[1] for r in rows],
                    payloads=[r[2] for r in rows],
                    vectors=[r[3] for r in rows],
                ),
            )
            # One read back: whatever is stored now (ours or an earlier writer's).
            stored = dict(
                (
                    await c.execute(
                        sa.text(
                            "SELECT vector_id,payload_hash FROM rag_vectors WHERE target=:target"
                            " AND tenant_id=CAST(:tenant AS uuid) AND vector_id = ANY(CAST(:ids AS text[]))"
                        ),
                        dict(scope_args, ids=ids),
                    )
                ).all()
            )
            for vector_id, payload_hash, _, _ in rows:
                if vector_id not in stored:
                    raise UncertainOutcomeError(
                        "Vector row missing right after its insert"
                    )
                error = (
                    None
                    if stored[vector_id] == payload_hash
                    else PermanentInputError(
                        "Immutable vector identity contains incompatible output"
                    )
                )
                results.append(VectorWriteResult(vector_id=vector_id, error=error))
        return tuple(results)

    async def verify_batch(self, scope, *, expected):
        """Verify every expected payload and float32 vector using a fresh query."""
        self._batch(scope, expected)
        results = []
        async with self.db.transaction(scope.tenant_id, lock=False) as (c, now):
            await self._ensure_target(c)
            found = {
                row["vector_id"]: row
                for row in (
                    await c.execute(
                        sa.text("""SELECT vector_id,payload_hash,embedding::text AS vector FROM rag_vectors
                    WHERE target=:target AND tenant_id=CAST(:tenant AS uuid) AND document_id=CAST(:doc AS uuid)
                    AND generation_id=CAST(:gen AS uuid) AND vector_id = ANY(CAST(:ids AS text[]))"""),
                        dict(
                            target=self.target_id,
                            tenant=str(scope.tenant_id),
                            doc=str(scope.document_id),
                            gen=str(scope.generation_id),
                            ids=[item.chunk.vector_id for item in expected],
                        ),
                    )
                ).mappings()
            }
            for item in expected:
                row = found.get(item.chunk.vector_id)
                ready = VectorReadiness.MISSING
                if row:
                    actual = json.loads(row["vector"])
                    same = len(actual) == len(item.vector) and all(
                        struct.pack("!f", a) == struct.pack("!f", b)
                        for a, b in zip(actual, item.vector, strict=True)
                    )
                    ready = (
                        VectorReadiness.READY
                        if row["payload_hash"] == binding(item) and same
                        else VectorReadiness.MISMATCH
                    )
                results.append(
                    VectorCheck(vector_id=item.chunk.vector_id, readiness=ready)
                )
        return tuple(results)

    MAX_DIGEST_IDS = MAX_DIGEST_IDS  # shared with the engine's digest queries

    @staticmethod
    def _digest_line(vector_id, payload_hash, vector_sha256):
        return f"{vector_id}\t{payload_hash}\t{vector_sha256}\n"

    def expected_digest(self, scope, items):
        """Digest of exactly what verify_batch compares, computed from the build.

        One line per vector, sorted by ID: the payload hash and the SHA-256
        of the float32 vector in pgvector's binary form (the same exact
        comparison verify_batch makes). stored_digests computes the same
        lines from the stored rows.
        """
        self._batch(scope, items)
        lines = []
        for item in sorted(items, key=lambda i: i.chunk.vector_id):
            raw = struct.pack(
                f"!hh{len(item.vector)}f", len(item.vector), 0, *item.vector
            )
            lines.append(
                self._digest_line(
                    item.chunk.vector_id,
                    binding(item),
                    hashlib.sha256(raw).hexdigest(),
                )
            )
        return hashlib.sha256("".join(lines).encode()).hexdigest()

    async def stored_digests(self, scope, *, batches):
        """One query: the digest of each ID group's stored rows, as expected_digest.

        A group whose rows are all missing yields None. Missing, extra or
        changed rows change the digest; the caller then runs verify_batch to
        classify them.
        """
        ids, groups = [], []
        for number, batch in enumerate(batches):
            ids.extend(batch)
            groups.extend([number] * len(batch))
        if len(ids) != len(set(ids)) or len(ids) > self.MAX_DIGEST_IDS:
            raise ValueError("Digest request must name at most 8192 unique IDs")
        prefix = f"{scope.tenant_id}:{scope.document_id}:{scope.generation_id}:"
        if any(not i.startswith(prefix) for i in ids):
            raise ValueError("Digest request names an ID outside its scope")
        found = {}
        async with self.db.transaction(scope.tenant_id, lock=False) as (c, now):
            await self._ensure_target(c)
            rows = await c.execute(
                sa.text("""SELECT w.grp, encode(sha256(convert_to(string_agg(
                        v.vector_id || E'\t' || v.payload_hash || E'\t'
                        || encode(sha256(vector_send(v.embedding)), 'hex') || E'\n',
                        '' ORDER BY v.vector_id COLLATE "C"), 'UTF8')), 'hex')
                    FROM unnest(CAST(:ids AS text[]), CAST(:grps AS int[])) AS w(vector_id, grp)
                    JOIN rag_vectors v ON v.target=:target AND v.tenant_id=CAST(:tenant AS uuid)
                        AND v.document_id=CAST(:doc AS uuid) AND v.generation_id=CAST(:gen AS uuid)
                        AND v.vector_id=w.vector_id
                    GROUP BY w.grp"""),
                dict(
                    ids=ids,
                    grps=groups,
                    target=self.target_id,
                    tenant=str(scope.tenant_id),
                    doc=str(scope.document_id),
                    gen=str(scope.generation_id),
                ),
            )
            found = {grp: digest for grp, digest in rows}
        return tuple(found.get(n) for n in range(len(batches)))

    async def delete_generation(self, scope):
        """Delete only one exact target/tenant/document/generation namespace."""
        async with self.db.transaction(scope.tenant_id) as (c, now):
            await c.execute(
                sa.text(
                    "DELETE FROM rag_vectors WHERE target=:target AND tenant_id=CAST(:tenant AS uuid) AND document_id=CAST(:doc AS uuid) AND generation_id=CAST(:gen AS uuid)"
                ),
                dict(
                    target=self.target_id,
                    tenant=str(scope.tenant_id),
                    doc=str(scope.document_id),
                    gen=str(scope.generation_id),
                ),
            )

    async def search_hits(
        self, *, tenant_id, vector, k, document_ids=(), exact=False, ef_search=None
    ):
        """The k nearest vectors of this tenant, best first (IDs and scores only).

        Paths:
        - document_ids given, exact=True, or no index: exact search over the
          matching rows (found through the scope index, then sorted). Nothing
          can be missed.
        - otherwise the HNSW index with an iterative scan: pgvector keeps
          walking the graph until k rows pass the tenant filter (up to
          hnsw_max_scan_tuples), so other tenants cannot starve the result.
          If it still returns fewer than k, the tenant is searched exactly:
          either the tenant holds fewer than k vectors (cheap) or the scan
          limit was reached (logged, so the limit can be raised).

        Superseded builds may appear; the search pipeline keeps only active ones.
        ef_search overrides the configured value (evaluation and tuning).
        """
        if len(vector) != self.dimensions or not 1 <= k <= 1000:
            raise ValueError("Invalid search request")
        if len(document_ids) > 1000:
            raise ValueError("At most 1000 document_ids")
        if document_ids or exact or self.index is None:
            return await self._exact_hits(tenant_id, vector, k, document_ids)
        ef = max(k, ef_search or self.hnsw_ef_search)
        if not 1 <= ef <= 1000:
            raise ValueError("ef_search must be 1..1000")
        params = dict(vector=json.dumps(vector), target=self.target_id, tenant=str(tenant_id), k=k)
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            await self._ensure_target(c)
            # SET LOCAL takes no bind parameters; both values are validated integers.
            await c.execute(sa.text(f"SET LOCAL hnsw.ef_search = {int(ef)}"))
            await c.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            await c.execute(
                sa.text(f"SET LOCAL hnsw.max_scan_tuples = {int(self.hnsw_max_scan_tuples)}")
            )
            rows = (
                await c.execute(
                    sa.text(
                        f"""WITH nearest AS MATERIALIZED (
                            SELECT v.vector_id, v.document_id, v.generation_id,
                                {self._key} <=> {self._query} AS distance
                            FROM rag_vectors v
                            WHERE v.target = :target AND v.tenant_id = CAST(:tenant AS uuid)
                            ORDER BY {self._key} <=> {self._query} LIMIT :k)
                        SELECT * FROM nearest ORDER BY distance"""
                    ),
                    params,
                )
            ).mappings().all()
        hits = self._hits(rows)
        if len(hits) < k:
            exact = await self._exact_hits(tenant_id, vector, k, ())
            if len(exact) > len(hits):
                log_event(
                    log,
                    "vector_search_scan_limit_reached",
                    logging.WARNING,
                    returned=len(hits),
                    exact=len(exact),
                    hint="raise INGEST_HNSW_MAX_SCAN_TUPLES",
                )
            return exact
        return hits

    async def _exact_hits(self, tenant_id, vector, k, document_ids):
        """Exact nearest vectors: filter rows first, then sort every match."""
        params = dict(vector=json.dumps(vector), target=self.target_id, tenant=str(tenant_id), k=k)
        scope = ""
        if document_ids:
            scope = " AND v.document_id = ANY(CAST(:docs AS uuid[]))"
            params["docs"] = [str(d) for d in document_ids]
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            await self._ensure_target(c)
            rows = (
                await c.execute(
                    sa.text(
                        """SELECT v.vector_id, v.document_id, v.generation_id,
                            v.embedding <=> CAST(:vector AS vector) AS distance
                        FROM rag_vectors v
                        WHERE v.target = :target AND v.tenant_id = CAST(:tenant AS uuid)"""
                        + scope
                        # "+ 0" keeps the planner off the HNSW index: exact order.
                        + " ORDER BY (v.embedding <=> CAST(:vector AS vector)) + 0 LIMIT :k"
                    ),
                    params,
                )
            ).mappings().all()
        return self._hits(rows)

    @staticmethod
    def _hits(rows):
        return tuple(
            VectorHit(
                vector_id=r["vector_id"],
                document_id=r["document_id"],
                generation_id=r["generation_id"],
                score=1.0 - float(r["distance"]),
            )
            for r in rows
        )

    async def aclose(self):
        """Close the owned vector connection pool."""
        await self.db.aclose()
