"""HNSW index: built and repaired correctly, used by search, never starving a tenant.

Real PostgreSQL with pgvector (opt-in DSN). Synthetic vectors are inserted
directly so the table is large enough for the planner to use the index.
"""

import logging
import math
import unittest
from unittest import mock
from uuid import uuid4

import sqlalchemy as sa

from rag_ingest.adapters.postgres.connection import Database
from rag_ingest.adapters.postgres.vectors import INDEX_NAME, PgVectorStore
from rag_ingest.runtime.search_eval import evaluate, format_report

import test_postgres_integration as integration

DSN = integration.DSN


class IndexChoice(unittest.TestCase):
    def test_index_type_follows_pgvector_dimension_limits(self):
        self.assertEqual(PgVectorStore(None, dimensions=1536).index, ("embedding", "vector_cosine_ops"))
        self.assertEqual(PgVectorStore(None, dimensions=3072).index[1], "halfvec_cosine_ops")
        self.assertIsNone(PgVectorStore(None, dimensions=5000).index)
        self.assertIsNone(PgVectorStore(None, dimensions=8, index="none").index)

    def test_parameters_are_bounded(self):
        for kwargs in (
            dict(hnsw_m=1),
            dict(hnsw_m=32, hnsw_ef_construction=40),
            dict(hnsw_ef_search=0),
            dict(hnsw_max_scan_tuples=10),
            dict(index="ivfflat"),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PgVectorStore(None, dimensions=8, **kwargs)


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class HnswIndex(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    @property
    def store(self):
        return self.engine.vectors

    async def sql(self, statement, **params):
        async with self.store.db.engine.begin() as c:
            result = await c.execute(sa.text(statement), params)
            return result.all() if result.returns_rows else None

    async def add_vectors(self, tenant, count):
        """count random 8-dimension vectors for one tenant (one document)."""
        doc, gen = uuid4(), uuid4()
        await self.sql(
            """INSERT INTO rag_vectors
                (target, tenant_id, document_id, generation_id, vector_id, payload_hash, payload, embedding)
            SELECT :target, CAST(:tenant AS uuid), CAST(:doc AS uuid), CAST(:gen AS uuid),
                :prefix || g, 'h', '{}'::jsonb,
                (SELECT array_agg(random()::real)::vector FROM generate_series(1, 8) WHERE g >= 0)
            FROM generate_series(0, :n - 1) g""",
            target=self.store.target_id,
            tenant=str(tenant),
            doc=str(doc),
            gen=str(gen),
            prefix=f"{tenant}:{doc}:{gen}:",
            n=count,
        )
        await self.sql("ANALYZE rag_vectors")
        return doc

    async def query_vector(self, tenant):
        (row,) = await self.sql(
            "SELECT embedding::text FROM rag_vectors WHERE tenant_id = CAST(:t AS uuid) LIMIT 1",
            t=str(tenant),
        )
        return tuple(float(x) for x in row[0].strip("[]").split(","))

    # ---- building the index -------------------------------------------------------

    async def test_index_is_built_valid_with_the_configured_parameters(self):
        rows = await self.sql(
            "SELECT i.indisvalid, cl.reloptions, am.amname FROM pg_class cl"
            " JOIN pg_index i ON i.indexrelid = cl.oid JOIN pg_am am ON am.oid = cl.relam"
            " WHERE cl.relname = :n",
            n=INDEX_NAME,
        )
        self.assertEqual(rows, [(True, ["m=16", "ef_construction=64"], "hnsw")])

    async def test_an_invalid_index_left_by_a_failed_build_is_rebuilt(self):
        await self.sql(
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = CAST(:n AS regclass)",
            n=INDEX_NAME,
        )
        await self.store.provision()
        rows = await self.sql(
            "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = CAST(:n AS regclass)",
            n=INDEX_NAME,
        )
        self.assertEqual(rows, [(True,)])

    async def test_a_rebuild_is_not_cut_off_by_the_pool_statement_timeout(self):
        """Building over existing rows takes minutes at scale (bench/search_scale.py)."""
        await self.add_vectors(uuid4(), 200)
        await self.sql(f"DROP INDEX {INDEX_NAME}")
        with mock.patch.object(PgVectorStore, "_index_state", autospec=True) as state:
            seen = {}

            async def record(store, c):
                seen["timeout"] = (await c.execute(sa.text("SHOW statement_timeout"))).scalar()
                return (False, False, None)

            state.side_effect = record
            await self.store._provision_index()
        self.assertEqual(seen["timeout"], "0", "the build connection has no statement limit")
        for _ in range(8):  # every pooled connection still has the 15 s limit
            self.assertEqual(await self.sql("SHOW statement_timeout"), [("15s",)])
        rows = await self.sql(
            "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = CAST(:n AS regclass)",
            n=INDEX_NAME,
        )
        self.assertEqual(rows, [(True,)])

    async def test_changed_build_parameters_are_reported_not_silently_applied(self):
        other = PgVectorStore(
            Database(DSN), dimensions=8, hnsw_m=8, hnsw_ef_construction=32
        )
        try:
            with self.assertLogs("rag_ingest", logging.WARNING) as logs:
                await other.provision()
        finally:
            await other.aclose()
        self.assertTrue(any("vector_index_settings_differ" in m for m in logs.output))

    # ---- searching ---------------------------------------------------------------------

    async def test_tenant_searches_use_the_index(self):
        big = uuid4()
        await self.add_vectors(big, 20_000)
        vector = await self.query_vector(big)
        async with self.store.db.engine.begin() as c:
            await c.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            plan = "\n".join(
                r[0]
                for r in await c.execute(
                    sa.text(
                        "EXPLAIN SELECT v.vector_id FROM rag_vectors v WHERE v.target = :t"
                        " AND v.tenant_id = CAST(:tenant AS uuid)"
                        " ORDER BY v.embedding <=> CAST(:v AS vector) LIMIT 10"
                    ),
                    dict(t=self.store.target_id, tenant=str(big), v=str(list(vector))),
                )
            )
        self.assertIn(INDEX_NAME, plan)

    async def test_a_small_tenant_beside_a_big_one_gets_complete_results(self):
        """The over-filtering case: walking the graph mostly meets other tenants."""
        big, small = uuid4(), uuid4()
        await self.add_vectors(big, 20_000)
        await self.add_vectors(small, 6)
        vector = await self.query_vector(small)
        hits = await self.store.search_hits(tenant_id=small, vector=vector, k=10)
        exact = await self.store.search_hits(tenant_id=small, vector=vector, k=10, exact=True)
        self.assertEqual(len(hits), 6, "every vector the tenant has")
        self.assertEqual([h.vector_id for h in hits], [h.vector_id for h in exact])

    async def test_index_results_are_ordered_and_scored_like_exact_search(self):
        big = uuid4()
        await self.add_vectors(big, 5_000)
        vector = await self.query_vector(big)
        hits = await self.store.search_hits(tenant_id=big, vector=vector, k=10)
        self.assertEqual(len(hits), 10)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(math.isclose(hits[0].score, 1.0, abs_tol=1e-6), "the query vector itself")

    async def test_document_scope_is_exact(self):
        big = uuid4()
        doc = await self.add_vectors(big, 3_000)
        await self.add_vectors(big, 3_000)
        vector = await self.query_vector(big)
        scoped = await self.store.search_hits(
            tenant_id=big, vector=vector, k=20, document_ids=(doc,)
        )
        self.assertEqual({h.document_id for h in scoped}, {doc})
        rows = await self.sql(
            "SELECT vector_id FROM rag_vectors WHERE document_id = CAST(:d AS uuid)"
            " ORDER BY (embedding <=> CAST(:v AS vector)) + 0 LIMIT 20",
            d=str(doc),
            v=str(list(vector)),
        )
        self.assertEqual([h.vector_id for h in scoped], [r[0] for r in rows])

    async def test_recall_against_exact_search(self):
        big = uuid4()
        await self.add_vectors(big, 10_000)
        report = await evaluate(self.store, samples=30, k=10, ef_values=(40, 100))
        self.assertEqual(report["samples"], 30)
        self.assertGreaterEqual(report["index"][100]["recall"], 0.95, report)
        self.assertGreaterEqual(report["index"][100]["recall"], report["index"][40]["recall"] - 0.01)
        self.assertEqual(report["index"][100]["missed"], 0)
        self.assertIn("missed entirely", format_report(report))


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class NoIndex(unittest.IsolatedAsyncioTestCase):
    async def test_index_none_builds_nothing_and_searches_exactly(self):
        store = PgVectorStore(Database(DSN), dimensions=8, index="none")
        try:
            async with store.db.engine.begin() as c:
                await c.execute(sa.text("DROP TABLE IF EXISTS rag_vectors"))
            with self.assertLogs("rag_ingest", logging.WARNING) as logs:
                await store.provision()
            self.assertTrue(any("vector_index_disabled" in m for m in logs.output))
            async with store.db.engine.begin() as c:
                found = (
                    await c.execute(
                        sa.text("SELECT count(*) FROM pg_class WHERE relname = :n"),
                        dict(n=INDEX_NAME),
                    )
                ).scalar_one()
            self.assertEqual(found, 0)
            hits = await store.search_hits(tenant_id=uuid4(), vector=(1.0,) * 8, k=5)
            self.assertEqual(hits, ())
        finally:
            await store.aclose()


if __name__ == "__main__":
    unittest.main()
