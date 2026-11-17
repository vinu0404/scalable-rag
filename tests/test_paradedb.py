"""The ParadeDB (pg_search BM25) example keyword index: the conformance kit,
the whole pipeline with it, and provisioning.

Needs a disposable *_test database on a PostgreSQL with pg_search and
pgvector (a ParadeDB image): INGEST_TEST_PARADEDB_DSN.
"""

import importlib.util
import logging
import os
import unittest
from pathlib import Path
from unittest import mock

import sqlalchemy as sa

import test_keyword as keyword_tests
import test_postgres_integration as integration

_spec = importlib.util.spec_from_file_location(
    "paradedb_keyword", Path(__file__).parents[1] / "examples" / "paradedb_keyword.py"
)
paradedb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paradedb)
INDEX_NAME, ParadeDBKeywordIndex = paradedb.INDEX_NAME, paradedb.ParadeDBKeywordIndex

DSN = os.environ.get("INGEST_TEST_PARADEDB_DSN")
if DSN:
    assert DSN.rsplit("/", 1)[-1].split("?")[0].endswith("_test"), "use a disposable *_test DB"


async def set_up(case):
    """The shared integration setup, on the ParadeDB database, plus the BM25 index."""
    with mock.patch.object(integration, "DSN", DSN):
        await integration.PostgresIntegration.asyncSetUp(case)
    case.paradedb = ParadeDBKeywordIndex(case.engine.metadata.db)
    await case.paradedb.provision()


async def tear_down(case):
    with mock.patch.object(integration, "DSN", DSN):
        await integration.PostgresIntegration.asyncTearDown(case)


@unittest.skipUnless(DSN, "Set INGEST_TEST_PARADEDB_DSN to a disposable *_test DB with pg_search")
class ParadeDBConformance(keyword_tests.PostgresKeywordConformance):
    async def asyncSetUp(self):
        await set_up(self)
        self.index = self.make_index()

    async def asyncTearDown(self):
        await tear_down(self)

    def make_index(self):
        return self.paradedb

    async def test_rarer_words_rank_higher(self):
        """BM25 weighs a word by its rarity; plain full-text ranking does not."""
        texts = ["common words here"] * 8 + ["common words and a zygomorphic flower"]
        scope, docs = await self.indexed(texts)
        hits = await self.find(scope, "common zygomorphic")
        self.assertEqual(hits[0].vector_id, docs[-1].vector_id)

    async def test_search_uses_a_top_k_scan(self):
        """Only the best k rows are read: the reason to use BM25 at scale."""
        scope, _ = await self.indexed(["alpha beta", "beta gamma"])
        async with self.engine.metadata.db.engine.connect() as c:
            plan = "\n".join(
                r[0]
                for r in await c.execute(
                    sa.text(
                        "EXPLAIN SELECT vector_id FROM search_chunks WHERE content ||| 'beta'"
                        " AND tenant_id = :t AND keyword_target = :target"
                        ' ORDER BY pdb.score(vector_id) DESC, vector_id COLLATE "C" LIMIT 5'
                    ),
                    dict(t=scope.tenant_id, target=self.index.target_id),
                )
            )
        self.assertIn("TopK", plan)
        self.assertNotIn("heap_filter", plan, "every filter runs inside the index")


@unittest.skipUnless(DSN, "Set INGEST_TEST_PARADEDB_DSN to a disposable *_test DB with pg_search")
class ParadeDBHybridPipeline(keyword_tests.HybridPipeline):
    """Every hybrid-pipeline test, with BM25 as the keyword index."""

    async def asyncSetUp(self):
        await set_up(self)
        self.use_keyword(self.paradedb)

    async def asyncTearDown(self):
        await tear_down(self)


@unittest.skipUnless(DSN, "Set INGEST_TEST_PARADEDB_DSN to a disposable *_test DB with pg_search")
class Provisioning(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = set_up
    asyncTearDown = tear_down

    async def sql(self, statement):
        async with self.engine.metadata.db.engine.begin() as c:
            result = await c.execute(sa.text(statement))
            return result.all() if result.returns_rows else None

    async def test_provisioning_is_repeatable_and_keeps_the_pool_limits(self):
        await self.paradedb.provision()
        for _ in range(6):
            self.assertEqual(await self.sql("SHOW statement_timeout"), [("15s",)])

    async def test_an_invalid_index_is_rebuilt(self):
        await self.sql(
            f"UPDATE pg_index SET indisvalid = false WHERE indexrelid = '{INDEX_NAME}'::regclass"
        )
        await self.paradedb.provision()
        self.assertEqual(
            await self.sql(f"SELECT indisvalid FROM pg_index WHERE indexrelid = '{INDEX_NAME}'::regclass"),
            [(True,)],
        )

    async def test_a_different_language_is_reported_not_applied(self):
        other = ParadeDBKeywordIndex(self.engine.metadata.db, language="french")
        with self.assertLogs(paradedb.log, logging.WARNING) as logs:
            await other.provision()
        self.assertTrue(any("keyword_index_settings_differ" in m for m in logs.output))

    async def test_doctor_refuses_a_missing_index(self):
        await self.sql(f"DROP INDEX {INDEX_NAME}")
        with self.assertRaisesRegex(ValueError, "BM25 index"):
            await self.paradedb.doctor()

    def test_unsupported_language_is_refused(self):
        with self.assertRaisesRegex(ValueError, "language"):
            ParadeDBKeywordIndex(object(), language="klingon")


if __name__ == "__main__":
    unittest.main()
