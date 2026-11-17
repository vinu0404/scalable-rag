"""Keyword index: the built-in adapter, the pipeline with it, hybrid search, reindex.

PostgreSQL tests need the opt-in DSN; fusion and the OpenSearch example's
request shapes run without a database.
"""

import asyncio
import hashlib
import json
import logging
import unittest
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import httpx
import sqlalchemy as sa

from rag_ingest.adapters.postgres.keyword import PostgresKeywordIndex
from rag_ingest.contracts.errors import PermanentInputError, RetryableError
from rag_ingest.contracts.keyword import KeywordExpectation, KeywordHit
from rag_ingest.contracts.vectors import VectorReadiness, VectorScope
from rag_ingest.domain.chunks import Chunk, SourceSpan
from rag_ingest.domain.search import SearchFilters
from rag_ingest.runtime import search as pipeline
from rag_ingest.runtime.keyword_reindex import reindex_keyword
from rag_ingest.testing.keyword_conformance import KeywordIndexConformance

import test_embedding_cache as segments
import test_postgres_integration as integration

DSN = integration.DSN


def chunk_for(document):
    """The chunk row the built-in adapter reads, for one conformance document."""
    tenant, doc, gen, ordinal = document.vector_id.split(":")
    return Chunk(
        tenant_id=UUID(tenant),
        document_id=UUID(doc),
        source_version_id=uuid4(),
        generation_id=UUID(gen),
        ordinal=int(ordinal),
        text=document.text,
        spans=(SourceSpan(extraction_sha256="0" * 64, page_number=1, start_char=0, end_char=1),),
    )


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class PostgresKeywordConformance(KeywordIndexConformance, unittest.IsolatedAsyncioTestCase):
    """The built-in adapter passes the same kit any engine must pass."""

    async def asyncSetUp(self):
        await integration.PostgresIntegration.asyncSetUp(self)
        self.index = self.make_index()

    async def asyncTearDown(self):
        await integration.PostgresIntegration.asyncTearDown(self)

    def make_index(self):
        return PostgresKeywordIndex(self.engine.metadata.db)

    async def make_scope(self, tenant_id=None):
        """A real accepted build (chunk rows reference generations)."""
        tenant = tenant_id or uuid4()
        data = f"conformance build {uuid4()}".encode()
        r = await self.engine.initiate_upload(
            tenant, filename="k.txt", size_bytes=len(data), request_key=str(uuid4())
        )
        u = r["upload"]
        self.engine.objects.client.put_object(Bucket=u.bucket, Key=u.object_key, Body=data)
        await self.engine.complete_upload(tenant, u.upload_id)
        await self.engine.verify_upload(tenant, u.upload_id)
        return VectorScope(
            tenant_id=tenant,
            document_id=u.document_id,
            generation_id=uuid5(u.upload_id, "generation"),
        )

    async def prepare(self, scope, documents):
        await self.engine.metadata.put_search_chunks(
            tenant_id=scope.tenant_id, chunks=tuple(chunk_for(d) for d in documents)
        )


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class HybridPipeline(unittest.IsolatedAsyncioTestCase):
    drive = segments.EmbeddingSegments.drive

    async def asyncSetUp(self):
        await integration.PostgresIntegration.asyncSetUp(self)
        self.use_keyword(PostgresKeywordIndex(self.engine.metadata.db))

    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    def use_keyword(self, keyword, **settings):
        self.engine.keyword = keyword
        self.engine.settings = replace(self.engine.settings, **settings)

    async def ingest(self, data, name="doc.txt", tags=None):
        r = await self.engine.initiate_upload(
            self.tenant, filename=name, size_bytes=len(data), request_key=str(uuid4()), tags=tags
        )
        u = r["upload"]
        self.engine.objects.client.put_object(Bucket=u.bucket, Key=u.object_key, Body=data)
        await self.engine.complete_upload(self.tenant, u.upload_id)
        await self.engine.verify_upload(self.tenant, u.upload_id)
        await self.engine.metadata.admit_generations(tenant_id=self.tenant, max_active=16)
        return u

    async def sql(self, statement, **params):
        async with self.engine.metadata.db.engine.begin() as c:
            result = await c.execute(sa.text(statement), {"t": self.tenant, **params})
            return result.all() if result.returns_rows else None

    async def published(self, *uploads):
        await self.drive()
        for u in uploads:
            d = await self.engine.metadata.get_document(tenant_id=self.tenant, document_id=u.document_id)
            self.assertEqual(d.active_generation_id, d.desired_generation_id)

    PLAIN = b"General notes about warehouses, shipping schedules and invoices. " * 6
    CODED = b"The parcel carries tracking code KX5521QR for the regional depot. " * 6

    # ---- writing and verifying ---------------------------------------------------------

    async def test_published_builds_are_in_the_keyword_index(self):
        await self.published(await self.ingest(self.PLAIN))
        rows = await self.sql(
            "SELECT count(*), count(*) FILTER (WHERE keyword_target = :target) FROM search_chunks"
            " WHERE tenant_id=:t",
            target=self.engine.keyword.target_id,
        )
        total, indexed = rows[0]
        self.assertGreater(total, 0)
        self.assertEqual(indexed, total)

    async def test_missing_keyword_entries_hold_publication(self):
        original = self.engine._index

        async def index_then_lose_keywords(job, *args):
            result = await original(job, *args)
            await self.sql("UPDATE search_chunks SET search_tsv = NULL WHERE tenant_id=:t")
            return result

        self.engine._index = index_then_lose_keywords
        u = await self.ingest(self.PLAIN)
        await self.drive(rounds=4)
        d = await self.engine.metadata.get_document(tenant_id=self.tenant, document_id=u.document_id)
        self.assertIsNone(d.active_generation_id)
        status = await self.sql("SELECT status FROM jobs WHERE tenant_id=:t AND stage='publish'")
        self.assertIn(status[0][0], ("retry_wait", "queued", "failed"))

    # ---- hybrid search -------------------------------------------------------------------

    async def test_hybrid_finds_an_exact_code_and_reports_ranks(self):
        coded = await self.ingest(self.CODED, name="coded.txt")
        plain = await self.ingest(self.PLAIN, name="plain.txt")
        await self.published(coded, plain)
        outcome = await pipeline.run(self.engine, self.tenant, "KX5521QR", limit=3, mode="hybrid")
        self.assertEqual(outcome.mode, "hybrid")
        self.assertEqual(outcome.degraded, ())
        top = outcome.results[0]
        self.assertEqual(top.filename, "coded.txt")
        self.assertEqual(set(dict(top.ranks)), {"vector", "keyword"}, "found by both")
        self.assertTrue(
            all(r.filename != "plain.txt" or "keyword" not in dict(r.ranks) for r in outcome.results),
            "the plain file never matches the code",
        )
        keyword_only = await pipeline.run(self.engine, self.tenant, "KX5521QR", limit=3, mode="keyword")
        self.assertEqual({r.filename for r in keyword_only.results}, {"coded.txt"})

    async def test_a_question_matches_chunks_with_some_of_its_words(self):
        """Not every word has to appear (a question rarely matches all of them)."""
        coded = await self.ingest(self.CODED, name="coded.txt")
        await self.published(coded)
        outcome = await pipeline.run(
            self.engine, self.tenant,
            "Which tracking code does the parcel for the regional depot carry?",
            mode="keyword",
        )
        self.assertTrue(outcome.results)
        self.assertEqual({r.filename for r in outcome.results}, {"coded.txt"})

    async def test_hybrid_respects_filters_and_deletion(self):
        coded = await self.ingest(self.CODED, name="coded.txt", tags=["depot"])
        await self.published(coded)
        f = SearchFilters.build(tags_any=["other"])
        self.assertEqual(
            await pipeline.search(self.engine, self.tenant, "KX5521QR", filters=f, mode="hybrid"), ()
        )
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=coded.document_id, deletion_id=uuid4()
        )
        self.assertEqual(
            await pipeline.search(self.engine, self.tenant, "KX5521QR", mode="keyword"), ()
        )

    async def test_a_failing_keyword_engine_degrades_unless_strict(self):
        await self.published(await self.ingest(self.CODED))
        keyword = self.engine.keyword

        async def broken(**kwargs):
            raise RetryableError("keyword engine down")

        self.engine.keyword = SimpleNamespace(capabilities=keyword.capabilities, search=broken, aclose=keyword.aclose)
        with self.assertLogs("rag_ingest", logging.WARNING) as logs:
            outcome = await pipeline.run(self.engine, self.tenant, "KX5521QR", mode="hybrid")
        self.assertEqual(outcome.degraded, ("keyword",))
        self.assertTrue(outcome.results, "vector results still returned")
        self.assertTrue(any("search_retriever_failed" in m for m in logs.output))
        self.engine.settings = replace(self.engine.settings, search_strict=True)
        with self.assertRaises(RetryableError):
            await pipeline.run(self.engine, self.tenant, "KX5521QR", mode="hybrid")

    async def test_a_slow_keyword_engine_times_out_and_degrades(self):
        await self.published(await self.ingest(self.CODED))
        keyword = self.engine.keyword

        async def slow(**kwargs):
            await asyncio.sleep(5)

        self.engine.keyword = SimpleNamespace(capabilities=keyword.capabilities, search=slow, aclose=keyword.aclose)
        self.engine.settings = replace(self.engine.settings, search_keyword_timeout_seconds=1)
        outcome = await pipeline.run(self.engine, self.tenant, "KX5521QR", mode="hybrid")
        self.assertEqual(outcome.degraded, ("keyword",))

    async def test_a_bug_in_an_adapter_is_not_hidden_as_degraded(self):
        await self.published(await self.ingest(self.CODED))
        keyword = self.engine.keyword

        async def buggy(**kwargs):
            return 1 / 0

        self.engine.keyword = SimpleNamespace(capabilities=keyword.capabilities, search=buggy, aclose=keyword.aclose)
        with self.assertRaises(ZeroDivisionError):
            await pipeline.run(self.engine, self.tenant, "KX5521QR", mode="hybrid")

    async def test_keyword_modes_need_a_keyword_index(self):
        self.use_keyword(None)
        with self.assertRaisesRegex(ValueError, "not configured"):
            await pipeline.run(self.engine, self.tenant, "anything", mode="hybrid")

    # ---- enabling later: stale builds and reindexing ---------------------------------------

    async def test_builds_indexed_without_keywords_are_reindexed(self):
        self.use_keyword(None)
        u = await self.ingest(self.CODED)
        await self.published(u)
        index = PostgresKeywordIndex(self.engine.metadata.db)
        self.use_keyword(index)
        self.assertEqual(
            await pipeline.search(self.engine, self.tenant, "KX5521QR", mode="keyword"), ()
        )
        result = await reindex_keyword(self.engine)
        self.assertEqual(result["builds"], 1)
        self.assertGreater(result["chunks"], 0)
        found = await pipeline.search(self.engine, self.tenant, "KX5521QR", mode="keyword")
        self.assertEqual({r.document_id for r in found}, {u.document_id})
        again = await reindex_keyword(self.engine)
        self.assertEqual(again["chunks"], result["chunks"], "safe to run again")

    async def test_cleanup_deletes_the_build_from_the_keyword_index(self):
        u = await self.ingest(self.CODED)
        await self.published(u)
        deleted, keyword = [], self.engine.keyword
        original = keyword.delete_generation

        async def recording(scope):
            deleted.append(scope.generation_id)
            await original(scope)

        keyword.delete_generation = recording
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=u.document_id, deletion_id=uuid4()
        )
        await self.sql(
            "UPDATE cleanup SET payload = jsonb_set(payload, '{after}',"
            " '\"2000-01-01T00:00:00+00:00\"') WHERE tenant_id = :t"
        )
        from rag_ingest.runtime import runners

        await runners.cleanup_once(self.engine)
        self.assertEqual(len(deleted), 1)


# ---- fusion (no database) -------------------------------------------------------------------


def hit(name, score=1.0):
    return KeywordHit(vector_id=name, document_id=uuid4(), generation_id=uuid4(), score=score)


class Fusion(unittest.TestCase):
    def test_rrf_rewards_agreement_and_uses_ranks_only(self):
        a, b, c = hit("a", 0.9), hit("b", 0.8), hit("c", 50.0)
        fused = pipeline.fuse(
            {"vector": [a, b], "keyword": [c, b]},
            method="rrf",
            rrf_k=60,
            weights={"vector": 1.0, "keyword": 1.0},
        )
        self.assertEqual([h.vector_id for h, _, _ in fused][0], "b", "in both lists")
        self.assertEqual(dict(fused[0][2]), {"vector": 2, "keyword": 2})

    def test_weights_shift_the_balance(self):
        a, c = hit("a"), hit("c")
        fused = pipeline.fuse(
            {"vector": [a], "keyword": [c]},
            method="rrf",
            rrf_k=60,
            weights={"vector": 1.0, "keyword": 3.0},
        )
        self.assertEqual(fused[0][0].vector_id, "c")

    def test_weighted_mode_normalizes_each_ranking(self):
        fused = pipeline.fuse(
            {"vector": [hit("a", 0.9), hit("b", 0.1)], "keyword": [hit("b", 900.0), hit("a", 899.0)]},
            method="weighted",
            rrf_k=60,
            weights={"vector": 1.0, "keyword": 1.0},
        )
        self.assertEqual({h.vector_id: round(s, 3) for h, s, _ in fused}, {"a": 1.0, "b": 1.0})


# ---- the OpenSearch example's requests (no cluster) ---------------------------------------


class OpenSearchExample(unittest.IsolatedAsyncioTestCase):
    def adapter(self, handler):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "opensearch_keyword", Path(__file__).parents[1] / "examples/opensearch_keyword.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.OpenSearchKeywordIndex(
            "https://search.test:9200", transport=httpx.MockTransport(handler)
        )

    async def test_search_is_a_match_query_filtered_and_routed_by_tenant(self):
        seen = {}

        def handler(request):
            seen["path"], seen["params"], seen["body"] = (
                request.url.path,
                dict(request.url.params),
                json.loads(request.content),
            )
            return httpx.Response(200, json={"hits": {"hits": []}})

        index = self.adapter(handler)
        tenant = uuid4()
        await index.search(tenant_id=tenant, query='title:secret AND "x', k=5)
        await index.aclose()
        self.assertEqual(seen["params"]["routing"], str(tenant))
        query = seen["body"]["query"]["bool"]
        self.assertEqual(query["must"], [{"match": {"text": {"query": 'title:secret AND "x'}}}])
        self.assertIn({"term": {"tenant_id": str(tenant)}}, query["filter"])
        self.assertNotIn("query_string", json.dumps(seen["body"]))

    async def test_writes_are_create_only_and_a_changed_text_is_refused(self):
        scope = VectorScope(tenant_id=uuid4(), document_id=uuid4(), generation_id=uuid4())
        text = "chunk text"
        from rag_ingest.testing.keyword_conformance import document

        doc = document(scope, 0, text)

        def handler(request):
            if request.url.path.endswith("_bulk"):
                action = json.loads(request.content.decode().splitlines()[0])
                assert "create" in action
                return httpx.Response(200, json={"items": [{"create": {"status": 409}}]})
            return httpx.Response(
                200,
                json={"docs": [{"_id": doc.vector_id, "found": True,
                                "_source": {"tenant_id": str(scope.tenant_id), "text_sha256": "f" * 64}}]},
            )

        index = self.adapter(handler)
        with self.assertRaises(PermanentInputError):
            await index.index_batch(scope, items=(doc,))
        await index.aclose()

    async def test_stored_but_not_yet_searchable_is_not_queryable(self):
        scope = VectorScope(tenant_id=uuid4(), document_id=uuid4(), generation_id=uuid4())
        vid = f"{scope.tenant_id}:{scope.document_id}:{scope.generation_id}:0"
        digest = hashlib.sha256(b"x").hexdigest()

        def handler(request):
            if request.url.path.endswith("_mget"):
                return httpx.Response(200, json={"docs": [{"_id": vid, "found": True,
                    "_source": {"tenant_id": str(scope.tenant_id), "text_sha256": digest}}]})
            return httpx.Response(200, json={"hits": {"hits": []}})

        index = self.adapter(handler)
        (check,) = await index.verify_batch(
            scope, expected=(KeywordExpectation(vector_id=vid, text_sha256=digest),)
        )
        await index.aclose()
        self.assertEqual(check.readiness, VectorReadiness.NOT_QUERYABLE)

    async def test_deletion_is_scoped_to_one_tenant_and_build(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        index = self.adapter(handler)
        scope = VectorScope(tenant_id=uuid4(), document_id=uuid4(), generation_id=uuid4())
        await index.delete_generation(scope)
        await index.aclose()
        self.assertEqual(
            seen["body"]["query"]["bool"]["filter"],
            [{"term": {"tenant_id": str(scope.tenant_id)}},
             {"term": {"generation_id": str(scope.generation_id)}}],
        )


if __name__ == "__main__":
    unittest.main()
