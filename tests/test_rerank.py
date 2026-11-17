"""Reranking: the pipeline step, its fallbacks, the kit, and the example adapters."""

import asyncio
import importlib.util
import json
import logging
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import httpx

from rag_ingest.contracts.errors import RateLimitedError, RetryableError
from rag_ingest.contracts.rerank import RerankCandidate, RerankerCapabilities, RerankScore
from rag_ingest.domain.pipeline import ComponentSpec
from rag_ingest.runtime import search as pipeline
from rag_ingest.testing.rerank_conformance import RerankerConformance

import test_embedding_cache as segments
import test_postgres_integration as integration

DSN = integration.DSN
ROOT = Path(__file__).parents[1]


class WordOverlapReranker:
    """Test double: scores a passage by how many query words it contains."""

    def __init__(self, capabilities=None):
        self.calls = []
        self._capabilities = capabilities or RerankerCapabilities()

    @property
    def spec(self):
        return ComponentSpec(driver="word-overlap", implementation_version="1")

    @property
    def capabilities(self):
        return self._capabilities

    async def rerank(self, query, candidates):
        self.calls.append((query, candidates))
        words = set(query.lower().replace("?", "").split())
        return tuple(
            RerankScore(id=c.id, score=float(len(words & set(c.text.lower().replace(".", "").split()))))
            for c in candidates
        )

    async def aclose(self):
        pass


class WordOverlapConformance(RerankerConformance, unittest.IsolatedAsyncioTestCase):
    """The kit itself works (and the test double meets the contract)."""

    def make_reranker(self):
        return WordOverlapReranker()


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class RerankStep(unittest.IsolatedAsyncioTestCase):
    drive = segments.EmbeddingSegments.drive

    async def asyncSetUp(self):
        await integration.PostgresIntegration.asyncSetUp(self)
        self.reranker = WordOverlapReranker()
        self.engine.reranker = self.reranker
        self.engine.settings = replace(self.engine.settings, rerank_top_n=20)
        docs = {
            "fruit.txt": b"Bananas are a yellow fruit and grow in warm places. " * 4,
            "france.txt": b"Paris is the capital city of France and its largest city. " * 4,
            "market.txt": b"The stock market closed lower after a quiet trading day. " * 4,
        }
        for name, data in docs.items():
            r = await self.engine.initiate_upload(
                self.tenant, filename=name, size_bytes=len(data), request_key=str(uuid4())
            )
            u = r["upload"]
            self.engine.objects.client.put_object(Bucket=u.bucket, Key=u.object_key, Body=data)
            await self.engine.complete_upload(self.tenant, u.upload_id)
            await self.engine.verify_upload(self.tenant, u.upload_id)
        await self.engine.metadata.admit_generations(tenant_id=self.tenant, max_active=16)
        await self.drive()

    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    QUESTION = "what is the capital city of France"

    async def ask(self, **kw):
        return await pipeline.run(self.engine, self.tenant, self.QUESTION, limit=2, **kw)

    async def test_reranking_reorders_the_pool_and_reports_its_rank(self):
        outcome = await self.ask(rerank=True)
        self.assertEqual(outcome.degraded, ())
        top = outcome.results[0]
        self.assertEqual(top.filename, "france.txt")
        self.assertEqual(dict(top.ranks)["rerank"], 1)
        self.assertGreater(top.score, 1.0, "the reranker's score, not cosine similarity")
        ((query, candidates),) = self.reranker.calls
        self.assertGreater(len(candidates), 2, "a pool larger than the limit is reranked")

    async def test_off_means_no_call(self):
        await self.ask(rerank=False)
        self.assertEqual(self.reranker.calls, [])

    async def test_text_is_cut_to_the_rerankers_limits(self):
        self.engine.reranker = self.reranker = WordOverlapReranker(
            RerankerCapabilities(max_candidates=2, max_query_characters=10, max_passage_characters=20)
        )
        await self.ask(rerank=True)
        ((query, candidates),) = self.reranker.calls
        self.assertLessEqual(len(query), 10)
        self.assertLessEqual(len(candidates), 2)
        self.assertTrue(all(len(c.text) <= 20 for c in candidates))

    async def failing(self, rerank):
        self.reranker.rerank = rerank
        with self.assertLogs("rag_ingest", logging.WARNING):
            return await self.ask(rerank=True)

    async def test_failure_timeout_or_bad_reply_keeps_the_fused_order(self):
        plain = await self.ask(rerank=False)

        async def down(query, candidates):
            raise RateLimitedError("busy")

        async def slow(query, candidates):
            await asyncio.sleep(5)

        async def missing_one(query, candidates):
            return tuple(RerankScore(id=c.id, score=1.0) for c in candidates[1:])

        async def not_finite(query, candidates):
            return tuple(RerankScore(id=c.id, score=float("nan")) for c in candidates)

        async def foreign_id(query, candidates):
            return tuple(RerankScore(id="x" + c.id, score=1.0) for c in candidates)

        self.engine.settings = replace(self.engine.settings, rerank_timeout_ms=200)
        for broken in (down, slow, missing_one, not_finite, foreign_id):
            with self.subTest(broken=broken.__name__):
                outcome = await self.failing(broken)
                self.assertEqual(outcome.degraded, ("rerank",))
                self.assertEqual(
                    [r.vector_id for r in outcome.results],
                    [r.vector_id for r in plain.results],
                )

    async def test_strict_mode_fails_instead(self):
        async def down(query, candidates):
            raise RetryableError("busy")

        self.reranker.rerank = down
        self.engine.settings = replace(self.engine.settings, search_strict=True)
        with self.assertRaises(RetryableError):
            await self.ask(rerank=True)

    async def test_a_bug_in_the_reranker_is_not_hidden(self):
        async def buggy(query, candidates):
            return 1 / 0

        self.reranker.rerank = buggy
        with self.assertRaises(ZeroDivisionError):
            await self.ask(rerank=True)

    async def test_asking_for_reranking_without_a_reranker_is_refused(self):
        self.engine.reranker = None
        with self.assertRaisesRegex(ValueError, "No reranker"):
            await self.ask(rerank=True)


class Wiring(unittest.TestCase):
    def test_build_engine_refuses_rerank_without_a_reranker(self):
        from rag_ingest.config import Settings
        from rag_ingest.runtime.factory import build_engine
        from rag_ingest.runtime.settings import ServiceSettings

        settings = ServiceSettings(
            core=Settings(metadata_dsn="postgresql://u:p@localhost/db", s3_bucket="b",
                          sqs_queue_url="https://sqs.us-east-1.amazonaws.com/1/q"),
            vector_dsn="postgresql://u:p@localhost/db", bucket="b",
            embedding_driver="deterministic", search_rerank=True,
        )
        with self.assertRaisesRegex(ValueError, "reranker"):
            build_engine(settings, metadata=object(), objects=_Store(), queue=object(), vectors=object(),
                         embedder=object())


class BuildEngineOverrides(unittest.TestCase):
    """keyword= and reranker= reach the engine through build_engine (the documented way)."""

    def test_keyword_and_reranker_are_accepted(self):
        from rag_ingest.config import Settings
        from rag_ingest.runtime.factory import build_engine
        from rag_ingest.runtime.settings import ServiceSettings

        settings = ServiceSettings(
            core=Settings(metadata_dsn="postgresql://u:p@localhost/db", s3_bucket="b",
                          sqs_queue_url="https://sqs.us-east-1.amazonaws.com/1/q"),
            vector_dsn="postgresql://u:p@localhost/db", bucket="b",
            embedding_driver="deterministic", search_rerank=True, search_hybrid=True,
        )
        keyword, reranker = object(), WordOverlapReranker()
        engine = build_engine(settings, metadata=object(), objects=_Store(), queue=object(),
                              vectors=object(), embedder=object(), keyword=keyword, reranker=reranker)
        self.assertIs(engine.keyword, keyword)
        self.assertIs(engine.reranker, reranker)
        with self.assertRaisesRegex(ValueError, "Unknown adapter override: rerankr"):
            build_engine(settings, rerankr=reranker)


class _Store:
    """Just enough of a CleanupObjectStore for build_engine's checks."""

    async def delete_prefix_versions(self, bucket, prefix): ...
    async def delete_all_key_versions(self, ref): ...


# ---- example adapters (no live service) ---------------------------------------------------


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANDIDATES = (RerankCandidate(id="a", text="first"), RerankCandidate(id="b", text="second"))


class HttpRerankerExample(unittest.IsolatedAsyncioTestCase):
    async def call(self, api, handler):
        reranker = load("examples/http_reranker.py", "http_reranker").HttpReranker(
            "https://rerank.test", api=api, model="m", transport=httpx.MockTransport(handler)
        )
        try:
            return await reranker.rerank("query", CANDIDATES)
        finally:
            await reranker.aclose()

    async def test_cohere_shape(self):
        def handler(request):
            body = json.loads(request.content)
            assert request.url.path == "/v2/rerank" and body["documents"] == ["first", "second"]
            return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9},
                                                         {"index": 0, "relevance_score": 0.1}]})

        scores = await self.call("cohere", handler)
        self.assertEqual({s.id: s.score for s in scores}, {"a": 0.1, "b": 0.9})

    async def test_tei_shape_and_throttling(self):
        def handler(request):
            assert request.url.path == "/rerank" and json.loads(request.content)["texts"]
            return httpx.Response(200, json=[{"index": 0, "score": 2.0}, {"index": 1, "score": 1.0}])

        self.assertEqual({s.id for s in await self.call("tei", handler)}, {"a", "b"})
        with self.assertRaises(RateLimitedError):
            await self.call("tei", lambda request: httpx.Response(429))


class OpenAIRerankerLive(unittest.IsolatedAsyncioTestCase):
    async def call(self, reply):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

        reranker = load("live_test/openai_reranker.py", "openai_reranker").OpenAIReranker(
            api_key="test", transport=httpx.MockTransport(handler)
        )
        try:
            return await reranker.rerank("query", CANDIDATES), seen
        finally:
            await reranker.aclose()

    async def test_labels_not_ids_go_to_the_model(self):
        scores, seen = await self.call({"scores": {"1": 3, "2": 9}})
        self.assertEqual({s.id: s.score for s in scores}, {"a": 3.0, "b": 9.0})
        prompt = seen["body"]["messages"][1]["content"]
        self.assertIn("[1] first", prompt)
        self.assertNotIn('"a"', prompt)

    async def test_a_reply_missing_a_score_is_a_failed_call(self):
        with self.assertRaises(RetryableError):
            await self.call({"scores": {"1": 3}})


if __name__ == "__main__":
    unittest.main()
