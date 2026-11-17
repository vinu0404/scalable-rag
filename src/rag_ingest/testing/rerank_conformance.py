"""Conformance tests for any Reranker implementation.

Usage:

    class MyRerankerConformance(RerankerConformance, unittest.IsolatedAsyncioTestCase):
        def make_reranker(self):
            return MyReranker(...)

The relevance test uses an obvious case; a model that fails it is not
useful as a reranker. Tests call the real reranker, so a paid API is billed
for a handful of short requests.
"""

import math

from rag_ingest.contracts.rerank import RerankCandidate, RerankerCapabilities
from rag_ingest.domain.pipeline import ComponentSpec

PASSAGES = (
    "Bananas are a yellow fruit rich in potassium.",
    "Paris is the capital and largest city of France.",
    "The stock market closed lower on Tuesday.",
)


class RerankerConformance:
    """Mix into unittest.IsolatedAsyncioTestCase and implement make_reranker()."""

    def make_reranker(self):
        raise NotImplementedError

    async def asyncSetUp(self):
        self.reranker = self.make_reranker()

    async def asyncTearDown(self):
        await self.reranker.aclose()

    def candidates(self, texts):
        return tuple(RerankCandidate(id=f"c{n}", text=t) for n, t in enumerate(texts))

    def assert_one_finite_score_each(self, candidates, scores):
        self.assertEqual(len(scores), len(candidates))
        self.assertEqual({s.id for s in scores}, {c.id for c in candidates})
        for s in scores:
            self.assertIsInstance(s.score, (int, float))
            self.assertTrue(math.isfinite(s.score))

    def test_declares_identity_and_limits(self):
        self.assertIsInstance(self.reranker.spec, ComponentSpec)
        self.assertIsInstance(self.reranker.capabilities, RerankerCapabilities)

    async def test_scores_every_candidate_exactly_once(self):
        candidates = self.candidates(PASSAGES)
        scores = await self.reranker.rerank("What is the capital of France?", candidates)
        self.assert_one_finite_score_each(candidates, scores)

    async def test_the_obviously_relevant_passage_scores_highest(self):
        candidates = self.candidates(PASSAGES)
        scores = await self.reranker.rerank("What is the capital of France?", candidates)
        best = max(scores, key=lambda s: s.score)
        self.assertEqual(best.id, "c1")

    async def test_a_single_candidate(self):
        candidates = self.candidates(["Only one passage here."])
        self.assert_one_finite_score_each(
            candidates, await self.reranker.rerank("one passage", candidates)
        )

    async def test_text_up_to_the_declared_limits_is_accepted(self):
        caps = self.reranker.capabilities
        long_text = ("Résumé naïve café — 東京 " * 1000)[: caps.max_passage_characters]
        query = ("long query " * 1000)[: caps.max_query_characters]
        candidates = self.candidates([long_text, "short passage"])
        self.assert_one_finite_score_each(candidates, await self.reranker.rerank(query, candidates))

    async def test_as_many_candidates_as_declared(self):
        n = min(self.reranker.capabilities.max_candidates, 50)
        candidates = self.candidates([f"Passage number {i} about topic {i % 7}." for i in range(n)])
        self.assert_one_finite_score_each(
            candidates, await self.reranker.rerank("topic 3", candidates)
        )
