"""Conformance tests for any KeywordIndex implementation.

Usage:

    class MyEngineConformance(KeywordIndexConformance, unittest.IsolatedAsyncioTestCase):
        def make_index(self):
            return MyKeywordIndex(...)

Optional hooks:
- make_scope(tenant_id=None): a scope the adapter can write to, in the given
  tenant or a new one (default: random IDs). The built-in Postgres adapter
  needs real builds, so its test creates them.
- prepare(scope, documents): store anything the adapter reads besides its own
  index before index_batch (the built-in adapter reads the chunk table).
- settle(): wait until new entries are searchable (engines with a refresh).
"""

import hashlib
from uuid import uuid4

from rag_ingest.contracts.errors import PermanentInputError
from rag_ingest.contracts.keyword import (
    KeywordCapabilities,
    KeywordDocument,
    KeywordExpectation,
)
from rag_ingest.contracts.vectors import VectorReadiness, VectorScope

HOSTILE_QUERIES = (
    "foo & bar | !baz",
    "pre:*",
    "(unbalanced",
    "'single quote",
    '"open phrase',
    "a <-> b",
    "back\\slash",
    "100% _wild_",
    "title:secret AND tenant_id:*",
    "*",
)


def document(scope, ordinal, text):
    return KeywordDocument(
        vector_id=f"{scope.tenant_id}:{scope.document_id}:{scope.generation_id}:{ordinal}",
        document_id=scope.document_id,
        generation_id=scope.generation_id,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


class KeywordIndexConformance:
    """Mix into unittest.IsolatedAsyncioTestCase and implement make_index()."""

    def make_index(self):
        raise NotImplementedError

    async def make_scope(self, tenant_id=None):
        return VectorScope(
            tenant_id=tenant_id or uuid4(), document_id=uuid4(), generation_id=uuid4()
        )

    async def prepare(self, scope, documents):
        """Store what the adapter reads besides its own index (default: nothing)."""

    async def settle(self):
        """Wait until indexed entries are searchable (default: immediately)."""

    async def asyncSetUp(self):
        await super().asyncSetUp() if hasattr(super(), "asyncSetUp") else None
        self.index = self.make_index()

    async def asyncTearDown(self):
        await self.index.aclose()
        if hasattr(super(), "asyncTearDown"):
            await super().asyncTearDown()

    async def indexed(self, texts, scope=None):
        scope = scope or await self.make_scope()
        docs = tuple(document(scope, n, t) for n, t in enumerate(texts))
        await self.prepare(scope, docs)
        await self.index.index_batch(scope, items=docs)
        await self.settle()
        return scope, docs

    async def find(self, scope, query, k=10, **kw):
        return await self.index.search(tenant_id=scope.tenant_id, query=query, k=k, **kw)

    # ---- the contract -----------------------------------------------------------------

    def test_declares_valid_capabilities_and_target(self):
        self.assertIsInstance(self.index.capabilities, KeywordCapabilities)
        self.assertTrue(isinstance(self.index.target_id, str) and self.index.target_id)

    async def test_finds_a_distinctive_word(self):
        scope, docs = await self.indexed(
            ["The zephyrine turbine spins quickly.", "Nothing relevant lives here."]
        )
        hits = await self.find(scope, "zephyrine")
        self.assertEqual([h.vector_id for h in hits], [docs[0].vector_id])
        self.assertEqual(
            (hits[0].document_id, hits[0].generation_id),
            (scope.document_id, scope.generation_id),
        )

    async def test_never_returns_another_tenants_entries(self):
        mine, _ = await self.indexed(["The quillwort report is ready."])
        theirs, _ = await self.indexed(["The quillwort report is ready."])
        for scope in (mine, theirs):
            hits = await self.find(scope, "quillwort")
            self.assertTrue(hits)
            self.assertTrue(all(h.vector_id.startswith(str(scope.tenant_id)) for h in hits))

    async def test_indexing_again_is_harmless(self):
        scope, docs = await self.indexed(["Marigold ferry arrives Thursdays."])
        await self.index.index_batch(scope, items=docs)
        await self.settle()
        self.assertEqual(len(await self.find(scope, "marigold")), 1)

    async def test_different_text_under_an_existing_id_is_refused(self):
        scope, docs = await self.indexed(["Original wording of the chunk."])
        changed = document(scope, 0, "Changed wording of the chunk.")
        with self.assertRaises(PermanentInputError):
            await self.index.index_batch(scope, items=(changed,))

    async def test_query_syntax_is_treated_as_plain_text(self):
        scope, _ = await self.indexed(["Ordinary text about foo and bar."])
        for query in HOSTILE_QUERIES:
            with self.subTest(query=query):
                hits = await self.find(scope, query)
                self.assertIsInstance(hits, tuple)

    async def test_a_query_without_searchable_words_returns_nothing(self):
        scope, _ = await self.indexed(["The and of to in."])
        self.assertEqual(await self.find(scope, "the and of"), ())

    async def test_verification_reports_each_state(self):
        scope, docs = await self.indexed(["Verification sample text."])
        unknown = document(scope, 99, "never indexed")
        checks = await self.index.verify_batch(
            scope,
            expected=(
                KeywordExpectation(vector_id=docs[0].vector_id, text_sha256=docs[0].text_sha256),
                KeywordExpectation(vector_id=unknown.vector_id, text_sha256=unknown.text_sha256),
            ),
        )
        self.assertEqual(
            [c.readiness for c in checks], [VectorReadiness.READY, VectorReadiness.MISSING]
        )
        wrong = await self.index.verify_batch(
            scope,
            expected=(KeywordExpectation(vector_id=docs[0].vector_id, text_sha256="f" * 64),),
        )
        self.assertEqual(wrong[0].readiness, VectorReadiness.MISMATCH)

    async def test_deleting_a_build_removes_only_that_build(self):
        scope, _ = await self.indexed(["Saxifrage notes for deletion."])
        other, _ = await self.indexed(
            ["Saxifrage notes kept."], scope=await self.make_scope(scope.tenant_id)
        )
        await self.index.delete_generation(scope)
        await self.index.delete_generation(scope)  # repeating is safe
        await self.settle()
        hits = await self.find(scope, "saxifrage")
        self.assertEqual({h.generation_id for h in hits}, {other.generation_id})

    async def test_document_filter_when_supported(self):
        if "document_ids" not in self.index.capabilities.filters:
            self.skipTest("engine does not filter by document")
        scope, _ = await self.indexed(["Heliotrope appears here."])
        hits = await self.find(scope, "heliotrope", document_ids=(uuid4(),))
        self.assertEqual(hits, ())
        hits = await self.find(scope, "heliotrope", document_ids=(scope.document_id,))
        self.assertEqual(len(hits), 1)
