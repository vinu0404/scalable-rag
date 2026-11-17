"""Search foundation: one copy of chunk text, live-only results, filters, collapse.

Real PostgreSQL (opt-in DSN) and boto3 against in-process S3.
"""

import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import httpx
import sqlalchemy as sa

from rag_ingest.api.app import create_app
from rag_ingest.domain.search import SearchFilters
from rag_ingest.runtime import runners
from rag_ingest.runtime import search as pipeline
from rag_ingest.runtime.search import search

import test_embedding_cache as segments
import test_postgres_integration as integration

DSN = integration.DSN


def text(word):
    return (f"The {word} report covers durable ingestion and recovery. " * 4).encode()


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class SearchFoundation(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown
    drive = segments.EmbeddingSegments.drive

    async def ingest(self, data, *, name="doc.txt", tags=None, metadata=None, allow=False):
        r = await self.engine.initiate_upload(
            self.tenant,
            filename=name,
            size_bytes=len(data),
            request_key=str(uuid4()),
            tags=tags,
            metadata=metadata,
            allow_duplicate=allow,
        )
        u = r["upload"]
        self.engine.objects.client.put_object(Bucket=u.bucket, Key=u.object_key, Body=data)
        await self.engine.complete_upload(self.tenant, u.upload_id)
        await self.engine.verify_upload(self.tenant, u.upload_id)
        await self.engine.metadata.admit_generations(tenant_id=self.tenant, max_active=16)
        return u

    async def published(self, *uploads):
        await self.drive()
        docs = []
        for u in uploads:
            d = await self.engine.metadata.get_document(
                tenant_id=self.tenant, document_id=u.document_id
            )
            self.assertEqual(d.active_generation_id, d.desired_generation_id)
            docs.append(d)
        return docs

    async def sql(self, statement, **params):
        async with self.engine.metadata.db.engine.begin() as c:
            result = await c.execute(sa.text(statement), {"t": self.tenant, **params})
            return result.all() if result.returns_rows else None

    async def find(self, query="durable ingestion", limit=10, **filters):
        f = SearchFilters.build(**filters) if filters else None
        return await search(self.engine, self.tenant, query, limit=limit, filters=f)

    # ---- storage: each thing stored once ---------------------------------------

    async def test_text_is_stored_once_and_vector_rows_hold_no_copy(self):
        (doc,) = await self.published(await self.ingest(text("alpha")))
        chunks = await self.sql(
            "SELECT vector_id, content, text_sha256 FROM search_chunks WHERE tenant_id=:t"
        )
        self.assertTrue(chunks)
        for _vector_id, content, digest in chunks:
            self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), digest)
        payloads = await self.sql(
            "SELECT payload, length(payload::text) FROM rag_vectors WHERE tenant_id=:t"
        )
        self.assertEqual(len(payloads), len(chunks))
        for payload, size in payloads:
            self.assertEqual(set(payload), {"vector_id", "document_id", "generation_id", "ordinal"})
            self.assertLess(size, 400, "no vector or text inside the payload")

    async def test_changed_chunk_text_blocks_publication(self):
        original = self.engine._index

        async def index_then_tamper(job, *args):
            result = await original(job, *args)
            await self.sql(
                "UPDATE search_chunks SET content = content || ' (edited)' WHERE tenant_id=:t"
                " AND ordinal = 0"
            )
            return result

        self.engine._index = index_then_tamper
        u = await self.ingest(text("beta"))
        await self.drive()
        d = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=u.document_id
        )
        self.assertIsNone(d.active_generation_id)
        (code,) = await self.sql(
            "SELECT failure_code FROM jobs WHERE tenant_id=:t AND stage='publish'"
        )
        self.assertEqual(code[0], "search_chunks_mismatch")

    # ---- results: live, visible, this tenant ---------------------------------------

    async def test_results_carry_file_name_pages_and_text(self):
        await self.published(await self.ingest(text("gamma"), name="gamma.txt"))
        (first, *_) = await self.find()
        self.assertEqual(first.filename, "gamma.txt")
        self.assertIn("gamma report", first.text)
        self.assertEqual(first.pages, (1,))
        self.assertEqual(await search(self.engine, uuid4(), "durable", limit=5), ())

    async def test_deletion_hides_at_once_and_cleanup_erases_text(self):
        u = await self.ingest(text("delta"))
        await self.published(u)
        await self.engine.metadata.tombstone_document(
            tenant_id=self.tenant, document_id=u.document_id, deletion_id=uuid4()
        )
        self.assertEqual(await self.find(), ())
        await self.sql(
            "UPDATE cleanup SET payload = jsonb_set(payload, '{after}',"
            " '\"2000-01-01T00:00:00+00:00\"') WHERE tenant_id = :t"
        )
        await runners.cleanup_once(self.engine)
        self.assertEqual(
            await self.sql("SELECT count(*) FROM search_chunks WHERE tenant_id=:t"), [(0,)]
        )

    async def test_only_the_active_build_is_returned_after_reprocessing(self):
        u = await self.ingest(text("epsilon"))
        (doc,) = await self.published(u)
        settings = self.engine.settings
        await self.engine.metadata.reprocess_document(
            tenant_id=self.tenant,
            document_id=u.document_id,
            request_id=uuid4(),
            policy=settings.core.execution_policy,
            max_generation_seconds=settings.core.max_generation_seconds,
            pipeline=settings.pipeline(),
        )
        (new,) = await self.published(u)
        self.assertNotEqual(new.active_generation_id, doc.active_generation_id)
        builds = await self.sql(
            "SELECT count(DISTINCT generation_id) FROM rag_vectors WHERE tenant_id=:t"
        )
        self.assertEqual(builds, [(2,)], "the old build's vectors still exist")
        results = await self.find()
        self.assertTrue(results)
        self.assertEqual({r.generation_id for r in results}, {new.active_generation_id})

    # ---- duplicates across files ------------------------------------------------------

    async def test_identical_passages_collapse_into_one_result(self):
        a = await self.ingest(text("zeta"), name="a.txt")
        b = await self.ingest(text("zeta"), name="b.txt", allow=True)
        await self.published(a, b)
        results = await self.find(limit=10)
        texts = [r.text for r in results]
        self.assertEqual(len(texts), len(set(texts)), "no passage twice")
        first = results[0]
        self.assertEqual(len(first.also_in), 1)
        self.assertEqual(
            {first.filename, first.also_in[0][1]}, {"a.txt", "b.txt"}
        )

    # ---- filters ------------------------------------------------------------------------

    async def labelled(self):
        hr = await self.ingest(text("eta"), name="hr.txt", tags=["hr", "policy"],
                               metadata={"dept": "hr", "year": 2026})
        fin = await self.ingest(text("theta"), name="fin.txt", tags=["finance"],
                                metadata={"dept": "finance", "year": 2025})
        await self.published(hr, fin)
        return hr, fin

    async def names(self, **filters):
        return {r.filename for r in await self.find(**filters)}

    async def test_filters_by_tags_metadata_documents_and_time(self):
        hr, fin = await self.labelled()
        self.assertEqual(await self.names(tags_any=["hr", "finance"]), {"hr.txt", "fin.txt"})
        self.assertEqual(await self.names(tags_all=["hr", "policy"]), {"hr.txt"})
        self.assertEqual(await self.names(tags_all=["hr", "finance"]), set())
        self.assertEqual(await self.names(metadata={"dept": "finance"}), {"fin.txt"})
        self.assertEqual(await self.names(metadata={"year": 2026}), {"hr.txt"})
        self.assertEqual(await self.names(document_ids=[fin.document_id]), {"fin.txt"})
        future = datetime.now(timezone.utc) + timedelta(days=1)
        self.assertEqual(await self.names(created_after=future), set())
        self.assertEqual(await self.names(created_before=future), {"hr.txt", "fin.txt"})

    async def test_broad_filters_take_the_filter_after_search_path(self):
        """With more matching documents than the exact path allows, results stay correct."""
        await self.labelled()
        with patch.object(pipeline, "EXACT_FILTER_DOCUMENTS", 0):
            self.assertEqual(await self.names(tags_any=["finance"]), {"fin.txt"})
            self.assertEqual(await self.names(metadata={"dept": "hr"}), {"hr.txt"})

    async def test_relabelling_takes_effect_immediately(self):
        hr, fin = await self.labelled()
        await self.engine.metadata.update_document_labels(
            tenant_id=self.tenant, document_id=fin.document_id, tags=["hr"]
        )
        self.assertEqual(await self.names(tags_any=["hr"]), {"hr.txt", "fin.txt"})
        doc = await self.engine.metadata.get_document(
            tenant_id=self.tenant, document_id=fin.document_id
        )
        self.assertEqual(doc.metadata_dict, {"dept": "finance", "year": 2025}, "kept")

    # ---- HTTP contract --------------------------------------------------------------------

    async def test_api_labels_filters_and_result_shape(self):
        app = create_app(self.engine, authenticate=lambda token: self.tenant)
        auth = {"Authorization": "Bearer x"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            data = text("iota")
            r = await c.post(
                "/v1/uploads",
                headers={**auth, "Idempotency-Key": "k1"},
                json={"filename": "iota.txt", "size_bytes": len(data),
                      "tags": ["hr"], "metadata": {"dept": "hr"}},
            )
            self.assertEqual(r.status_code, 201, r.text)
            u = r.json()
            bad = await c.post(
                "/v1/uploads",
                headers={**auth, "Idempotency-Key": "k2"},
                json={"filename": "x.txt", "size_bytes": 5, "tags": ["no!good"]},
            )
            self.assertEqual(bad.status_code, 400)
            replace_with_labels = await c.post(
                "/v1/uploads",
                headers={**auth, "Idempotency-Key": "k3"},
                json={"filename": "x.txt", "size_bytes": 5, "document_id": u["document_id"],
                      "tags": ["hr"]},
            )
            self.assertEqual(replace_with_labels.status_code, 400)
            self.engine.objects.client.put_object(
                Bucket=self.engine.settings.bucket,
                Key=f"tenants/{self.tenant}/uploads/{u['upload_id']}/source",
                Body=data,
            )
            await c.post(f"/v1/uploads/{u['upload_id']}/complete", headers=auth)
            await self.engine.verify_upload(self.tenant, uuid_of(u["upload_id"]))
            await self.engine.metadata.admit_generations(tenant_id=self.tenant, max_active=16)
            await self.drive()
            doc = (await c.get(f"/v1/documents/{u['document_id']}", headers=auth)).json()
            self.assertEqual((doc["tags"], doc["metadata"]), (["hr"], {"dept": "hr"}))
            patched = await c.patch(
                f"/v1/documents/{u['document_id']}", headers=auth,
                json={"metadata": {"dept": "legal"}},
            )
            self.assertEqual(patched.json()["metadata"], {"dept": "legal"})
            self.assertEqual(patched.json()["tags"], ["hr"])
            found = await c.post(
                "/v1/search", headers=auth,
                json={"query": "durable", "limit": 3, "filters": {"metadata": {"dept": "legal"}}},
            )
            self.assertEqual(found.status_code, 200, found.text)
            self.assertEqual((found.json()["mode"], found.json()["degraded"]), ("vector", []))
            (hit, *_) = found.json()["results"]
            self.assertEqual(
                set(hit),
                {"document_id", "generation_id", "vector_id", "filename", "text",
                 "score", "pages", "spans", "also_in", "ranks"},
            )
            none = await c.post(
                "/v1/search", headers=auth,
                json={"query": "durable", "filters": {"metadata": {"dept": "hr"}}},
            )
            self.assertEqual(none.json()["results"], [])
            naive = await c.post(
                "/v1/search", headers=auth,
                json={"query": "durable", "filters": {"created_after": "2026-01-01T00:00:00"}},
            )
            self.assertEqual(naive.status_code, 400, "times need a timezone")


def uuid_of(value):
    from uuid import UUID

    return UUID(value)


class Collapse(unittest.TestCase):
    def test_keeps_best_first_and_lists_other_places(self):
        from rag_ingest.domain.search import SearchResult

        def r(doc, name, score, text, page):
            return SearchResult(
                vector_id=f"{doc}:{page}", document_id=doc, generation_id=doc,
                filename=name, ordinal=0, text=text,
                spans=({"page_number": page, "start_char": 0, "end_char": 1},),
                score=score, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )

        a, b = uuid4(), uuid4()
        out = pipeline.collapse([r(a, "a", 0.9, "same", 1), r(b, "b", 0.8, "same", 2),
                                 r(a, "a", 0.7, "other", 3)])
        self.assertEqual([x.text for x in out], ["same", "other"])
        self.assertEqual(out[0].also_in, ((b, "b", (2,)),))
        self.assertEqual(json.loads(json.dumps(pipeline.result_json(out[0])))["also_in"][0]["pages"], [2])


if __name__ == "__main__":
    unittest.main()
