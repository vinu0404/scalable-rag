"""The live-test demo page and grounded chat endpoint (model calls are faked)."""

import json
import unittest
from dataclasses import replace
from uuid import uuid4

import httpx

from rag_ingest.api.app import create_app

import test_embedding_cache as segments
import test_postgres_integration as integration

DSN = integration.DSN
TOKEN = "demo-token-long-enough-1234567890"


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class DemoChat(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await integration.PostgresIntegration.asyncSetUp(self)
        from live_test.demo import add_demo_routes

        self.engine.settings = replace(
            self.engine.settings, api_tokens={TOKEN: self.tenant}
        )
        self.prompts = []

        def model(request):
            body = json.loads(request.content)
            self.prompts.append(body["messages"][-1]["content"])
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "Leases protect work [1]."}}]}
            )

        app = create_app(self.engine, authenticate=lambda t: self.tenant)
        self.llm = add_demo_routes(app, self.engine, transport=httpx.MockTransport(model))
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.llm.aclose()
        await integration.PostgresIntegration.asyncTearDown(self)

    ingest = segments.EmbeddingSegments.ingest
    drive = segments.EmbeddingSegments.drive

    async def ask(self, question, token=TOKEN):
        return await self.client.post(
            "/demo/chat",
            headers={"Authorization": "Bearer " + token},
            json={"question": question},
        )

    async def test_page_is_served(self):
        r = await self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Document Chat", r.text)

    async def test_answer_comes_with_numbered_sources_from_live_documents(self):
        await self.ingest()
        await self.drive()
        r = await self.ask("What protects durable work?")
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["answer"], "Leases protect work [1].")
        self.assertTrue(data["sources"])
        first = data["sources"][0]
        self.assertEqual(first["n"], 1)
        self.assertEqual(first["filename"], "doc.txt")
        self.assertIn("[1] doc.txt", self.prompts[0], "the model sees numbered sources")
        self.assertIn(first["text"][:40], self.prompts[0])

    async def test_nothing_searchable_means_no_model_call(self):
        r = await self.ask("anything?")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["sources"], [])
        self.assertEqual(self.prompts, [])

    async def test_a_wrong_token_is_refused(self):
        self.assertEqual((await self.ask("x", token="nope-" + str(uuid4()))).status_code, 401)
        r = await self.client.post("/demo/chat", json={"question": "x"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
