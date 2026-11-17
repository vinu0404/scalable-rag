"""Test HTTP authorization in-process without opening network sockets."""

import unittest
from types import SimpleNamespace
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import httpx
from rag_ingest.api.app import create_app
from rag_ingest.domain.documents import Document


class API(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.owner = uuid4()
        self.other = uuid4()
        self.docid = uuid4()
        self.document = Document(
            tenant_id=self.owner,
            document_id=self.docid,
            display_name="secret.txt",
            acl_ref="tenant",
            created_at=datetime.now(timezone.utc),
        )

        async def get_document(*, tenant_id, document_id):
            return (
                self.document
                if (tenant_id, document_id) == (self.owner, self.docid)
                else None
            )

        metadata = SimpleNamespace(
            get_document=get_document,
            now=AsyncMock(return_value=datetime.now(timezone.utc)),
        )
        settings = SimpleNamespace(
            api_tokens={
                "owner-token-long-enough-123456": self.owner,
                "other-token-long-enough-123456": self.other,
            }
        )
        self.engine = SimpleNamespace(
            metadata=metadata, settings=settings, aclose=AsyncMock()
        )
        self.app = create_app(self.engine)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)

    async def test_missing_token_denied(self):
        self.assertEqual(
            (await self.client.get("/v1/documents/" + str(self.docid))).status_code, 401
        )

    async def test_other_tenant_cannot_read_document(self):
        response = await self.client.get(
            "/v1/documents/" + str(self.docid),
            headers={"Authorization": "Bearer other-token-long-enough-123456"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("secret", response.text)

    async def test_owner_document_response_has_no_internal_credentials(self):
        response = await self.client.get(
            "/v1/documents/" + str(self.docid),
            headers={"Authorization": "Bearer owner-token-long-enough-123456"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], str(self.docid))
        self.assertEqual(response.json()["jobs"], [])

    async def test_caller_cannot_supply_tenant_in_upload_body(self):
        response = await self.client.post(
            "/v1/uploads",
            headers={
                "Authorization": "Bearer owner-token-long-enough-123456",
                "Idempotency-Key": "test",
            },
            json={"filename": "a.txt", "size_bytes": 5, "tenant_id": str(self.other)},
        )
        self.assertEqual(response.status_code, 422)
