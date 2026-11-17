"""Per-tenant API rate limits, opt-in API docs, and the database TLS check."""

import hashlib
import unittest
from dataclasses import replace
from uuid import uuid4

import httpx
import sqlalchemy as sa

from rag_ingest.adapters.postgres.connection import tls_required
from rag_ingest.api.app import create_app

import test_postgres_integration as integration
from test_limits_and_settings import settings_from

DSN = integration.DSN
DATA = b"Leases protect durable work. Outbox notifications survive crashes. " * 5


class DatabaseTls(unittest.TestCase):
    def test_remote_connections_must_insist_on_tls(self):
        url = sa.engine.make_url
        self.assertFalse(tls_required(url("postgresql://u:p@db.example.com/x")))
        self.assertFalse(
            tls_required(url("postgresql://u:p@db.example.com/x?sslmode=prefer"))
        )
        for mode in ("require", "verify-ca", "verify-full"):
            self.assertTrue(
                tls_required(url(f"postgresql://u:p@db.example.com/x?sslmode={mode}"))
            )

    def test_local_hosts_and_sockets_are_exempt(self):
        url = sa.engine.make_url
        for dsn in (
            "postgresql://u:p@localhost/x",
            "postgresql://u:p@127.0.0.1/x",
            "postgresql://u:p@/x?host=/var/run/postgresql",
        ):
            self.assertTrue(tls_required(url(dsn)), dsn)


class Settings(unittest.TestCase):
    def test_defaults_and_bounds(self):
        s = settings_from()
        self.assertEqual(
            (s.api_uploads_per_minute, s.api_searches_per_minute, s.api_docs),
            (600, 120, False),
        )
        self.assertEqual(settings_from(INGEST_API_UPLOADS_PER_MINUTE="0").api_uploads_per_minute, 0)
        with self.assertRaises(ValueError):
            settings_from(INGEST_API_SEARCHES_PER_MINUTE="-1")
        self.assertTrue(settings_from(INGEST_API_DOCS="true").api_docs)


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class ApiLimits(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    def client(self, **settings):
        self.engine.settings = replace(self.engine.settings, **settings)
        self.other = uuid4()
        tenants = {"a": self.tenant, "b": self.other}
        app = create_app(self.engine, authenticate=lambda token: tenants.get(token))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")

    async def start(self, c, token="a"):
        data = DATA + uuid4().bytes  # distinct content, so duplicates never interfere
        return await c.post(
            "/v1/uploads",
            headers={"Authorization": "Bearer " + token, "Idempotency-Key": str(uuid4())},
            json={
                "filename": "a.txt",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )

    async def test_uploads_over_the_limit_get_429_with_retry_after(self):
        async with self.client(api_uploads_per_minute=2) as c:
            self.assertEqual([(await self.start(c)).status_code for _ in range(2)], [201, 201])
            refused = await self.start(c)
            self.assertEqual(refused.status_code, 429)
            self.assertGreaterEqual(int(refused.headers["Retry-After"]), 1)
            self.assertEqual((await self.start(c, token="b")).status_code, 201, "per tenant")

    async def test_searches_have_their_own_limit(self):
        async with self.client(api_searches_per_minute=1, api_uploads_per_minute=0) as c:
            ask = lambda: c.post(  # noqa: E731
                "/v1/search",
                headers={"Authorization": "Bearer a"},
                json={"query": "durable work", "limit": 3},
            )
            self.assertEqual((await ask()).status_code, 200)
            self.assertEqual((await ask()).status_code, 429)
            self.assertEqual((await self.start(c)).status_code, 201, "uploads unaffected")

    async def test_zero_means_unlimited(self):
        async with self.client(api_uploads_per_minute=0) as c:
            codes = [(await self.start(c)).status_code for _ in range(5)]
            self.assertEqual(codes, [201] * 5)

    async def test_the_limit_is_shared_by_every_api_process(self):
        """Two app instances (two processes in production) draw from one bucket."""
        async with self.client(api_uploads_per_minute=2) as first:
            app = create_app(self.engine, authenticate=lambda token: self.tenant)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as second:
                self.assertEqual((await self.start(first)).status_code, 201)
                self.assertEqual((await self.start(second)).status_code, 201)
                self.assertEqual((await self.start(first)).status_code, 429)


@unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
class ApiDocs(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.PostgresIntegration.asyncSetUp
    asyncTearDown = integration.PostgresIntegration.asyncTearDown

    async def paths(self, docs):
        self.engine.settings = replace(self.engine.settings, api_docs=docs)
        app = create_app(self.engine, authenticate=lambda token: self.tenant)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            return [
                (await c.get(path)).status_code
                for path in ("/docs", "/redoc", "/openapi.json")
            ]

    async def test_docs_are_off_by_default(self):
        self.assertEqual(await self.paths(False), [404, 404, 404])

    async def test_docs_can_be_turned_on(self):
        self.assertEqual(await self.paths(True), [200, 200, 200])


if __name__ == "__main__":
    unittest.main()
