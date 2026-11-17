"""Shared limits stay linked, and the embedding call settings take effect."""

import json
import os
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import httpx

from rag_ingest.adapters.postgres.vectors import PgVectorStore
from rag_ingest.contracts.embedding import EmbeddingLimits
from rag_ingest.domain import limits
from rag_ingest.domain.inventory_reader import InventoryReadLimits
from rag_ingest.domain.manifest_reader import ManifestReadLimits
from rag_ingest.domain.pipeline import ComponentSpec

from test_processing import chunk, pipeline

REQUIRED = {
    "INGEST_METADATA_DSN": "postgresql://u:p@localhost/db",
    "INGEST_S3_BUCKET": "bucket",
    "INGEST_SQS_QUEUE_URL": "https://sqs.us-east-1.amazonaws.com/1/q",
    "INGEST_EMBEDDING_DRIVER": "openai",
}


def settings_from(**env):
    from rag_ingest.runtime.settings import ServiceSettings

    with patch.dict(os.environ, {**REQUIRED, **env}, clear=True):
        return ServiceSettings.from_env()


class EmbeddingDriver(unittest.TestCase):
    def test_only_the_real_embedder_can_be_configured(self):
        """The test embedder is injected in code, never selected by configuration."""
        with self.assertRaisesRegex(ValueError, "must be openai"):
            settings_from(INGEST_EMBEDDING_DRIVER="deterministic")


class LinkedLimits(unittest.TestCase):
    """Writers and readers of the same format must use the same number."""

    def test_readers_use_the_shared_limits(self):
        self.assertEqual(InventoryReadLimits().max_batches, limits.MAX_INVENTORY_BATCHES)
        self.assertEqual(InventoryReadLimits().max_chunks, limits.MAX_GENERATION_CHUNKS)
        self.assertEqual(ManifestReadLimits().max_bytes, limits.MAX_MANIFEST_BYTES)
        self.assertEqual(PgVectorStore.MAX_DIGEST_IDS, limits.MAX_DIGEST_IDS)

    def test_a_build_lists_every_job(self):
        # one embed job per batch, plus extract, chunk, index and publish
        self.assertEqual(limits.MAX_GENERATION_JOBS, limits.MAX_INVENTORY_BATCHES + 4)

    def test_batch_size_is_capped_by_the_shared_item_limit(self):
        self.assertEqual(
            settings_from(INGEST_EMBEDDING_BATCH_SIZE=str(limits.MAX_BATCH_ITEMS)).batch_size,
            limits.MAX_BATCH_ITEMS,
        )
        with self.assertRaises(ValueError):
            settings_from(INGEST_EMBEDDING_BATCH_SIZE=str(limits.MAX_BATCH_ITEMS + 1))


class EmbeddingCallSettings(unittest.TestCase):
    def test_defaults_keep_todays_behaviour(self):
        s = settings_from()
        self.assertEqual((s.embedding_timeout_seconds, s.embedding_max_batch_tokens), (30, 200_000))

    def test_values_are_read_and_bounded(self):
        s = settings_from(
            INGEST_EMBEDDING_TIMEOUT_SECONDS="120",
            INGEST_EMBEDDING_MAX_BATCH_TOKENS="300000",
        )
        self.assertEqual((s.embedding_timeout_seconds, s.embedding_max_batch_tokens), (120, 300_000))
        for name, value in (
            ("INGEST_EMBEDDING_TIMEOUT_SECONDS", "4"),
            ("INGEST_EMBEDDING_TIMEOUT_SECONDS", "601"),
            ("INGEST_EMBEDDING_MAX_BATCH_TOKENS", "999"),
        ):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                settings_from(**{name: value})

    def test_they_are_not_part_of_the_recipe(self):
        """Changing a runtime limit must not make accepted work look like a new build."""
        a = settings_from().pipeline().fingerprint
        b = settings_from(
            INGEST_EMBEDDING_TIMEOUT_SECONDS="120",
            INGEST_EMBEDDING_MAX_BATCH_TOKENS="300000",
        ).pipeline().fingerprint
        self.assertEqual(a, b)


class EmbedderTimeout(unittest.IsolatedAsyncioTestCase):
    """Each HTTP attempt gets two thirds of the call budget (30 s -> 20 s, as before)."""

    async def read_timeout_for(self, timeout_seconds):
        from rag_ingest.adapters.processing.embedding import OpenAIEmbedder

        seen = {}

        def respond(request):
            seen.update(request.extensions["timeout"])
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.25, 0.5]}]})

        adapter = OpenAIEmbedder(api_key="test", transport=httpx.MockTransport(respond))
        p = replace(
            pipeline(),
            embedder=ComponentSpec(
                driver="openai",
                implementation_version="1",
                options_json=json.dumps(
                    {"model": "m", "dimensions": 2, "encoding": "cl100k_base"}
                ),
            ),
        )
        try:
            with patch(
                "tiktoken.get_encoding",
                return_value=Mock(encode=lambda text, **kw: list(text)),
            ):
                await adapter.embed_batch(
                    (chunk(),),
                    pipeline=p,
                    dimensions=2,
                    limits=EmbeddingLimits(
                        max_items=2,
                        max_input_characters=100,
                        max_batch_tokens=100,
                        timeout_seconds=timeout_seconds,
                    ),
                )
        finally:
            await adapter.aclose()
        return seen

    async def test_default_budget_keeps_the_20_second_attempt(self):
        t = await self.read_timeout_for(30)
        self.assertEqual((t["read"], t["connect"]), (20, 5))

    async def test_a_longer_budget_really_lengthens_each_attempt(self):
        t = await self.read_timeout_for(120)
        self.assertEqual(t["read"], 80, "a slow model is no longer cut off at 20 s")


if __name__ == "__main__":
    unittest.main()
