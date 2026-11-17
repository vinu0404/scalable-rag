"""Verify custom adapter result validation and bounded local text processing."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
import httpx
from rag_ingest.adapters.processing.text import TextExtractor
from rag_ingest.domain import extraction_recipe
from rag_ingest.domain.pipeline import ComponentSpec, PipelineConfig
from rag_ingest.domain.chunks import Chunk, SourceSpan
from rag_ingest.contracts.extraction import ExtractionLimits
from rag_ingest.contracts.embedding import EmbeddingLimits
from rag_ingest.contracts.errors import UncertainOutcomeError, RateLimitedError
from rag_ingest.adapters.postgres.codec import encode, decode


def chunk():
    """Create a small valid input that preserves source identity."""
    id = uuid4()
    return Chunk(
        tenant_id=id,
        document_id=id,
        source_version_id=id,
        generation_id=id,
        ordinal=0,
        text="hello",
        spans=(
            SourceSpan(
                extraction_sha256=hashlib.sha256(b"hello").hexdigest(),
                start_char=0,
                end_char=5,
                page_number=1,
            ),
        ),
    )


def pipeline():
    """Use a recipe for the hypothetical example gateway."""
    return PipelineConfig(
        extractor=ComponentSpec(
            driver="document",
            implementation_version="1",
            options_json=extraction_recipe.options_json(extraction_recipe.validate({})),
        ),
        chunker=ComponentSpec(driver="characters", implementation_version="1"),
        embedder=ComponentSpec(
            driver="company_http",
            implementation_version="1",
            options_json=json.dumps(
                {"model": "company-v2", "dimensions": 2, "encoding": "utf8-bytes"}
            ),
        ),
    )


class Processing(unittest.IsolatedAsyncioTestCase):
    async def adapter(self, handler):
        spec = importlib.util.spec_from_file_location(
            "company_embedder",
            Path(__file__).parents[1] / "examples/company_embedder.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CompanyEmbedder(
            base_url="https://company.invalid",
            token="test",
            transport=httpx.MockTransport(handler),
        )

    async def test_company_adapter_success_and_codec_round_trip(self):
        adapter = await self.adapter(
            lambda r: httpx.Response(200, json={"vectors": [[0.5, 0.25]]})
        )
        c = chunk()
        p = pipeline()
        try:
            result = await adapter.embed_batch(
                (c,),
                pipeline=p,
                dimensions=2,
                limits=EmbeddingLimits(
                    max_items=2,
                    max_input_characters=100,
                    max_batch_tokens=100,
                    timeout_seconds=1,
                ),
            )
            result.validate_for((c,), pipeline_fingerprint=p.fingerprint, dimensions=2)
            self.assertEqual(
                decode(json.loads(json.dumps(encode(result.items[0])))), result.items[0]
            )
        finally:
            await adapter.aclose()

    async def test_company_adapter_missing_output_is_uncertain(self):
        adapter = await self.adapter(
            lambda r: httpx.Response(200, json={"vectors": []})
        )
        try:
            with self.assertRaises(UncertainOutcomeError):
                await adapter.embed_batch(
                    (chunk(),),
                    pipeline=pipeline(),
                    dimensions=2,
                    limits=EmbeddingLimits(
                        max_items=2,
                        max_input_characters=100,
                        max_batch_tokens=100,
                        timeout_seconds=1,
                    ),
                )
        finally:
            await adapter.aclose()

    async def test_company_adapter_throttling_classified(self):
        adapter = await self.adapter(lambda r: httpx.Response(429))
        try:
            with self.assertRaises(RateLimitedError):
                await adapter.embed_batch(
                    (chunk(),),
                    pipeline=pipeline(),
                    dimensions=2,
                    limits=EmbeddingLimits(
                        max_items=2,
                        max_input_characters=100,
                        max_batch_tokens=100,
                        timeout_seconds=1,
                    ),
                )
        finally:
            await adapter.aclose()

    async def test_text_parser_runs_in_child_and_preserves_text(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.txt"
            source.write_text("Hello ingestion.\nSecond line.", encoding="utf8")
            extractor = TextExtractor()
            async with extractor.open_pages(
                source,
                spec=pipeline().extractor,
                limits=ExtractionLimits(
                    max_pages=5,
                    max_page_characters=1000,
                    max_total_characters=1000,
                    timeout_seconds=10,
                ),
            ) as pages:
                result = [p async for p in pages]
            self.assertEqual(result[0].text, source.read_text())

    async def test_default_openai_adapter_maps_provider_index(self):
        from dataclasses import replace
        from unittest.mock import patch, Mock
        from rag_ingest.adapters.processing.embedding import OpenAIEmbedder

        def respond(request):
            body = json.loads(request.content)
            self.assertEqual(body["input"], ["hello"])
            self.assertEqual(body["dimensions"], 2)
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.25, 0.5]}]}
            )

        adapter = OpenAIEmbedder(api_key="test", transport=httpx.MockTransport(respond))
        p = replace(
            pipeline(),
            embedder=ComponentSpec(
                driver="openai",
                implementation_version="1",
                options_json=json.dumps(
                    {"model": "test-model", "dimensions": 2, "encoding": "cl100k_base"}
                ),
            ),
        )
        try:
            with patch(
                "tiktoken.get_encoding",
                return_value=Mock(encode=lambda text, **kw: list(text)),
            ):
                result = await adapter.embed_batch(
                    (chunk(),),
                    pipeline=p,
                    dimensions=2,
                    limits=EmbeddingLimits(
                        max_items=2,
                        max_input_characters=100,
                        max_batch_tokens=100,
                        timeout_seconds=1,
                    ),
                )
            self.assertEqual(result.items[0].vector, (0.25, 0.5))
        finally:
            await adapter.aclose()

    async def test_default_openai_adapter_foreign_index_is_uncertain(self):
        from dataclasses import replace
        from unittest.mock import patch, Mock
        from rag_ingest.adapters.processing.embedding import OpenAIEmbedder

        adapter = OpenAIEmbedder(
            api_key="test",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json={"data": [{"index": 9, "embedding": [0.25, 0.5]}]}
                )
            ),
        )
        p = replace(
            pipeline(),
            embedder=ComponentSpec(
                driver="openai",
                implementation_version="1",
                options_json=json.dumps(
                    {"model": "test-model", "dimensions": 2, "encoding": "cl100k_base"}
                ),
            ),
        )
        try:
            with patch(
                "tiktoken.get_encoding",
                return_value=Mock(encode=lambda text, **kw: list(text)),
            ):
                with self.assertRaises(UncertainOutcomeError):
                    await adapter.embed_batch(
                        (chunk(),),
                        pipeline=p,
                        dimensions=2,
                        limits=EmbeddingLimits(
                            max_items=2,
                            max_input_characters=100,
                            max_batch_tokens=100,
                            timeout_seconds=1,
                        ),
                    )
        finally:
            await adapter.aclose()
