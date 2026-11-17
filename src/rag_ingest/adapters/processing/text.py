"""Isolated PDF/text extraction and reproducible character-window chunking."""

import asyncio
import hashlib
import json
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from rag_ingest.adapters.processing import pdf_tools
from rag_ingest.contracts.extraction import ExtractedPage
from rag_ingest.domain import extraction_recipe
from rag_ingest.domain.chunks import Chunk, SourceSpan
from rag_ingest.contracts.errors import PermanentInputError


class TextExtractor:
    """Extract UTF-8 text and PDF pages, and classify pages for recognition.

    Runs the document v1 recipe: native text plus per-page classification.
    Pages that need recognition are yielded with needs_recognition=True; the
    service recognizes them with the injected Recognizer. Nothing is dropped
    silently: each page records how its text was obtained.
    """

    def open_pages(self, source_path, *, spec, limits):
        """Dispatch on the saved recipe; never substitute another behavior."""
        if (spec.driver, spec.implementation_version) == (
            extraction_recipe.DRIVER,
            extraction_recipe.VERSION,
        ):
            return self._document_pages(source_path, spec=spec, limits=limits)
        raise ValueError("Unsupported extractor recipe")

    @asynccontextmanager
    async def _document_pages(self, source_path, *, spec, limits):
        """Analyze in the bounded parser child, then classify every page."""
        options = extraction_recipe.validate(json.loads(spec.options_json))
        with open(source_path, "rb") as src:
            is_pdf = src.read(5) == b"%PDF-"
        with tempfile.TemporaryDirectory(prefix="rag-analyze-") as temp:
            output = Path(temp) / "pages.jsonl"
            await pdf_tools.analyze(
                source_path,
                output,
                limits=asdict(limits),
                options={"strict_pdf": options["strict_pdf"]},
                timeout_seconds=limits.timeout_seconds,
            )

            async def pages():
                """Yield one classified page at a time, in source order."""
                with output.open(encoding="utf-8") as stream:
                    while True:
                        line = await asyncio.to_thread(stream.readline)
                        if not line:
                            break
                        facts = json.loads(line)
                        text = facts["text"]
                        if options["normalize_text"]:
                            text = extraction_recipe.normalize(text)
                        method, needs = extraction_recipe.classify(
                            text,
                            facts["image_count"],
                            facts["content_bytes"],
                            options,
                            is_pdf=is_pdf,
                        )
                        if method == "unrecognized" and (
                            extraction_recipe.garbage_ratio(text)
                            > options["max_garbage_ratio"]
                        ):
                            text = ""  # an unreadable text layer is not content
                        yield ExtractedPage(
                            page_number=facts["page_number"],
                            text=text,
                            method=method,
                            needs_recognition=needs,
                        )

            yield pages()

    async def aclose(self):
        """No shared parser process is retained between calls."""


class CharacterChunker:
    """Split bounded saved extraction with explicit size and overlap in characters."""

    @asynccontextmanager
    async def open_chunks(self, extraction, *, source, generation, spec, limits):
        """Preserve exact source offsets and deterministic chunk identities."""
        if spec.driver != "characters" or spec.implementation_version != "1":
            raise ValueError("Unsupported chunker recipe")
        options = json.loads(spec.options_json)
        if set(options) != {"size", "overlap"}:
            raise ValueError("Character chunker requires size and overlap")
        size, overlap = options["size"], options["overlap"]
        if (
            type(size) is not int
            or type(overlap) is not int
            or not 0 <= overlap < size <= limits.max_chunk_characters
        ):
            raise ValueError("Invalid chunk size or overlap")
        if (source.tenant_id, source.document_id, source.source_version_id) != (
            generation.tenant_id,
            generation.document_id,
            generation.source_version_id,
        ):
            raise ValueError("Chunk source mismatch")
        # The extraction service enforces a total text limit before this bounded read.
        data = await asyncio.to_thread(extraction.text_path.read_bytes)
        page_bytes = await asyncio.to_thread(extraction.page_map_path.read_bytes)
        if (
            hashlib.sha256(data).hexdigest() != extraction.extraction_sha256
            or hashlib.sha256(page_bytes).hexdigest() != extraction.page_map_sha256
        ):
            raise ValueError("Extraction artifacts changed")
        text = data.decode("utf-8")
        mapping = json.loads(page_bytes)
        if mapping.get("schema_version") != 1:
            raise ValueError("Unsupported page-map format")

        async def chunks():
            """Yield chunks one at a time and retain page intersections as citations.

            The time budget counts only this generator's own work. A timeout
            scope spanning yields would cancel the consumer's task instead.
            """
            loop = asyncio.get_running_loop()
            spent = 0.0
            resumed = loop.time()
            ordinal = 0
            for start in range(0, len(text), size - overlap):
                end = min(start + size, len(text))
                part = text[start:end]
                if not part.strip():
                    continue
                if ordinal >= limits.max_chunks:
                    raise PermanentInputError("Chunk count exceeds policy")
                spans = tuple(
                    SourceSpan(
                        extraction_sha256=extraction.extraction_sha256,
                        start_char=max(start, p["start"]),
                        end_char=min(end, p["end"]),
                        page_number=p["page"],
                    )
                    for p in mapping["pages"]
                    if max(start, p["start"]) < min(end, p["end"])
                )
                if not spans:
                    raise ValueError("Chunk has no valid source provenance")
                spent += loop.time() - resumed
                if spent > limits.timeout_seconds:
                    raise TimeoutError("Chunking exceeded its time budget")
                yield Chunk(
                    tenant_id=source.tenant_id,
                    document_id=source.document_id,
                    source_version_id=source.source_version_id,
                    generation_id=generation.generation_id,
                    ordinal=ordinal,
                    text=part,
                    spans=spans,
                )
                resumed = loop.time()
                ordinal += 1
                await asyncio.sleep(0)
                if end == len(text):
                    break

        yield chunks()

    async def aclose(self):
        """No shared chunker resources are retained."""
