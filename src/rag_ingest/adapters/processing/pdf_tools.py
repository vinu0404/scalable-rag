"""Run PDF analysis, page rendering and page splitting in killable subprocesses.

The parent never parses untrusted PDFs in-process: each call starts the
bounded parser child, enforces a timeout, kills it on timeout or
cancellation, and maps exit codes to contract error categories.
"""

import asyncio
import json
import sys
from pathlib import Path

from rag_ingest.contracts.errors import OperatorActionRequiredError, PermanentInputError

EXIT_INPUT, EXIT_DEPENDENCY = 2, 3


async def _run(args, timeout_seconds):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "rag_ingest.adapters.processing.parser",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode == EXIT_DEPENDENCY:
        raise OperatorActionRequiredError(
            "Page rendering needs the optional extra: pip install 'rag-ingest[ocr]'"
        )
    if process.returncode:
        raise PermanentInputError(
            "File is unsupported, corrupt, encrypted, or exceeds limits"
        )


async def analyze(source, output, *, limits, options, timeout_seconds):
    """Write per-page text and visual facts (JSONL) for the document recipe."""
    await _run(
        ["analyze", str(source), str(output), json.dumps(limits), json.dumps(options)],
        timeout_seconds,
    )


async def render(
    source, outdir, *, pages, dpi, max_pixels, max_bytes, media_types, timeout_seconds
):
    """Rasterize pages; return {page: entry} with file/media_type or error."""
    params = {
        "pages": list(pages),
        "dpi": dpi,
        "max_pixels": max_pixels,
        "max_bytes": max_bytes,
        "media_types": sorted(media_types),
    }
    await _run(
        ["render", str(source), str(outdir), json.dumps(params)], timeout_seconds
    )
    manifest = Path(outdir) / "manifest.jsonl"
    entries = {}
    for line in manifest.read_text().splitlines():
        if line:
            entry = json.loads(line)
            entries[entry["page"]] = entry
    if sorted(entries) != sorted(pages):
        raise PermanentInputError("Page rendering did not cover the requested pages")
    return entries


async def split(source, output, *, pages, strict_pdf, timeout_seconds):
    """Write a PDF that contains exactly the listed pages, in order."""
    params = {"pages": list(pages), "strict_pdf": strict_pdf}
    await _run(["split", str(source), str(output), json.dumps(params)], timeout_seconds)
