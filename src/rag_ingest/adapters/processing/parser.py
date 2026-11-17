"""Run file parsing in a killable child process, never in the API event loop.

Commands (argv[1]):
- analyze source output limits options — page text plus visual-content facts
  for the document recipe; the parent decides which pages need recognition.
- render source outdir params — rasterize listed pages to images that fit a
  recognizer's pixel/byte limits (needs the optional rendering extra).
- split source output params — write a PDF containing only listed pages.

Exit codes: 2 unsupported/corrupt/excessive input, 3 missing optional
dependency. Diagnostics never include input content or paths.
"""

import io
import json
import math
import sys
from pathlib import Path

EXIT_INPUT, EXIT_DEPENDENCY = 2, 3


def _limit_memory():
    """Bound parser memory on Unix in addition to the parent's killable timeout."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    except (ImportError, ValueError, OSError):
        pass


def _visual_facts(page):
    """Count images and content-stream bytes; unknown counts as visual (-1)."""
    try:
        images = len(page.images)
    except Exception:
        images = -1
    try:
        contents = page.get_contents()
        content_bytes = len(contents.get_data()) if contents is not None else 0
    except Exception:
        content_bytes = -1
    return images, content_bytes


def analyze(source, output, limits, options):
    """Write page text and visual facts; zero text is a valid analysis result."""
    _limit_memory()
    from pypdf import PdfReader

    total = 0
    with open(source, "rb") as src:
        signature = src.read(5)
    with open(output, "w", encoding="utf-8", newline="\n") as dest:
        if signature == b"%PDF-":
            reader = PdfReader(source, strict=bool(options.get("strict_pdf")))
            if reader.is_encrypted:
                raise ValueError("encrypted_pdf")
            if len(reader.pages) > limits["max_pages"]:
                raise ValueError("page_limit")
            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                images, content_bytes = _visual_facts(page)
                total += len(text)
                if (
                    len(text) > limits["max_page_characters"]
                    or total > limits["max_total_characters"]
                ):
                    raise ValueError("text_limit")
                dest.write(
                    json.dumps(
                        {
                            "page_number": index + 1,
                            "text": text,
                            "image_count": images,
                            "content_bytes": content_bytes,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        else:
            if Path(source).stat().st_size > limits["max_total_characters"] * 4:
                raise ValueError("text_limit")
            text = Path(source).read_text(encoding="utf-8")
            if len(text) > limits["max_total_characters"]:
                raise ValueError("text_limit")
            dest.write(
                json.dumps(
                    {
                        "page_number": 1,
                        "text": text,
                        "image_count": 0,
                        "content_bytes": 0,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def _encode(image, media_types, max_bytes):
    """Return (bytes, media_type) within max_bytes, preferring lossless PNG."""
    if "image/png" in media_types:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        if max_bytes is None or buffer.tell() <= max_bytes:
            return buffer.getvalue(), "image/png"
    if "image/jpeg" in media_types:
        rgb = image.convert("RGB")
        for quality in (90, 80, 70, 60, 50):
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
            if max_bytes is None or buffer.tell() <= max_bytes:
                return buffer.getvalue(), "image/jpeg"
    return None, None


def render(source, outdir, params):
    """Rasterize pages within pixel and byte limits; record unrenderable pages."""
    _limit_memory()
    try:
        import pypdfium2
    except ImportError:
        sys.exit(EXIT_DEPENDENCY)
    try:
        import PIL  # noqa: F401  (needed by to_pil)
    except ImportError:
        sys.exit(EXIT_DEPENDENCY)
    out = Path(outdir)
    pdf = pypdfium2.PdfDocument(source)
    try:
        lines = []
        for number in params["pages"]:
            page = pdf[number - 1]
            width_pt, height_pt = page.get_size()
            scale = params["dpi"] / 72
            pixels = width_pt * height_pt * scale * scale
            if params.get("max_pixels") and pixels > params["max_pixels"]:
                scale *= math.sqrt(params["max_pixels"] / pixels)
            entry = {"page": number}
            for _ in range(12):  # shrink until pixels and encoded bytes fit
                image = page.render(scale=scale).to_pil()
                limit = params.get("max_pixels")
                if limit and image.width * image.height > limit:
                    # Rounding can overshoot the cap; tools may enforce it hard.
                    scale *= math.sqrt(limit / (image.width * image.height)) * 0.999
                    continue
                data, media_type = _encode(
                    image, params["media_types"], params.get("max_bytes")
                )
                if data is not None:
                    name = f"page-{number:06d}" + (
                        ".png" if media_type == "image/png" else ".jpg"
                    )
                    (out / name).write_bytes(data)
                    entry.update(
                        file=name,
                        media_type=media_type,
                        width=image.width,
                        height=image.height,
                        dpi=round(scale * 72, 2),
                        size=len(data),
                    )
                    break
                if min(image.width, image.height) < 400:
                    break
                scale *= 0.75
            if "file" not in entry:
                entry["error"] = "image_too_large"
            page.close()
            lines.append(json.dumps(entry))
        (out / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    finally:
        pdf.close()


def split(source, output, params):
    """Write a PDF of only the listed pages, in the listed order."""
    _limit_memory()
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(source, strict=bool(params.get("strict_pdf")))
    if reader.is_encrypted:
        raise ValueError("encrypted_pdf")
    writer = PdfWriter()
    for number in params["pages"]:
        writer.add_page(reader.pages[number - 1])
    with open(output, "wb") as dest:
        writer.write(dest)


COMMANDS = {"analyze", "render", "split"}

if __name__ == "__main__":
    try:
        command, args = sys.argv[1], sys.argv[2:]
        if command == "analyze":
            analyze(args[0], args[1], json.loads(args[2]), json.loads(args[3]))
        elif command == "render":
            render(args[0], args[1], json.loads(args[2]))
        elif command == "split":
            split(args[0], args[1], json.loads(args[2]))
        else:
            raise ValueError("unknown parser command")
    except SystemExit:
        raise
    except Exception:
        # Raw parser diagnostics may contain input content or file paths.
        sys.stderr.write("unsupported_corrupt_or_excessive_input")
        sys.exit(EXIT_INPUT)
