"""Assemble the 50-PDF live test set in test_data/pdfs (git-ignored).

30 text papers from arXiv, 10 generated scans, 5 mixed (text + scanned
pages), 5 edge cases. Generated scans have known text, written to
test_data/ground_truth.json (never uploaded) to measure recognition accuracy.

arXiv is fetched politely: one request every 3 seconds, a generic client
name, and files already present are skipped, so the script can be re-run.
"""

import hashlib
import io
import json
import random
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pypdf import PdfReader, PdfWriter

from rag_ingest.testing.samples import build_pdf

OUT = Path("test_data/pdfs")
TRUTH = Path("test_data/ground_truth.json")
ARXIV = [
    "1706.03762",
    "1810.04805",
    "1512.03385",
    "1409.0473",
    "1412.6980",
    "1502.03167",
    "1406.2661",
    "1312.6114",
    "2010.11929",
    "2203.02155",
    "2302.13971",
    "2005.11401",
    "1301.3781",
    "1607.06450",
    "1505.04597",
    "1506.02640",
    "1311.2524",
    "1703.06870",
    "1905.11946",
    "2106.09685",
    "2001.08361",
    "1909.11942",
    "1907.11692",
    "2006.11239",
    "1508.04025",
    "1804.02767",
    "2201.11903",
    "1910.10683",
    "2005.14165",
    "2112.10752",
]
FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]
SENTENCES = [
    "Durable ingestion keeps accepted work recoverable after a worker crashes.",
    "Each document is verified against its exact stored version before processing.",
    "The outbox records notification intent in the same transaction as the job.",
    "Leases expire so that another worker can resume abandoned work safely.",
    "Recognized pages are cached so retries never bill the same page twice.",
    "Invoice number 48213 was issued on 14 March 2026 for 1,250 units.",
    "The quarterly report shows revenue growth of 12 percent year over year.",
    "Please sign and return the attached agreement within thirty days.",
    "Shipment tracking code KX-5521-QR arrived at the regional warehouse.",
    "Meeting minutes: the committee approved the revised safety procedure.",
]


def scan_page(lines, *, font, seed, rotate=0.0, noise=False):
    """Render text as a 200 dpi page image, optionally skewed and noisy."""
    rng = random.Random(seed)
    image = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(image)
    face = ImageFont.truetype(font, 34)
    y = 160
    for line in lines:
        draw.text((140, y), line, fill=0, font=face)
        y += 70
    if noise:
        pixels = image.load()
        for _ in range(25000):
            pixels[rng.randrange(1700), rng.randrange(2200)] = rng.choice((0, 180))
        image = image.filter(ImageFilter.GaussianBlur(0.6))
    if rotate:
        image = image.rotate(rotate, expand=False, fillcolor=255)
    return image.convert("RGB")


def images_to_pdf(images):
    buffer = io.BytesIO()
    images[0].save(
        buffer, "PDF", save_all=True, append_images=images[1:], resolution=200.0
    )
    return buffer.getvalue()


def page_text(index, count=8):
    rng = random.Random(index)
    return [rng.choice(SENTENCES) for _ in range(count)]


def fetch_arxiv():
    client = httpx.Client(
        headers={"User-Agent": "rag-ingest-live-test/0.1"},
        timeout=120,
        follow_redirects=True,
    )
    for number, paper in enumerate(ARXIV, start=1):
        path = OUT / f"text-{number:02d}-arxiv-{paper}.pdf"
        if path.exists():
            continue
        response = client.get(f"https://arxiv.org/pdf/{paper}")
        response.raise_for_status()
        if not response.content.startswith(b"%PDF-"):
            raise RuntimeError(f"arXiv {paper} did not return a PDF")
        path.write_bytes(response.content)
        print(
            f"downloaded {path.name} ({len(response.content) / 1024**2:.1f} MiB)",
            flush=True,
        )
        time.sleep(3)
    client.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fetch_arxiv()
    truth = {}
    # 10 scans: 1-4 pages each, different fonts, some rotated or noisy.
    for n in range(1, 11):
        pages, images = [], []
        for p in range(1 + n % 4):
            lines = page_text(n * 10 + p)
            images.append(
                scan_page(
                    lines,
                    font=FONTS[n % 4],
                    seed=n * 10 + p,
                    rotate=(1.5 if n % 3 == 0 else 0.0),
                    noise=(n % 2 == 0),
                )
            )
            pages.append("\n".join(lines))
        name = f"scan-{n:02d}.pdf"
        (OUT / name).write_bytes(images_to_pdf(images))
        truth[name] = {str(i + 1): t for i, t in enumerate(pages)}
    # 5 mixed: two real text pages of a paper, then two scanned pages.
    for n in range(1, 6):
        source = PdfReader(OUT / f"text-{n:02d}-arxiv-{ARXIV[n - 1]}.pdf")
        writer = PdfWriter()
        for page in source.pages[:2]:
            writer.add_page(page)
        lines = [page_text(500 + n * 2), page_text(501 + n * 2)]
        scans = PdfReader(
            io.BytesIO(
                images_to_pdf(
                    [scan_page(page, font=FONTS[n % 4], seed=500 + n) for page in lines]
                )
            )
        )
        for page in scans.pages:
            writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        name = f"mixed-{n:02d}.pdf"
        (OUT / name).write_bytes(buffer.getvalue())
        truth[name] = {"3": "\n".join(lines[0]), "4": "\n".join(lines[1])}
    # 5 edge cases.
    small = build_pdf([("text", "This document is protected by a password.")])
    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(small)))
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    (OUT / "edge-01-encrypted.pdf").write_bytes(buffer.getvalue())
    corrupt = (OUT / "text-06-arxiv-1502.03167.pdf").read_bytes()
    (OUT / "edge-02-corrupt.pdf").write_bytes(corrupt[: len(corrupt) // 3])
    (OUT / "edge-03-blank.pdf").write_bytes(
        build_pdf([("blank", None), ("blank", None)])
    )
    long_pages = [page_text(900 + p) for p in range(30)]
    (OUT / "edge-04-30-page-scan.pdf").write_bytes(
        images_to_pdf(
            [
                scan_page(page, font=FONTS[p % 4], seed=900 + p)
                for p, page in enumerate(long_pages)
            ]
        )
    )
    truth["edge-04-30-page-scan.pdf"] = {
        str(p + 1): "\n".join(page) for p, page in enumerate(long_pages)
    }
    (OUT / "edge-05-duplicate-of-text-01.pdf").write_bytes(
        (OUT / f"text-01-arxiv-{ARXIV[0]}.pdf").read_bytes()
    )
    TRUTH.write_text(json.dumps(truth, indent=1, sort_keys=True))
    files = sorted(OUT.glob("*.pdf"))
    total = sum(f.stat().st_size for f in files)
    print(f"{len(files)} PDFs, {total / 1024**2:.1f} MiB total")
    for f in files:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        print(f"  {f.name:40s} {f.stat().st_size / 1024**2:7.2f} MiB  {digest}")


if __name__ == "__main__":
    main()
