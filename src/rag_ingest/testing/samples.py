"""Build small, realistic PDFs for tests: text, scanned, mixed and blank pages.

Each page spec is a (kind, value) pair:
- ("text", "Hello"): native text a parser can extract.
- ("image", "SCANNED"): an image-only page (like a scan), text drawn in pixels.
- ("mixed", ("Caption", "IN IMAGE")): native text plus an embedded image.
- ("blank", None): no text and no images.

The output is a valid PDF 1.4 file written without third-party PDF tools;
image pages need Pillow to encode the embedded JPEG.
"""

import io


def _escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_stream(text, top=720):
    lines = text.split("\n")
    body = [f"BT /F1 12 Tf 72 {top} Td 14 TL"]
    for line in lines:
        body.append(f"({_escape(line)}) Tj T*")
    body.append("ET")
    return "\n".join(body).encode("latin-1")


def _jpeg(label, width=850, height=1100):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in range(8):
        draw.text((60, 80 + row * 120), f"{label} line {row + 1}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue(), width, height


def build_pdf(pages):
    """Return PDF bytes for the page specs, in order."""
    objects = []  # list of bytes bodies; object number = index + 1

    def add(body):
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # placeholders filled once page ids are known
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for kind, value in pages:
        resources, content = [f"/Font << /F1 {font} 0 R >>"], b""
        if kind in ("image", "mixed"):
            label = value if kind == "image" else value[1]
            data, width, height = _jpeg(label)
            image = add(
                f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
                f"/Length {len(data)} >>\nstream\n".encode()
                + data
                + b"\nendstream"
            )
            resources.append(f"/XObject << /Im1 {image} 0 R >>")
            box = "612 0 0 792 0 0" if kind == "image" else "300 0 0 388 72 100"
            content += f"q {box} cm /Im1 Do Q\n".encode()
        if kind == "text":
            content += _text_stream(value)
        elif kind == "mixed":
            content += _text_stream(value[0])
        elif kind not in ("image", "blank"):
            raise ValueError("Unknown sample page kind")
        stream = add(
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << {' '.join(resources)} >> /Contents {stream} 0 R >>".encode()
            )
        )
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode()
    objects[pages_obj - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    )
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def sample_png(label="SAMPLE", width=600, height=200):
    """A small PNG with drawn text, for page-image recognizer tests."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).text((20, 80), label, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
