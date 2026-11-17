"""Immutable embedding segments: compact, self-verifying vector files.

A segment holds embedded chunks of ONE batch from ONE attempt:

    MAGIC (8 bytes, includes the format version)
    header length (4 bytes, big-endian)
    header (UTF-8 JSON): recipe, dimensions, count, and the chunks
    body: count x dimensions little-endian float32 values
    trailer: SHA-256 of everything before it (32 bytes)

Vectors are stored as float32 because that is exactly what the vector index
keeps and what publication verifies; nothing that reaches search is lost.
Segments are written once (create-only) and then serve two purposes: the
progress cache read by retries, and the data a stage manifest references by
pinned version and hash. Any defect raises ValueError; callers decide
whether that means "skip this cache entry" or "the saved build is corrupt".
"""

import hashlib
import json
import math
import struct

from rag_ingest.adapters.postgres.codec import decode, encode
from rag_ingest.domain.chunks import Chunk
from rag_ingest.domain.embeddings import EmbeddedChunk

MAGIC = b"RAGSEG\x00\x01"
FORMAT_NAME, FORMAT_VERSION = "embedding-segment", 1
HEADER_FIELDS = {"schema_version", "recipe", "dimensions", "count", "chunks"}
MAX_HEADER_BYTES = 64 * 1024**2


def encode_segment(items, *, recipe):
    """Serialize embedded chunks; vectors are rounded to float32 (finite only)."""
    items = tuple(items)
    if not items:
        raise ValueError("A segment needs at least one embedded chunk")
    dimensions = items[0].dimensions
    for item in items:
        if not isinstance(item, EmbeddedChunk):
            raise ValueError("Segments hold EmbeddedChunk records only")
        if item.dimensions != dimensions or item.pipeline_fingerprint != recipe:
            raise ValueError(
                "All segment items must share the build's recipe and dimensions"
            )
    header = json.dumps(
        {
            "schema_version": 1,
            "recipe": recipe,
            "dimensions": dimensions,
            "count": len(items),
            "chunks": [encode(item.chunk) for item in items],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    values = [value for item in items for value in item.vector]
    try:
        body = struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error):
        raise ValueError("A vector value is outside the float32 range") from None
    if any(math.isinf(v) for v in struct.unpack(f"<{len(values)}f", body)):
        raise ValueError("A vector value is outside the float32 range")
    data = MAGIC + struct.pack(">I", len(header)) + header + body
    return data + hashlib.sha256(data).digest()


def decode_segment(data, *, recipe, dimensions):
    """Parse and fully verify a segment; return its EmbeddedChunk records."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < len(MAGIC) + 4 + 32:
        raise ValueError("Segment is truncated")
    payload, trailer = bytes(data[:-32]), bytes(data[-32:])
    if hashlib.sha256(payload).digest() != trailer:
        raise ValueError("Segment checksum mismatch")
    if not payload.startswith(MAGIC):
        raise ValueError("Not an embedding segment of a supported version")
    (header_length,) = struct.unpack(">I", payload[len(MAGIC) : len(MAGIC) + 4])
    start = len(MAGIC) + 4
    if header_length > MAX_HEADER_BYTES or start + header_length > len(payload):
        raise ValueError("Segment header length is invalid")
    header = json.loads(payload[start : start + header_length])
    body = payload[start + header_length :]
    if not isinstance(header, dict) or set(header) != HEADER_FIELDS:
        raise ValueError("Segment header fields are invalid")
    if header["schema_version"] != 1 or header["recipe"] != recipe:
        raise ValueError("Segment belongs to another build recipe")
    count, dims = header["count"], header["dimensions"]
    if dims != dimensions or type(count) is not int or count < 1:
        raise ValueError("Segment dimensions or count are invalid")
    if len(header["chunks"]) != count or len(body) != count * dims * 4:
        raise ValueError("Segment body does not match its header")
    values = struct.unpack(f"<{count * dims}f", body)
    items = []
    for index, raw in enumerate(header["chunks"]):
        chunk = decode(raw)
        if not isinstance(chunk, Chunk):
            raise ValueError("Segment header holds a non-chunk record")
        items.append(
            EmbeddedChunk(
                chunk=chunk,
                pipeline_fingerprint=recipe,
                dimensions=dims,
                vector=tuple(values[index * dims : (index + 1) * dims]),
            )
        )
    return tuple(items)


def merge_segments(segments, expected_ids):
    """Combine segments in order (first wins) into exactly the expected chunks.

    segments: iterable of item tuples, already in manifest (role) order.
    Raises ValueError for a missing chunk or a chunk that is not expected,
    so a checkpoint can never stand for an incomplete or foreign batch.
    """
    expected = tuple(expected_ids)
    merged = {}
    for items in segments:
        for item in items:
            merged.setdefault(item.chunk.vector_id, item)
    if set(merged) != set(expected):
        raise ValueError("Embedding segments do not cover exactly the batch's chunks")
    return tuple(merged[vector_id] for vector_id in expected)
