"""The segment format: exact float32 round trip, self-verification, merging."""

import json
import struct
import unittest
from uuid import uuid4

from rag_ingest.adapters.postgres.codec import encode
from rag_ingest.domain.chunks import Chunk, SourceSpan
from rag_ingest.domain.embeddings import EmbeddedChunk
from rag_ingest.runtime.embedding_segments import (
    decode_segment,
    encode_segment,
    merge_segments,
)

RECIPE = "a" * 64
IDS = uuid4()


def item(ordinal, vector, recipe=RECIPE):
    chunk = Chunk(
        tenant_id=IDS,
        document_id=IDS,
        source_version_id=IDS,
        generation_id=IDS,
        ordinal=ordinal,
        text=f"chunk number {ordinal}",
        spans=(
            SourceSpan(
                extraction_sha256="0" * 64, page_number=1, start_char=0, end_char=10
            ),
        ),
    )
    return EmbeddedChunk(
        chunk=chunk,
        pipeline_fingerprint=recipe,
        dimensions=len(vector),
        vector=tuple(vector),
    )


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class SegmentFormat(unittest.TestCase):
    def setUp(self):
        self.items = [item(0, (0.1, -0.2, 0.3, 1e-8)), item(1, (1.5, 2.25, -3.0, 0.0))]
        self.data = encode_segment(self.items, recipe=RECIPE)

    def test_round_trip_is_exact_at_float32_precision(self):
        back = decode_segment(self.data, recipe=RECIPE, dimensions=4)
        self.assertEqual([b.chunk for b in back], [i.chunk for i in self.items])
        for original, restored in zip(self.items, back):
            self.assertEqual(restored.vector, tuple(f32(v) for v in original.vector))

    def test_any_changed_byte_is_detected(self):
        for position in (
            0,
            9,
            20,
            len(self.data) // 2,
            len(self.data) - 40,
            len(self.data) - 1,
        ):
            damaged = bytearray(self.data)
            damaged[position] ^= 0x01
            with self.subTest(position=position), self.assertRaises(ValueError):
                decode_segment(bytes(damaged), recipe=RECIPE, dimensions=4)

    def test_truncation_and_foreign_builds_are_rejected(self):
        for data, recipe, dims in (
            (self.data[:-1], RECIPE, 4),
            (self.data[:10], RECIPE, 4),
            (self.data, "b" * 64, 4),  # another build's recipe
            (self.data, RECIPE, 8),  # another dimension
        ):
            with self.assertRaises(ValueError):
                decode_segment(data, recipe=recipe, dimensions=dims)

    def test_values_outside_float32_are_refused_when_writing(self):
        with self.assertRaises(ValueError):
            encode_segment([item(0, (1e39, 0.0))], recipe=RECIPE)

    def test_mixed_recipes_or_dimensions_are_refused_when_writing(self):
        with self.assertRaises(ValueError):
            encode_segment(
                [item(0, (1.0, 2.0)), item(1, (1.0, 2.0, 3.0))], recipe=RECIPE
            )
        with self.assertRaises(ValueError):
            encode_segment([item(0, (1.0, 2.0), recipe="c" * 64)], recipe=RECIPE)

    def test_segment_is_far_smaller_than_json(self):
        import random

        rng = random.Random(1)
        batch = [item(n, [rng.uniform(-1, 1) for _ in range(1536)]) for n in range(64)]
        segment = len(encode_segment(batch, recipe=RECIPE))
        as_json = len(json.dumps(encode(tuple(batch)), sort_keys=True).encode())
        self.assertLess(
            segment, 64 * 1536 * 4 + 64 * 1024
        )  # float32 body + small header
        self.assertLess(segment * 4, as_json, f"{segment} vs JSON {as_json} bytes")


class Merging(unittest.TestCase):
    def test_first_segment_wins_and_order_follows_the_batch(self):
        a = (item(0, (1.0, 0.0)), item(1, (1.0, 0.0)))
        b = (item(1, (0.0, 1.0)), item(2, (0.0, 1.0)))
        ids = [i.chunk.vector_id for i in (a[0], a[1], b[1])]
        merged = merge_segments([a, b], ids)
        self.assertEqual([m.chunk.ordinal for m in merged], [0, 1, 2])
        self.assertEqual(merged[1].vector, (1.0, 0.0), "the earlier segment wins")

    def test_missing_or_foreign_chunks_are_rejected(self):
        a = (item(0, (1.0, 0.0)),)
        with self.assertRaises(ValueError):  # chunk 1 missing
            merge_segments(
                [a], [a[0].chunk.vector_id, item(1, (1.0, 0.0)).chunk.vector_id]
            )
        with self.assertRaises(ValueError):  # chunk 0 not part of this batch
            merge_segments([a], [item(5, (1.0, 0.0)).chunk.vector_id])


if __name__ == "__main__":
    unittest.main()
