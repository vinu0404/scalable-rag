"""Decode a bounded inventory and verify its expected generation-wide ID digest.

The service must still read every registered batch manifest and validate it
against this inventory, then verify outputs and search visibility separately.
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, fields
from uuid import UUID

from rag_ingest.domain.inventory import ChunkBatch, GenerationInventory
from rag_ingest.domain.manifest_reader import _reject_constant, _unique_object
from rag_ingest.domain.manifests import ManifestOutput
from rag_ingest.domain.limits import MAX_GENERATION_CHUNKS, MAX_INVENTORY_BATCHES


@dataclass(frozen=True, kw_only=True)
class InventoryReadLimits:
    """Bound input bytes, batch entries, per-batch chunks, and digest work.

    These are configurable starting limits, not measured throughput claims.
    The downloader must enforce the byte limit before buffering the whole file.
    """

    max_bytes: int = 2_097_152
    max_batches: int = MAX_INVENTORY_BATCHES
    max_chunks_per_batch: int = 4096
    max_chunks: int = MAX_GENERATION_CHUNKS

    def __post_init__(self) -> None:
        """Require positive integer limits for memory and CPU bounds."""
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{item.name} must be a positive integer")


def _lexical_ordinals(count: int) -> Iterator[int]:
    """Yield 0 through count-1 in string order without storing a sorted list.

    For count=12, order is 0, 1, 10, 11, 2, ... 9. Walk decimal prefixes so
    memory stays constant instead of growing with the total number of chunks.
    The caller validates count before using this internal helper.
    """
    if count == 0:
        return
    yield 0
    current = 1
    for _ in range(1, count):
        yield current
        if current * 10 < count:
            current *= 10
        else:
            while current and (current % 10 == 9 or current + 1 >= count):
                current //= 10
            current += 1


def compute_vector_id_digest(
    *,
    tenant_id: UUID,
    document_id: UUID,
    generation_id: UUID,
    chunk_count: int,
    max_chunks: int,
) -> str:
    """Hash the canonical sorted ID array for contiguous chunk ordinals.

    Produce the same bytes as the version-1 JSON-array encoding incrementally.
    Limit chunk_count to bound CPU work. This checks the planned set, not which
    vectors or batch files actually exist. Use it when creating an inventory
    and when checking one after a restart.
    """
    if any(
        not isinstance(value, UUID) for value in (tenant_id, document_id, generation_id)
    ):
        raise ValueError("Digest scope must contain UUID identifiers")
    if type(max_chunks) is not int or max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer")
    if type(chunk_count) is not int or not 0 <= chunk_count <= max_chunks:
        raise ValueError("chunk_count exceeds the permitted digest work")
    digest = hashlib.sha256()
    digest.update(b"[")
    prefix = f"{tenant_id}:{document_id}:{generation_id}:"
    for position, ordinal in enumerate(_lexical_ordinals(chunk_count)):
        if position:
            digest.update(b",")
        digest.update(
            json.dumps(prefix + str(ordinal), ensure_ascii=True).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def read_generation_inventory(
    data: bytes,
    *,
    expected: ManifestOutput,
    limits: InventoryReadLimits,
) -> GenerationInventory:
    """Verify pinned bytes, decode canonical version 1, and check the planned ID set.

    expected must come from durable generation registration, not the same file
    being verified. Check limits before expensive construction or digest work.
    Successful decoding does not seal this plan in PostgreSQL or prove that
    batch contents match it. The service must also bind source, recipe, scope,
    dimensions, and vector target to the registered generation.
    """
    if not isinstance(data, bytes) or not isinstance(expected, ManifestOutput):
        raise ValueError(
            "Inventory bytes and a registered output reference are required"
        )
    if not isinstance(limits, InventoryReadLimits):
        raise ValueError("Inventory read limits are required")
    if (
        expected.format_name != "generation-inventory"
        or expected.format_version != GenerationInventory.SCHEMA_VERSION
    ):
        raise ValueError("Unsupported inventory format")
    if len(data) > limits.max_bytes or len(data) != expected.size_bytes:
        raise ValueError("Inventory byte count is invalid or exceeds its limit")
    if hashlib.sha256(data).hexdigest() != expected.sha256:
        raise ValueError("Inventory hash does not match its registered reference")
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, RecursionError):
        raise ValueError(
            "Inventory must contain valid UTF-8 JSON without duplicate keys"
        ) from None
    expected_fields = {item.name for item in fields(GenerationInventory)} | {
        "schema_version"
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("Inventory has missing or unexpected fields")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != GenerationInventory.SCHEMA_VERSION
    ):
        raise ValueError("Unsupported inventory schema version")
    count = payload["chunk_count"]
    if type(count) is not int or not 0 <= count <= limits.max_chunks:
        raise ValueError("Inventory chunk count is invalid or exceeds its limit")
    batches = payload["batches"]
    if not isinstance(batches, list) or len(batches) > limits.max_batches:
        raise ValueError("Inventory batch count is invalid or exceeds its limit")
    batch_fields = {item.name for item in fields(ChunkBatch)}
    output_fields = {item.name for item in fields(ManifestOutput)}
    for batch in batches:
        if not isinstance(batch, dict) or set(batch) != batch_fields:
            raise ValueError("Inventory batch fields are invalid")
        ref = batch["manifest"]
        if not isinstance(ref, dict) or set(ref) != output_fields:
            raise ValueError("Inventory manifest-reference fields are invalid")
        start, end = batch["start_ordinal"], batch["end_ordinal"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= count
        ):
            raise ValueError("Inventory batch range is invalid")
        if end - start > limits.max_chunks_per_batch:
            raise ValueError("Inventory batch exceeds its chunk limit")
    try:
        values = {
            key: value
            for key, value in payload.items()
            if key not in {"schema_version", "batches"}
        }
        for name in ("tenant_id", "document_id", "source_version_id", "generation_id"):
            if not isinstance(values[name], str):
                raise ValueError
            values[name] = UUID(values[name])
        inventory = GenerationInventory(
            **values,
            batches=tuple(
                ChunkBatch(
                    batch_key=b["batch_key"],
                    start_ordinal=b["start_ordinal"],
                    end_ordinal=b["end_ordinal"],
                    manifest=ManifestOutput(**b["manifest"]),
                )
                for b in batches
            ),
        )
        if inventory.to_json().encode("utf-8") != data:
            raise ValueError
    except (ValueError, TypeError, AttributeError, RecursionError, OverflowError):
        raise ValueError("Inventory fields or canonical encoding are invalid") from None
    actual_digest = compute_vector_id_digest(
        tenant_id=inventory.tenant_id,
        document_id=inventory.document_id,
        generation_id=inventory.generation_id,
        chunk_count=inventory.chunk_count,
        max_chunks=limits.max_chunks,
    )
    if actual_digest != inventory.vector_ids_sha256:
        raise ValueError("Inventory expected vector-ID digest is inconsistent")
    return inventory
