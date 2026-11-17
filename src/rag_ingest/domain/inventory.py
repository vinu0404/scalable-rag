"""Describe the expected chunk batches of one processing generation.

An inventory is a plan to verify, not a claim that embedding or indexing has
finished. Save and seal it before releasing downstream batch work.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import ClassVar
from uuid import UUID

from rag_ingest.domain.enums import ProcessingStage
from rag_ingest.domain.manifests import ManifestOutput, StageManifest


@dataclass(frozen=True, kw_only=True)
class ChunkBatch:
    """Identify a saved chunk-batch manifest and its expected ordinal range.

    start_ordinal is included; end_ordinal is excluded. For example, [0, 100)
    means chunks 0 through 99. Each referenced manifest lists the exact IDs and
    chunk-output files for that batch. It uses the version-1 stage-manifest format.
    """

    batch_key: str
    start_ordinal: int
    end_ordinal: int
    manifest: ManifestOutput

    def __post_init__(self) -> None:
        """Reject invalid ranges or references that are not supported manifests."""
        if not isinstance(self.batch_key, str) or not self.batch_key.strip():
            raise ValueError("batch_key must not be blank")
        if (
            type(self.start_ordinal) is not int
            or type(self.end_ordinal) is not int
            or self.start_ordinal < 0
            or self.end_ordinal <= self.start_ordinal
        ):
            raise ValueError("Batch range must have 0 <= start_ordinal < end_ordinal")
        if not isinstance(self.manifest, ManifestOutput):
            raise ValueError("manifest must be a ManifestOutput")
        if (
            self.manifest.format_name != "stage-manifest"
            or self.manifest.format_version != StageManifest.SCHEMA_VERSION
        ):
            raise ValueError("Batch must reference the supported stage-manifest format")
        if self.manifest.size_bytes <= 0:
            raise ValueError("Batch manifest must not be empty")


@dataclass(frozen=True, kw_only=True)
class GenerationInventory:
    """List every expected chunk batch for one source and pipeline generation.

    Require contiguous ranges from zero through chunk_count, without gaps or
    overlaps. Store batch references instead of every vector or chunk text.
    The service must bound inventory size, batch count, and each batch size.

    vector_ids_sha256 is the expected generation-wide ID-set digest using
    version-1 sorted JSON-array encoding. This class checks its format only;
    a verifier must recompute it across the exact IDs in all referenced batches.
    Empty inventories are representable, but product policy decides whether
    a document with no chunks can be accepted or published.
    """

    tenant_id: UUID
    document_id: UUID
    source_version_id: UUID
    generation_id: UUID
    source_sha256: str
    pipeline_fingerprint: str
    vector_target: str
    dimensions: int
    chunk_count: int
    vector_ids_sha256: str
    batches: tuple[ChunkBatch, ...]

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Check identity and exact range coverage without opening batch files."""
        for name in ("tenant_id", "document_id", "source_version_id", "generation_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        for name in ("source_sha256", "pipeline_fingerprint", "vector_ids_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if not isinstance(self.vector_target, str) or not self.vector_target.strip():
            raise ValueError("vector_target must not be blank")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if type(self.chunk_count) is not int or self.chunk_count < 0:
            raise ValueError("chunk_count must be a nonnegative integer")
        if not isinstance(self.batches, tuple) or any(
            not isinstance(b, ChunkBatch) for b in self.batches
        ):
            raise ValueError("batches must be a tuple of ChunkBatch records")
        keys = [batch.batch_key for batch in self.batches]
        if len(keys) != len(set(keys)):
            raise ValueError("Batch keys must be unique")
        next_ordinal = 0
        for batch in sorted(self.batches, key=lambda batch: batch.start_ordinal):
            if batch.start_ordinal != next_ordinal:
                raise ValueError("Batch ranges must not overlap or leave gaps")
            next_ordinal = batch.end_ordinal
        if next_ordinal != self.chunk_count:
            raise ValueError("Batch ranges must cover exactly chunk_count chunks")

    def to_json(self) -> str:
        """Encode the complete version-1 plan with deterministic batch ordering."""
        payload = asdict(self)
        for name in ("tenant_id", "document_id", "source_version_id", "generation_id"):
            payload[name] = str(getattr(self, name))
        payload["schema_version"] = self.SCHEMA_VERSION
        payload["batches"] = [
            asdict(b) for b in sorted(self.batches, key=lambda b: b.start_ordinal)
        ]
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @cached_property
    def sha256(self) -> str:
        """Hash the exact canonical inventory bytes for durable plan registration.

        Computed once per (immutable) object: plan_for runs once per batch.
        """
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def validate_batch(self, batch_key: str, manifest: StageManifest) -> None:
        """Check one downloaded chunk manifest against its pinned inventory entry.

        Check its scope, recipe, stage, hash, size, and exact expected vector-ID
        range. The caller must first enforce per-batch limits and verify actual
        downloaded bytes. Output-file contents and vector readiness are separate.
        """
        if not any(b.batch_key == batch_key for b in self.batches):
            raise ValueError("A registered batch and stage manifest are required")
        self.plan_for(batch_key).validate_batch(manifest)

    def plan_for(self, batch_key: str) -> "BatchPlan":
        """The one batch plus the generation facts an embedding job needs."""
        batch = next((b for b in self.batches if b.batch_key == batch_key), None)
        if batch is None:
            raise ValueError("Batch is not part of the sealed inventory")
        return BatchPlan(
            tenant_id=self.tenant_id,
            document_id=self.document_id,
            generation_id=self.generation_id,
            source_sha256=self.source_sha256,
            pipeline_fingerprint=self.pipeline_fingerprint,
            vector_target=self.vector_target,
            dimensions=self.dimensions,
            inventory_sha256=self.sha256,
            batch=batch,
        )


@dataclass(frozen=True, kw_only=True)
class BatchPlan:
    """One sealed batch with the generation facts its embedding job needs.

    Copied from the sealed inventory in the same transaction that schedules
    the batch's job, so it cannot disagree with that inventory. An embedding
    job reads this one small record instead of the whole inventory, whose
    size grows with the document (reading it per batch would be quadratic).
    """

    tenant_id: UUID
    document_id: UUID
    generation_id: UUID
    source_sha256: str
    pipeline_fingerprint: str
    vector_target: str
    dimensions: int
    inventory_sha256: str
    batch: ChunkBatch

    def __post_init__(self) -> None:
        """Reject records that could not have come from a valid inventory."""
        for name in ("tenant_id", "document_id", "generation_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        for name in ("source_sha256", "pipeline_fingerprint", "inventory_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if type(self.dimensions) is not int or self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if not isinstance(self.batch, ChunkBatch):
            raise ValueError("batch must be a ChunkBatch")

    @property
    def vector_ids(self) -> tuple[str, ...]:
        """The exact logical IDs this batch must produce, in ordinal order."""
        prefix = f"{self.tenant_id}:{self.document_id}:{self.generation_id}:"
        return tuple(
            prefix + str(i)
            for i in range(self.batch.start_ordinal, self.batch.end_ordinal)
        )

    def validate_batch(self, manifest: StageManifest) -> None:
        """Check one downloaded chunk manifest against this pinned batch entry."""
        batch = self.batch
        if not isinstance(manifest, StageManifest):
            raise ValueError("A registered batch and stage manifest are required")
        for name in (
            "tenant_id",
            "document_id",
            "generation_id",
            "source_sha256",
            "pipeline_fingerprint",
        ):
            if getattr(self, name) != getattr(manifest, name):
                raise ValueError(
                    "Batch manifest does not match its generation inventory"
                )
        if (
            manifest.stage is not ProcessingStage.CHUNK
            or manifest.shard_key != batch.batch_key
        ):
            raise ValueError(
                "Batch manifest stage or shard does not match the inventory"
            )
        if (
            manifest.sha256 != batch.manifest.sha256
            or len(manifest.to_json().encode("utf-8")) != batch.manifest.size_bytes
        ):
            raise ValueError(
                "Batch manifest bytes do not match the inventory reference"
            )
        if set(manifest.vector_ids) != set(self.vector_ids):
            raise ValueError("Batch manifest IDs do not match its exact planned range")
