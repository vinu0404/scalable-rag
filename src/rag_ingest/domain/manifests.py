"""Describe versioned output inventories for individual stage jobs or batches.

Save canonical JSON bytes to immutable storage, then register their SHA-256 in
a checkpoint. A manifest lists expected outputs; it does not prove they exist.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import ClassVar
from uuid import UUID

from rag_ingest.domain.checkpoints import StageCheckpoint
from rag_ingest.domain.enums import ProcessingStage


def _check_hash(name: str, value: str) -> None:
    """Require a SHA-256 hex string without calculating or verifying its content."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, kw_only=True)
class ManifestOutput:
    """Locate one saved output file and identify how its bytes should be read.

    role names its purpose, such as extracted_text, page_map, or vectors.
    format_name and format_version identify its serialization. sha256 and
    size_bytes let recovery check the exact file. Use pinned object versions
    or protected write-once keys; this class does not enforce storage policy.
    """

    role: str
    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    format_name: str
    format_version: int
    object_version_id: str | None = None

    def __post_init__(self) -> None:
        """Check locations, format labels, size, and hash without reading storage."""
        for name in ("role", "bucket", "object_key", "format_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be blank")
        _check_hash("sha256", self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        if type(self.format_version) is not int or self.format_version < 1:
            raise ValueError("format_version must be a positive integer")
        if self.object_version_id is not None:
            if (
                not isinstance(self.object_version_id, str)
                or not self.object_version_id.strip()
            ):
                raise ValueError("object_version_id must not be blank when provided")


@dataclass(frozen=True, kw_only=True)
class StageManifest:
    """List completed outputs for one stage job, not an entire unbounded import.

    Each output role appears once. vector_ids lists the exact logical chunk
    IDs represented by this batch when applicable. Extraction uses no vector
    IDs. An index-stage receipt file can describe remote writes, but that file
    alone does not establish search visibility; vector verification is separate.

    The service enforces manifest byte/entry limits and stage-specific required
    roles before storage. Large workloads use bounded batches and a separate
    generation inventory. These records do not define a publication decision.
    """

    tenant_id: UUID
    document_id: UUID
    generation_id: UUID
    job_id: UUID
    stage: ProcessingStage
    shard_key: str
    source_sha256: str
    input_sha256: str
    pipeline_fingerprint: str
    outputs: tuple[ManifestOutput, ...]
    vector_ids: tuple[str, ...] = ()

    SCHEMA_VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Reject malformed identities, hashes, duplicate roles, or foreign IDs."""
        for name in ("tenant_id", "document_id", "generation_id", "job_id"):
            if not isinstance(getattr(self, name), UUID):
                raise ValueError(f"{name} must be a UUID")
        if (
            not isinstance(self.stage, ProcessingStage)
            or self.stage is ProcessingStage.PUBLISH
        ):
            raise ValueError("stage must precede publication")
        if not isinstance(self.shard_key, str) or not self.shard_key.strip():
            raise ValueError("shard_key must not be blank")
        for name in ("source_sha256", "input_sha256", "pipeline_fingerprint"):
            _check_hash(name, getattr(self, name))
        if (
            not isinstance(self.outputs, tuple)
            or not self.outputs
            or any(not isinstance(output, ManifestOutput) for output in self.outputs)
        ):
            raise ValueError(
                "outputs must be a nonempty tuple of ManifestOutput records"
            )
        roles = [output.role for output in self.outputs]
        if len(roles) != len(set(roles)):
            raise ValueError("Each manifest output role must be unique")
        if not isinstance(self.vector_ids, tuple) or any(
            not isinstance(v, str) for v in self.vector_ids
        ):
            raise ValueError("vector_ids must be a tuple of logical ID strings")
        if len(self.vector_ids) != len(set(self.vector_ids)):
            raise ValueError("vector_ids must be unique within a manifest")
        prefix = f"{self.tenant_id}:{self.document_id}:{self.generation_id}:"
        for identity in self.vector_ids:
            suffix = identity.removeprefix(prefix)
            if (
                not identity.startswith(prefix)
                or not suffix.isascii()
                or not suffix.isdecimal()
            ):
                raise ValueError("vector_ids must match this scope and a chunk ordinal")
            if len(suffix) > 1 and suffix.startswith("0"):
                raise ValueError("Vector ordinal must use its canonical decimal form")
        if self.stage is ProcessingStage.EXTRACT and self.vector_ids:
            raise ValueError("Extraction manifests cannot contain vector IDs")

    def to_json(self) -> str:
        """Encode version-1 JSON consistently so saved bytes have a stable hash.

        Sort output roles and vector IDs; their input tuple order does not
        change this inventory. Chunk order remains encoded in each vector ID.
        """
        return json.dumps(
            {
                "schema_version": self.SCHEMA_VERSION,
                "tenant_id": str(self.tenant_id),
                "document_id": str(self.document_id),
                "generation_id": str(self.generation_id),
                "job_id": str(self.job_id),
                "stage": self.stage.value,
                "shard_key": self.shard_key,
                "source_sha256": self.source_sha256,
                "input_sha256": self.input_sha256,
                "pipeline_fingerprint": self.pipeline_fingerprint,
                "outputs": [
                    asdict(o) for o in sorted(self.outputs, key=lambda o: o.role)
                ],
                "vector_ids": sorted(self.vector_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @property
    def sha256(self) -> str:
        """Hash the exact UTF-8 bytes returned by to_json for checkpoint registration."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def vector_ids_sha256(self) -> str:
        """Hash the exact ID set using version-1 sorted JSON-array encoding.

        Do not concatenate strings without boundaries. This digest covers only
        this batch; generation-wide evidence must cover the complete inventory.
        """
        encoded = json.dumps(
            sorted(self.vector_ids), separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def validate_checkpoint(self, checkpoint: StageCheckpoint) -> None:
        """Match manifest identity and canonical bytes to a checkpoint record.

        A storage reader must also hash the actual downloaded bytes and inspect
        listed outputs. This in-memory comparison does not perform those reads.
        """
        if not isinstance(checkpoint, StageCheckpoint):
            raise ValueError("checkpoint must be a StageCheckpoint")
        for name in (
            "tenant_id",
            "document_id",
            "generation_id",
            "job_id",
            "stage",
            "shard_key",
            "source_sha256",
            "input_sha256",
            "pipeline_fingerprint",
        ):
            if getattr(self, name) != getattr(checkpoint, name):
                raise ValueError("Manifest does not match checkpoint work identity")
        if (
            self.sha256 != checkpoint.manifest_sha256
            or len(self.to_json().encode("utf-8")) != checkpoint.manifest_size_bytes
        ):
            raise ValueError("Manifest bytes do not match checkpoint hash or size")
