"""Decode a bounded stage manifest and bind it to its saved checkpoint.

The caller downloads bytes with a streaming size limit first. This reader
does not access storage or verify the output files listed by the manifest.
"""

import hashlib
import json
from dataclasses import dataclass, fields
from uuid import UUID

from rag_ingest.domain.checkpoints import StageCheckpoint
from rag_ingest.domain.enums import ProcessingStage
from rag_ingest.domain.limits import MAX_MANIFEST_BYTES
from rag_ingest.domain.manifests import ManifestOutput, StageManifest


@dataclass(frozen=True, kw_only=True)
class ManifestReadLimits:
    """Bound manifest bytes and inventory entries before constructing records.

    These are initial configurable limits, not measured capacity claims.
    Large pipelines must use bounded batches instead of enormous manifests.
    """

    max_bytes: int = MAX_MANIFEST_BYTES
    max_outputs: int = 128
    max_vector_ids: int = 4096

    def __post_init__(self) -> None:
        """Reject nonpositive or noninteger limits."""
        for name in ("max_bytes", "max_outputs", "max_vector_ids"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys rather than silently keeping the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Manifest JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    """Reject nonstandard JSON numbers such as NaN and Infinity."""
    raise ValueError("Manifest JSON contains a nonstandard numeric constant")


def read_stage_manifest(
    data: bytes,
    *,
    checkpoint: StageCheckpoint,
    limits: ManifestReadLimits,
) -> StageManifest:
    """Check saved bytes, decode version 1, and return a matching manifest.

    First enforce byte length and checkpoint SHA-256. Then reject duplicate
    keys, unknown fields, unsupported manifest versions, and excessive entry
    counts. Construct domain records only after checking JSON structure.

    Require the canonical encoding produced by StageManifest.to_json; merely
    reformatting saved JSON changes the bytes bound to its checkpoint. Errors
    do not repeat raw payload values. A successful return does not establish
    that listed outputs exist, that their formats are supported, or that any
    vectors are queryable. Those checks follow in the recovery service.
    """
    if not isinstance(data, bytes):
        raise ValueError("Manifest input must be bytes")
    if not isinstance(checkpoint, StageCheckpoint) or not isinstance(
        limits, ManifestReadLimits
    ):
        raise ValueError("A checkpoint and manifest read limits are required")
    if len(data) > limits.max_bytes:
        raise ValueError("Manifest exceeds the configured byte limit")
    if len(data) != checkpoint.manifest_size_bytes:
        raise ValueError("Manifest byte count does not match checkpoint")
    if hashlib.sha256(data).hexdigest() != checkpoint.manifest_sha256:
        raise ValueError("Manifest byte hash does not match checkpoint")

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, RecursionError):
        raise ValueError(
            "Manifest must contain valid UTF-8 JSON without duplicate keys"
        ) from None

    expected = {
        "schema_version",
        "tenant_id",
        "document_id",
        "generation_id",
        "job_id",
        "stage",
        "shard_key",
        "source_sha256",
        "input_sha256",
        "pipeline_fingerprint",
        "outputs",
        "vector_ids",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Manifest has missing or unexpected fields")
    version = payload["schema_version"]
    if type(version) is not int or version != StageManifest.SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema version")
    outputs = payload["outputs"]
    vector_ids = payload["vector_ids"]
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= limits.max_outputs:
        raise ValueError("Manifest output count is invalid or exceeds its limit")
    if not isinstance(vector_ids, list) or len(vector_ids) > limits.max_vector_ids:
        raise ValueError("Manifest vector-ID count is invalid or exceeds its limit")
    output_fields = {item.name for item in fields(ManifestOutput)}
    for output in outputs:
        if not isinstance(output, dict) or set(output) != output_fields:
            raise ValueError("Manifest output has missing or unexpected fields")

    try:
        identities = {}
        for name in ("tenant_id", "document_id", "generation_id", "job_id"):
            if not isinstance(payload[name], str):
                raise ValueError
            identities[name] = UUID(payload[name])
        manifest = StageManifest(
            **identities,
            stage=ProcessingStage(payload["stage"]),
            shard_key=payload["shard_key"],
            source_sha256=payload["source_sha256"],
            input_sha256=payload["input_sha256"],
            pipeline_fingerprint=payload["pipeline_fingerprint"],
            outputs=tuple(ManifestOutput(**output) for output in outputs),
            vector_ids=tuple(vector_ids),
        )
        manifest.validate_checkpoint(checkpoint)
    except (ValueError, TypeError, AttributeError, RecursionError, OverflowError):
        raise ValueError(
            "Manifest fields or canonical encoding do not match the checkpoint"
        ) from None
    return manifest
