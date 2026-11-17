"""Encode known domain records as JSON without using pickle or importing user code."""

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID
from importlib import import_module

from rag_ingest.contracts.errors import RecordVersionError
from rag_ingest.contracts.storage import ObjectRef

_MODULES = (
    "documents",
    "jobs",
    "uploads",
    "artifacts",
    "checkpoints",
    "pipeline",
    "policy",
    "events",
    "outbox",
    "multipart",
    "manifests",
    "inventory",
    "chunks",
    "embeddings",
    "enums",
)
_REGISTRY = {}
for _name in _MODULES:
    for _cls in vars(import_module("rag_ingest.domain." + _name)).values():
        if isinstance(_cls, type) and (is_dataclass(_cls) or issubclass(_cls, Enum)):
            _REGISTRY[_cls.__name__] = _cls

_REGISTRY["ObjectRef"] = ObjectRef


def encode(value):
    """Convert supported records into JSON values while preserving types."""
    if isinstance(value, Enum):
        return {"$enum": type(value).__name__, "value": value.value}
    if isinstance(value, UUID):
        return {"$uuid": str(value)}
    if isinstance(value, datetime):
        return {"$time": value.isoformat()}
    if is_dataclass(value):
        return {
            "$record": type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, tuple):
        return {"$tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    return value


def _newer(what):
    """Report version skew; never drop data a newer release wrote."""
    return RecordVersionError(
        f"Saved {what} was written by a newer rag-ingest release; "
        "finish the upgrade before running this version"
    )


def decode(value):
    """Restore only allowlisted types; constructors revalidate saved records.

    Rows from older releases decode when added fields have defaults. Rows
    from newer releases are refused: callers read-modify-write whole payloads,
    so ignoring an unknown field would silently erase it on the next save.
    """
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "$record" in value:
        cls = _REGISTRY.get(value["$record"])
        if cls is None:
            raise _newer("record type")
        if set(value["fields"]) - {f.name for f in fields(cls)}:
            raise _newer(cls.__name__)
        return cls(**{k: decode(v) for k, v in value["fields"].items()})
    if "$enum" in value:
        cls = _REGISTRY.get(value["$enum"])
        if cls is None:
            raise _newer("enum type")
        try:
            return cls(value["value"])
        except ValueError:
            raise _newer(cls.__name__ + " value") from None
    if "$uuid" in value:
        return UUID(value["$uuid"])
    if "$time" in value:
        return datetime.fromisoformat(value["$time"])
    if "$tuple" in value:
        return tuple(decode(v) for v in value["$tuple"])
    return {k: decode(v) for k, v in value.items()}
