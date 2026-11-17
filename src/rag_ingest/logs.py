"""Structured, secret-safe logging for the package and its service commands.

The library only emits records (each module uses logging.getLogger(__name__)
under the "rag_ingest" hierarchy) and never installs handlers. Applications
own configuration: call configure_logging() or use your own logging setup;
structured fields are on each record as `event` and `fields`.

Exception output keeps the stack (file, line, function) and exception types,
but only shows messages of this package's own errors, whose text is fixed.
Third-party messages can carry DSNs, signed URLs or document content.
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone

from rag_ingest.contracts.errors import IngestionError

_HANDLER_MARK = "_rag_ingest_handler"


def log_event(logger, name, level=logging.INFO, exc_info=None, **fields):
    """Emit one named event with safe string fields for text or JSON output."""
    values = {key: str(value) for key, value in fields.items()}
    text = " ".join([name, *(f"{k}={v}" for k, v in values.items())])
    logger.log(level, text, exc_info=exc_info, extra={"event": name, "fields": values})


def _safe_exception(exc_info):
    """Describe an exception chain without third-party messages."""
    if exc_info is True:
        exc_info = sys.exc_info()
    error = exc_info[1] if isinstance(exc_info, tuple) else exc_info
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(
            {
                "type": type(error).__qualname__,
                "message": str(error) if isinstance(error, IngestionError) else None,
                "stack": [
                    f"{frame.filename}:{frame.lineno} in {frame.name}"
                    for frame in traceback.extract_tb(error.__traceback__)
                ],
            }
        )
        error = error.__cause__ or (
            None if error.__suppress_context__ else error.__context__
        )
    return chain


def _timestamp(record):
    return (
        datetime.fromtimestamp(record.created, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class JsonFormatter(logging.Formatter):
    """One JSON object per line for log pipelines (Loki, CloudWatch, Datadog)."""

    def format(self, record):
        entry = {
            "ts": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", None) or record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exception"] = _safe_exception(record.exc_info)
        return json.dumps(entry, default=str)


class TextFormatter(logging.Formatter):
    """Readable single-line records plus a redacted stack for local use."""

    def format(self, record):
        line = (
            f"{_timestamp(record)} {record.levelname:<7} {record.name}: "
            f"{record.getMessage()}"
        )
        if record.exc_info:
            for part in _safe_exception(record.exc_info):
                detail = f": {part['message']}" if part["message"] else ""
                line += f"\n  {part['type']}{detail}"
                line += "".join(f"\n    at {frame}" for frame in part["stack"])
        return line


def configure_logging(level="INFO", fmt="json", stream=None):
    """Install one handler on the root logger for a service process.

    Safe to call again: it replaces only the handler it installed earlier.
    Libraries embedding rag_ingest should configure logging themselves.
    """
    if fmt not in ("json", "text"):
        raise ValueError("INGEST_LOG_FORMAT must be 'json' or 'text'")
    numeric = logging.getLevelName(str(level).upper())
    if not isinstance(numeric, int):
        raise ValueError("INGEST_LOG_LEVEL must be DEBUG, INFO, WARNING or ERROR")
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    setattr(handler, _HANDLER_MARK, True)
    root.addHandler(handler)
    root.setLevel(numeric)
    return handler
