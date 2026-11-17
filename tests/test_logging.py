"""Logging: structured records, levels, safe exception output and CLI wiring."""

import io
import json
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path

import rag_ingest
from rag_ingest.contracts.errors import RetryableError
from rag_ingest.logs import JsonFormatter, TextFormatter, configure_logging, log_event
from rag_ingest.runtime import runners

SECRET = "postgresql://user:topsecret@db.internal/prod"


def _record(exc=None, level=logging.ERROR):
    """Capture one record emitted through the package's event helper."""
    logger = logging.getLogger("rag_ingest.test")
    with unittest.TestCase().assertLogs(logger, level=logging.DEBUG) as logs:
        log_event(logger, "job_failed", level, exc, job_id="j-1", code="c")
    return logs.records[0]


def _raised(error, cause=None):
    try:
        try:
            raise cause if cause else error
        except Exception as inner:
            if cause:
                raise error from inner
            raise
    except Exception as caught:
        return caught


class LoggingTests(unittest.TestCase):
    def test_package_installs_only_a_null_handler(self):
        handlers = logging.getLogger(rag_ingest.__name__).handlers
        self.assertTrue(any(isinstance(h, logging.NullHandler) for h in handlers))

    def test_json_line_has_structured_fields(self):
        entry = json.loads(JsonFormatter().format(_record()))
        self.assertEqual(entry["event"], "job_failed")
        self.assertEqual((entry["job_id"], entry["code"]), ("j-1", "c"))
        self.assertEqual(entry["level"], "ERROR")
        self.assertEqual(entry["logger"], "rag_ingest.test")
        self.assertTrue(entry["ts"].endswith("Z"))

    def test_plain_formatters_still_see_fields_in_the_message(self):
        self.assertEqual(_record().getMessage(), "job_failed job_id=j-1 code=c")

    def test_third_party_exception_message_is_redacted_but_stack_kept(self):
        error = _raised(ValueError(SECRET))
        for formatter in (JsonFormatter(), TextFormatter()):
            output = formatter.format(_record(error))
            self.assertNotIn("topsecret", output)
            self.assertIn("ValueError", output)
            self.assertIn("test_logging.py", output)  # stack frames present

    def test_package_error_message_and_cause_chain_are_shown(self):
        error = _raised(
            RetryableError("PostgreSQL connection failed"), KeyError(SECRET)
        )
        chain = json.loads(JsonFormatter().format(_record(error)))["exception"]
        self.assertEqual([c["type"] for c in chain], ["RetryableError", "KeyError"])
        self.assertEqual(chain[0]["message"], "PostgreSQL connection failed")
        self.assertIsNone(chain[1]["message"])

    def test_configure_logging_is_idempotent_and_validated(self):
        root = logging.getLogger()
        before, level = list(root.handlers), root.level
        try:
            configure_logging("DEBUG", "text", io.StringIO())
            configure_logging("WARNING", "json", io.StringIO())
            ours = [h for h in root.handlers if h not in before]
            self.assertEqual(len(ours), 1)
            self.assertIsInstance(ours[0].formatter, JsonFormatter)
            self.assertEqual(root.level, logging.WARNING)
            with self.assertRaises(ValueError):
                configure_logging("LOUD", "json")
            with self.assertRaises(ValueError):
                configure_logging("INFO", "xml")
        finally:
            for handler in [h for h in root.handlers if h not in before]:
                root.removeHandler(handler)
            root.setLevel(level)

    def test_runner_events_carry_their_severity(self):
        with self.assertLogs("rag_ingest", level=logging.DEBUG) as logs:
            runners.event("job_unexpected_error", logging.ERROR, _raised(KeyError("k")))
            runners.event("delivery_acknowledged", logging.DEBUG, job_id="j")
        error, ack = logs.records
        self.assertEqual((error.levelno, ack.levelno), (logging.ERROR, logging.DEBUG))
        self.assertIsNotNone(error.exc_info)
        self.assertEqual(error.name, "rag_ingest.runtime.runners")


class CliLoggingTests(unittest.TestCase):
    def test_failed_command_logs_json_without_leaking_the_dsn(self):
        env = {
            **os.environ,
            "INGEST_METADATA_DSN": "postgresql://user:topsecret@127.0.0.1:1/x_test",
            "INGEST_S3_BUCKET": "b",
            "INGEST_SQS_QUEUE_URL": "http://localhost:4566/000000000000/q",
            "INGEST_EMBEDDING_DRIVER": "openai",
            "INGEST_LOG_FORMAT": "json",
        }
        result = subprocess.run(
            [sys.executable, "-m", "rag_ingest", "doctor"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("topsecret", result.stderr + result.stdout)
        lines = [json.loads(line) for line in result.stderr.splitlines() if line]
        failed = [e for e in lines if e["event"] == "command_failed"]
        self.assertEqual(len(failed), 1, result.stderr)
        self.assertEqual(failed[0]["level"], "ERROR")
        self.assertTrue(failed[0]["exception"][0]["stack"])


if __name__ == "__main__":
    unittest.main()
