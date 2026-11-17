"""Infrastructure settings. Loading settings never opens connections."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from rag_ingest.domain.policy import ExecutionPolicy, RetryPolicy


def _validate_url(name: str, value: str, schemes: set[str]) -> None:
    """Validate an address without including credentials in error messages."""
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in schemes and bool(parsed.hostname)
        port = parsed.port
        valid = valid and (port is None or 1 <= port <= 65535)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a valid URL with an allowed scheme")


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Hold connection settings and initial execution-policy defaults.

    The service must copy policy into durable work records at acceptance.
    Reading new environment values must not reset existing job budgets.
    """

    metadata_dsn: str = field(repr=False)
    s3_bucket: str
    sqs_queue_url: str = field(repr=False)
    s3_region: str = "ap-south-1"
    sqs_region: str = "ap-south-1"
    s3_endpoint_url: str | None = field(default=None, repr=False)
    sqs_endpoint_url: str | None = field(default=None, repr=False)
    # All interfaces, as containers need; set INGEST_API_HOST=127.0.0.1 locally.
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 8080
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    max_generation_seconds: float = 3600.0

    def __post_init__(self) -> None:
        """Reject invalid addresses, ports, and execution-policy settings."""
        _validate_url(
            "INGEST_METADATA_DSN", self.metadata_dsn, {"postgresql", "postgres"}
        )
        _validate_url("INGEST_SQS_QUEUE_URL", self.sqs_queue_url, {"http", "https"})
        for name, value in (
            ("INGEST_S3_ENDPOINT_URL", self.s3_endpoint_url),
            ("INGEST_SQS_ENDPOINT_URL", self.sqs_endpoint_url),
        ):
            if value is not None:
                _validate_url(name, value, {"http", "https"})
        for name, value in (
            ("INGEST_S3_BUCKET", self.s3_bucket),
            ("INGEST_S3_REGION", self.s3_region),
            ("INGEST_SQS_REGION", self.sqs_region),
            ("INGEST_API_HOST", self.api_host),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be blank")
        if type(self.api_port) is not int or not 1 <= self.api_port <= 65535:
            raise ValueError("INGEST_API_PORT must be an integer between 1 and 65535")
        if not isinstance(self.execution_policy, ExecutionPolicy):
            raise ValueError("execution_policy must be an ExecutionPolicy")
        try:
            valid = (
                type(self.max_generation_seconds) in (int, float)
                and math.isfinite(self.max_generation_seconds)
                and self.max_generation_seconds > 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(
                "INGEST_MAX_GENERATION_SECONDS must be finite and positive"
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Read process environment, or an explicitly supplied mapping."""
        source = os.environ if env is None else env

        def required(name: str) -> str:
            """Read a mandatory value, reporting only its name when missing."""
            value = source.get(name, "").strip()
            if not value:
                raise ValueError(f"Missing required setting: {name}")
            return value

        def optional(name: str) -> str | None:
            """Treat a blank optional address or region as unset."""
            return source.get(name, "").strip() or None

        def integer(name: str, default: int) -> int:
            """Read an integer override; reject invalid or blank numeric values."""
            try:
                return int(source.get(name, str(default)))
            except ValueError:
                raise ValueError(f"{name} must be an integer") from None

        def number(name: str, default: float) -> float:
            """Read a numeric override; domain policy validation checks its range."""
            try:
                return float(source.get(name, str(default)))
            except ValueError:
                raise ValueError(f"{name} must be a number") from None

        defaults = ExecutionPolicy()
        execution_policy = ExecutionPolicy(
            retry=RetryPolicy(
                max_attempts=integer(
                    "INGEST_MAX_ATTEMPTS", defaults.retry.max_attempts
                ),
                base_delay_seconds=number(
                    "INGEST_RETRY_BASE_SECONDS", defaults.retry.base_delay_seconds
                ),
                max_delay_seconds=number(
                    "INGEST_RETRY_CAP_SECONDS", defaults.retry.max_delay_seconds
                ),
            ),
            max_job_seconds=number("INGEST_MAX_JOB_SECONDS", defaults.max_job_seconds),
            max_attempt_seconds=number(
                "INGEST_MAX_ATTEMPT_SECONDS", defaults.max_attempt_seconds
            ),
            lease_seconds=integer("INGEST_LEASE_SECONDS", defaults.lease_seconds),
            heartbeat_seconds=number(
                "INGEST_HEARTBEAT_SECONDS", defaults.heartbeat_seconds
            ),
        )

        region = optional("AWS_DEFAULT_REGION") or "ap-south-1"
        return cls(
            metadata_dsn=required("INGEST_METADATA_DSN"),
            s3_bucket=required("INGEST_S3_BUCKET"),
            sqs_queue_url=required("INGEST_SQS_QUEUE_URL"),
            s3_region=optional("INGEST_S3_REGION") or region,
            sqs_region=optional("INGEST_SQS_REGION") or region,
            s3_endpoint_url=optional("INGEST_S3_ENDPOINT_URL"),
            sqs_endpoint_url=optional("INGEST_SQS_ENDPOINT_URL"),
            api_host=source.get("INGEST_API_HOST", "0.0.0.0"),  # noqa: S104
            api_port=integer("INGEST_API_PORT", 8080),
            execution_policy=execution_policy,
            max_generation_seconds=number("INGEST_MAX_GENERATION_SECONDS", 3600.0),
        )
