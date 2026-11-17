"""Bound AWS SDK calls and translate provider failures without leaking secrets."""

import asyncio
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    EndpointConnectionError,
    ConnectTimeoutError,
    ReadTimeoutError,
    ConnectionClosedError,
    ParamValidationError,
)
from rag_ingest.contracts.errors import (
    RetryableError,
    RateLimitedError,
    UncertainOutcomeError,
    OperatorActionRequiredError,
    PermanentInputError,
)


def sdk_config():
    """Keep SDK retries explicit so core retries do not multiply hidden loops."""
    return Config(
        signature_version="v4",
        connect_timeout=5,
        read_timeout=25,
        retries={"total_max_attempts": 1, "mode": "standard"},
        max_pool_connections=16,
        s3={"addressing_style": "path"},
    )


async def call(client, method, *, write=False, **kwargs):
    """Run blocking SDK transport off the event loop and classify known failures.

    Cancellation cannot stop a running SDK thread. Callers must treat writes
    cancelled after dispatch as uncertain and let idempotency/reconciliation
    determine their outcome rather than assuming they failed remotely.
    """
    try:
        return await asyncio.to_thread(getattr(client, method), **kwargs)
    except (NoCredentialsError, PartialCredentialsError):
        raise OperatorActionRequiredError("AWS credentials are unavailable") from None
    except (EndpointConnectionError, ConnectTimeoutError):
        raise RetryableError(
            "Storage/queue connection could not be established"
        ) from None
    except (ReadTimeoutError, ConnectionClosedError):
        error = UncertainOutcomeError if write else RetryableError
        raise error("Provider response was lost") from None
    except ParamValidationError:
        raise ValueError("Provider request parameters are invalid") from None
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if code in (
            "AccessDenied",
            "InvalidAccessKeyId",
            "SignatureDoesNotMatch",
            "ExpiredToken",
            "InvalidClientTokenId",
            "AuthorizationError",
        ) or status in (401, 403):
            raise OperatorActionRequiredError("Provider access was rejected") from None
        if status == 429 or code in (
            "SlowDown",
            "Throttling",
            "ThrottlingException",
            "RequestThrottled",
        ):
            hint = (
                exc.response.get("ResponseMetadata", {})
                .get("HTTPHeaders", {})
                .get("retry-after")
            )
            try:
                delay = max(0, float(hint)) if hint is not None else None
            except ValueError:
                delay = None
            raise RateLimitedError(
                "Provider throttled the request", retry_after_seconds=delay
            ) from None
        if status >= 500:
            error = UncertainOutcomeError if write else RetryableError
            raise error("Provider service failed during the operation") from None
        # Callers need these safe machine codes for absence/precondition paths.
        error = PermanentInputError("Provider rejected the operation")
        error.provider_code = code
        error.status_code = status
        raise error from None
