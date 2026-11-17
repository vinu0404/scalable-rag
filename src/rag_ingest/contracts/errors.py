"""Give the core shared failure categories across different adapters.

Adapters must translate known provider failures into these categories using
safe messages. Do not copy credentials, document text, or raw provider
responses into messages. These classes do not redact messages automatically.
Unknown programming errors should remain visible instead of being blindly
converted into retryable failures.
"""

import math


class IngestionError(Exception):
    """Base class for expected ingestion failures the application can handle.

    Catch a specific subclass to decide what to do. Catching this base class
    does not establish that retrying, acknowledging, or failing a job is safe.
    """


class AdapterError(IngestionError):
    """Report a known failure while communicating with an external component.

    The adapter must classify its cause and outcome before choosing a subclass.
    """


class RetryableError(AdapterError):
    """Report a temporary failure eligible for the core's bounded retry policy.

    retry_after_seconds is optional provider guidance for the minimum delay.
    The scheduler must also enforce attempt limits and overall deadlines.
    Only use this category when repeating the operation is safe under its
    contract. Unknown write outcomes belong in UncertainOutcomeError instead.
    """

    def __init__(
        self, message: str, *, retry_after_seconds: float | None = None
    ) -> None:
        """Keep a safe diagnostic message and an optional finite retry delay.

        Reject negative, nonnumeric, or infinite delays. Constructing this
        exception does not sleep or schedule another attempt.
        """
        if retry_after_seconds is not None:
            try:
                valid = (
                    type(retry_after_seconds) in (int, float)
                    and math.isfinite(retry_after_seconds)
                    and retry_after_seconds >= 0
                )
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("retry_after_seconds must be finite and nonnegative")
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class RateLimitedError(RetryableError):
    """Report provider throttling so the core can delay and reduce pressure.

    Example: an embedding request is rejected because its quota is exhausted.
    Convert supported provider delay guidance into retry_after_seconds.
    """


class UncertainOutcomeError(AdapterError):
    """Report that a remote operation may have succeeded despite a lost response.

    Example: a vector upsert times out after sending its request. The caller
    must verify the outcome or repeat an operation with safe stable identities.
    This is deliberately separate from RetryableError to require explicit
    handling. A repeated embedding request may incur another provider charge.
    """


class OperatorActionRequiredError(AdapterError):
    """Report a dependency problem that needs intervention, such as bad credentials.

    The runtime must expose the problem and control affected work rather than
    endlessly retrying every document or labeling their content invalid.
    """


class RecordVersionError(OperatorActionRequiredError):
    """Report a saved record written by a newer release than this process.

    Stop this process without changing the record or failing its job; a
    process running the newer release can continue the work.
    """


class PermanentInputError(IngestionError):
    """Report input that cannot succeed unchanged, such as an unsupported file.

    Record a visible terminal outcome using the current lease. A corrected
    upload can be submitted separately; do not endlessly retry this input.
    """


class DuplicateContentError(PermanentInputError):
    """Refuse a file whose exact bytes a live document in this tenant already has.

    document_id and filename name that existing document. Nothing was
    uploaded, or the stored copy was not accepted. Send allow_duplicate to
    keep a second copy on purpose.
    """

    code = "duplicate_content"

    def __init__(self, document_id, filename):
        super().__init__("Identical content is already a live document")
        self.document_id = document_id
        self.filename = filename


class LeaseLostError(IngestionError):
    """Report that this worker attempt no longer owns the job.

    Stop authoritative progress writes and publication for this attempt.
    Do not mark the job failed using an expired or replaced lease. Queue
    disposition must follow the runner's recovery policy and durable state.
    """


class AwaitingRecognitionError(IngestionError):
    """Report that a stage submitted work to a job-style tool and must wait.

    Not a failure: the runner parks the job (no lease, no attempt spent) and
    the scheduler re-queues it once every recorded provider job finishes.
    """


def with_code(error, code):
    """Attach a safe machine failure code (letters, digits, _ and -) to an error.

    Runners record it as the job's failure code instead of a generic one.
    """
    if (
        not isinstance(code, str)
        or not 1 <= len(code) <= 100
        or any(not (ch.isalnum() or ch in "_-") for ch in code)
    ):
        raise ValueError("Invalid failure code")
    error.code = code
    return error
