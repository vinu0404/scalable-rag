"""AWS Textract as a user-supplied Recognizer (live test only; not in the package).

Textract reads multi-page PDFs from S3 through its asynchronous API, so this
adapter declares unit="document", call_style="job", inputs={"storage_ref"}.
It never sees credentials beyond the standard AWS chain.

- submit(): StartDocumentTextDetection on the exact pinned S3 version, with
  ClientRequestToken = the package's deterministic request_id, so a repeated
  submit for the same request returns the same Textract job (no double bill).
- poll(): GetDocumentTextDetection, reading every result page (NextToken),
  assembling LINE blocks per page in reading order, and mapping Textract's
  positional page i to job.page_numbers[i-1]. A page-count disagreement is
  reported as a failure rather than guessed.
"""

import json

import boto3

from rag_ingest.adapters.aws.common import call, sdk_config
from rag_ingest.contracts.errors import (
    OperatorActionRequiredError,
    PermanentInputError,
    RateLimitedError,
    RetryableError,
)
from rag_ingest.contracts.recognition import (
    JobPoll,
    PageFailure,
    RecognitionResult,
    RecognizedPage,
    RecognizerCapabilities,
)
from rag_ingest.domain.pipeline import ComponentSpec

THROTTLED = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "LimitExceededException",
}
OPERATOR = {
    "AccessDeniedException",
    "InvalidS3ObjectException",
    "InvalidKMSKeyException",
}


class TextractRecognizer:
    def __init__(self, *, region, session=None, max_requests_per_second=1.0):
        self.client = (session or boto3.Session()).client(
            "textract", region_name=region, config=sdk_config()
        )
        self._spec = ComponentSpec(
            driver="aws_textract_text",
            implementation_version="1",
            options_json=json.dumps(
                {"api": "StartDocumentTextDetection", "assembly": "lines-by-page"}
            ),
        )
        self._caps = RecognizerCapabilities(
            unit="document",
            call_style="job",
            inputs=frozenset({"storage_ref"}),
            output_format="text",
            max_pages_per_request=3000,
            max_bytes_per_request=500 * 1024**2,
            max_requests_per_second=max_requests_per_second,
        )

    @property
    def spec(self):
        return self._spec

    @property
    def capabilities(self):
        return self._caps

    async def _call(self, method, *, write=False, **kwargs):
        """Use the package's AWS classification, plus Textract-specific codes."""
        try:
            return await call(self.client, method, write=write, **kwargs)
        except PermanentInputError as error:
            code = getattr(error, "provider_code", "")
            if code in THROTTLED:
                raise RateLimitedError("Textract throttled the request") from None
            if code in OPERATOR:
                raise OperatorActionRequiredError(
                    "Textract cannot read the object: check IAM, region and KMS"
                ) from None
            raise

    async def recognize(self, request):
        raise NotImplementedError("Textract multi-page PDFs use the job API")

    async def submit(self, request):
        ref = request.input.ref
        response = await self._call(
            "start_document_text_detection",
            write=True,
            DocumentLocation={
                "S3Object": {
                    "Bucket": ref.bucket,
                    "Name": ref.key,
                    "Version": ref.version_id,
                }
            },
            ClientRequestToken=request.request_id[:64],
        )
        return response["JobId"]

    async def poll(self, job):
        blocks, token, first = [], None, None
        while True:
            args = {"JobId": job.job_id, "MaxResults": 1000}
            if token:
                args["NextToken"] = token
            try:
                response = await self._call("get_document_text_detection", **args)
            except PermanentInputError as error:
                if getattr(error, "provider_code", "") == "InvalidJobIdException":
                    return JobPoll(state="expired")
                raise
            first = first or response
            status = response["JobStatus"]
            if status == "IN_PROGRESS":
                return JobPoll(state="pending", retry_after_seconds=5)
            if status == "FAILED":
                return JobPoll(
                    state="failed",
                    error=PermanentInputError(
                        "Textract could not process the document"
                    ),
                )
            blocks.extend(response.get("Blocks", ()))
            token = response.get("NextToken")
            if not token:
                break
        count = first.get("DocumentMetadata", {}).get("Pages")
        if count != len(job.page_numbers):
            return JobPoll(
                state="failed",
                error=PermanentInputError(
                    "Textract page count differs from the request"
                ),
            )
        lines, seen = {}, set()
        for block in blocks:
            if block.get("BlockType") == "PAGE":
                seen.add(block["Page"])
            elif block.get("BlockType") == "LINE":
                lines.setdefault(block["Page"], []).append(block.get("Text", ""))
        pages, failures = [], []
        for position, original in enumerate(job.page_numbers, start=1):
            if position in seen:
                pages.append(
                    RecognizedPage(
                        page_number=original, text="\n".join(lines.get(position, ()))
                    )
                )
            else:  # PARTIAL_SUCCESS: this page was not processed
                failures.append(
                    PageFailure(
                        page_number=original,
                        error=RetryableError("Textract skipped this page"),
                    )
                )
        return JobPoll(
            state="done",
            result=RecognitionResult(pages=tuple(pages), failures=tuple(failures)),
        )

    async def aclose(self):
        import asyncio

        await asyncio.to_thread(self.client.close)
