"""Default SQS transport, also configurable for tested compatible endpoints."""

import asyncio
import boto3
from .common import call, sdk_config
from rag_ingest.contracts.queue import QueueDelivery
from rag_ingest.domain.events import JobAvailable


class SQSQueue:
    """Use an existing queue; never create infrastructure during normal startup."""

    def __init__(
        self, queue_url, *, region="ap-south-1", endpoint_url=None, session=None
    ):
        """Resolve credentials through boto3's default chain or an injected session."""
        if queue_url.rstrip("/").endswith(".fifo"):
            raise ValueError("The default adapter requires a standard SQS queue")
        self.queue_url = queue_url
        self.client = (session or boto3.Session()).client(
            "sqs", region_name=region, endpoint_url=endpoint_url, config=sdk_config()
        )
        self.closed = False

    async def publish(self, event):
        """Send the same durable event identity and confirm service acceptance."""
        if not isinstance(event, JobAvailable):
            raise ValueError("JobAvailable is required")
        result = await call(
            self.client,
            "send_message",
            write=True,
            QueueUrl=self.queue_url,
            MessageBody=event.to_json(),
        )
        if not result.get("MessageId"):
            from rag_ingest.contracts.errors import UncertainOutcomeError

            raise UncertainOutcomeError("Queue did not return a message identity")

    async def publish_batch(self, events):
        """Send up to 10 events in one call; one outcome per event (None = accepted).

        A refused entry is retryable unless SQS says the request itself was
        at fault. An entry SQS reports neither way is uncertain. If the call
        itself fails, the error propagates and every event is resent later.
        """
        from rag_ingest.contracts.errors import (
            PermanentInputError,
            RetryableError,
            UncertainOutcomeError,
        )

        if (
            not isinstance(events, tuple)
            or not 1 <= len(events) <= 10
            or any(not isinstance(e, JobAvailable) for e in events)
        ):
            raise ValueError("1..10 JobAvailable events are required")
        result = await call(
            self.client,
            "send_message_batch",
            write=True,
            QueueUrl=self.queue_url,
            Entries=[
                {"Id": str(n), "MessageBody": e.to_json()} for n, e in enumerate(events)
            ],
        )
        accepted = {
            s["Id"] for s in result.get("Successful", ()) if s.get("MessageId")
        }
        refused = {f["Id"]: f for f in result.get("Failed", ())}
        outcomes = []
        for n in range(len(events)):
            entry = refused.get(str(n))
            if str(n) in accepted:
                outcomes.append(None)
            elif entry is None:
                outcomes.append(
                    UncertainOutcomeError("Queue did not report this message")
                )
            elif entry.get("SenderFault"):
                outcomes.append(
                    PermanentInputError(f"Queue refused message: {entry.get('Code')}")
                )
            else:
                outcomes.append(
                    RetryableError(f"Queue could not take message: {entry.get('Code')}")
                )
        return tuple(outcomes)

    async def receive(self, *, max_messages, wait_seconds, visibility_seconds):
        """Long-poll a bounded batch, keeping each current delivery receipt."""
        for value, low, high in (
            (max_messages, 1, 10),
            (wait_seconds, 0, 20),
            (visibility_seconds, 1, 43200),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("SQS receive parameter outside supported range")
        response = await call(
            self.client,
            "receive_message",
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
            VisibilityTimeout=visibility_seconds,
            MessageSystemAttributeNames=["ApproximateReceiveCount"],
        )
        return tuple(
            QueueDelivery(
                message_id=m["MessageId"],
                body=m["Body"],
                receipt_handle=m["ReceiptHandle"],
                receive_count=int(
                    m.get("Attributes", {}).get("ApproximateReceiveCount", 1)
                ),
            )
            for m in response.get("Messages", ())
        )

    async def acknowledge(self, delivery):
        """Delete this delivery only after the runner establishes a durable outcome."""
        await call(
            self.client,
            "delete_message",
            write=True,
            QueueUrl=self.queue_url,
            ReceiptHandle=delivery.receipt_handle,
        )

    async def extend_visibility(self, delivery, *, seconds):
        """Renew queue visibility separately from the PostgreSQL job lease."""
        if type(seconds) is not int or not 0 <= seconds <= 43200:
            raise ValueError("Invalid SQS visibility duration")
        await call(
            self.client,
            "change_message_visibility",
            write=True,
            QueueUrl=self.queue_url,
            ReceiptHandle=delivery.receipt_handle,
            VisibilityTimeout=seconds,
        )

    async def doctor(self):
        """Check queue access without consuming or deleting messages."""
        return await call(
            self.client,
            "get_queue_attributes",
            QueueUrl=self.queue_url,
            AttributeNames=["QueueArn", "VisibilityTimeout"],
        )

    async def aclose(self):
        """Release the owned SDK pool without acknowledging pending deliveries."""
        if not self.closed:
            self.closed = True
            await asyncio.to_thread(self.client.close)
