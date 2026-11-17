"""Run explicit setup commands and independent ingestion service roles."""

import argparse
import asyncio
import logging
import signal
from pathlib import Path
from urllib.parse import urlsplit


def main():
    """Load an optional environment file, then execute the requested role."""
    parser = argparse.ArgumentParser(description="Recoverable document ingestion")
    parser.add_argument(
        "--env-file", help="Explicit dotenv path; existing environment wins"
    )
    parser.add_argument(
        "command",
        choices=[
            "api",
            "serve",
            "worker",
            "publisher",
            "scheduler",
            "migrate",
            "doctor",
            "local-init",
            "search-eval",
            "reindex-keyword",
        ],
    )
    parser.add_argument("--samples", type=int, default=100, help="search-eval: queries to sample")
    parser.add_argument("--k", type=int, default=10, help="search-eval: results per query")
    parser.add_argument(
        "--ef", default="40,100,200", help="search-eval: comma-separated ef_search values"
    )
    args = parser.parse_args()
    if args.env_file:
        from rag_ingest.bootstrap import load_settings

        load_settings(env_file=args.env_file)
    import os
    from rag_ingest.logs import configure_logging, log_event

    configure_logging(
        level=os.environ.get("INGEST_LOG_LEVEL", "INFO"),
        fmt=os.environ.get("INGEST_LOG_FORMAT", "json"),
    )
    from rag_ingest.runtime.settings import ServiceSettings

    settings = ServiceSettings.from_env()
    try:
        if args.command in ("api", "serve"):
            import uvicorn
            from rag_ingest.api.app import create_app

            uvicorn.run(
                create_app(settings=settings, background=args.command == "serve"),
                host=settings.core.api_host,
                port=settings.core.api_port,
                access_log=False,
                log_config=None,  # keep the process-wide format and redaction
            )
        elif args.command == "local-init":
            local_init(settings)
        elif args.command == "migrate":
            migrate(settings)
        elif args.command == "search-eval":
            asyncio.run(search_eval(settings, args))
        elif args.command == "reindex-keyword":
            asyncio.run(reindex(settings))
        else:
            asyncio.run(run(args.command, settings))
    except KeyboardInterrupt:
        pass
    except Exception as error:
        # The formatter keeps the stack but redacts third-party messages (DSNs).
        log_event(
            logging.getLogger("rag_ingest.cli"),
            "command_failed",
            logging.ERROR,
            error,
            category=type(error).__name__,
            hint="check configuration and README troubleshooting",
        )
        raise SystemExit(1) from None


def migrate(settings):
    """Apply versioned metadata DDL and explicitly provision the pgvector target."""
    from alembic.config import Config
    from alembic import command

    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    config.attributes["dsn"] = settings.core.metadata_dsn
    command.upgrade(config, "head")

    async def vectors():
        """Create vector storage only during an operator-requested setup command."""
        from rag_ingest.adapters.postgres.connection import Database
        from rag_ingest.adapters.postgres.vectors import PgVectorStore

        store = PgVectorStore(
            Database(settings.vector_dsn),
            **settings.vector_store_options(),
        )
        try:
            await store.provision()
        finally:
            await store.aclose()

    asyncio.run(vectors())
    print("Metadata migration and vector provisioning completed.")


def local_init(settings):
    """Create local emulator resources only when both endpoints are loopback URLs.

    This guard prevents a local convenience command from provisioning live AWS
    resources accidentally. Production infrastructure must be supplied explicitly.
    """
    from rag_ingest.adapters.aws.s3 import S3ObjectStore
    from rag_ingest.adapters.aws.sqs import SQSQueue

    for endpoint in (settings.core.s3_endpoint_url, settings.core.sqs_endpoint_url):
        if not endpoint or urlsplit(endpoint).hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise ValueError(
                "local-init requires explicit loopback S3 and SQS endpoints"
            )
    storage = S3ObjectStore(
        region=settings.core.s3_region, endpoint_url=settings.core.s3_endpoint_url
    )
    client = storage.client
    try:
        client.head_bucket(Bucket=settings.bucket)
    except client.exceptions.ClientError as error:
        if error.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
            raise
        params = {"Bucket": settings.bucket}
        if settings.core.s3_region != "us-east-1":
            params["CreateBucketConfiguration"] = {
                "LocationConstraint": settings.core.s3_region
            }
        client.create_bucket(**params)
    client.put_bucket_versioning(
        Bucket=settings.bucket, VersioningConfiguration={"Status": "Enabled"}
    )
    client.put_bucket_lifecycle_configuration(
        Bucket=settings.bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "abort-incomplete-transfers",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                }
            ]
        },
    )
    queue = SQSQueue(
        settings.core.sqs_queue_url,
        region=settings.core.sqs_region,
        endpoint_url=settings.core.sqs_endpoint_url,
    )
    result = queue.client.create_queue(
        QueueName=settings.core.sqs_queue_url.rstrip("/").split("/")[-1],
        Attributes={
            "VisibilityTimeout": str(settings.queue_visibility_seconds),
            "MessageRetentionPeriod": "1209600",
        },
    )
    if result["QueueUrl"].rstrip("/") != settings.core.sqs_queue_url.rstrip("/"):
        print(
            "Set INGEST_SQS_QUEUE_URL to the emulator-returned URL: "
            + result["QueueUrl"]
        )
    print(
        "Local bucket versioning, multipart lifecycle, and standard queue configured."
    )


async def reindex(settings):
    """Index every published build in the configured keyword index."""
    from rag_ingest.runtime.factory import build_engine
    from rag_ingest.runtime.keyword_reindex import reindex_keyword

    engine = build_engine(settings)
    try:
        result = await reindex_keyword(engine)
        print(f"indexed {result['chunks']} chunks of {result['builds']} published builds")
    finally:
        await engine.aclose()


async def search_eval(settings, args):
    """Print recall@k and latency of the vector index against exact search."""
    from rag_ingest.adapters.postgres.connection import Database
    from rag_ingest.adapters.postgres.vectors import PgVectorStore
    from rag_ingest.runtime.search_eval import evaluate, format_report

    ef_values = tuple(int(v) for v in args.ef.split(",") if v.strip())
    store = PgVectorStore(Database(settings.vector_dsn), **settings.vector_store_options())
    try:
        report = await evaluate(store, samples=args.samples, k=args.k, ef_values=ef_values)
        print(format_report(report))
    finally:
        await store.aclose()


async def run(command, settings):
    """Use finite polling loops, stop on signals, and close owned clients."""
    from rag_ingest.runtime.factory import build_engine
    from rag_ingest.runtime.runners import run_publisher, run_scheduler, run_workers

    if command == "doctor":
        from rag_ingest.adapters.postgres.connection import Database
        from rag_ingest.adapters.postgres.metadata import PostgresMetadataStore
        from rag_ingest.adapters.postgres.vectors import PgVectorStore
        from rag_ingest.adapters.aws.s3 import S3ObjectStore
        from rag_ingest.adapters.aws.sqs import SQSQueue
        from uuid import UUID

        metadata = PostgresMetadataStore(Database(settings.core.metadata_dsn))
        vectors = PgVectorStore(
            Database(settings.vector_dsn),
            **settings.vector_store_options(),
        )
        objects = S3ObjectStore(
            region=settings.core.s3_region, endpoint_url=settings.core.s3_endpoint_url
        )
        queue = SQSQueue(
            settings.core.sqs_queue_url,
            region=settings.core.sqs_region,
            endpoint_url=settings.core.sqs_endpoint_url,
        )
        try:
            await metadata.now(UUID(int=0))
            await metadata.tenants()  # Also proves that migrations were applied.
            await vectors.doctor()
            await objects.doctor(settings.bucket)
            await queue.doctor()
            if settings.embedding_driver == "openai" and not settings.embedding_api_key:
                raise ValueError("Embedding API key missing")
            print(
                "SQL schema, vector dimensions, versioned bucket, and queue access passed."
            )
            print(
                "Embedding credentials were checked for presence; no paid request was sent."
            )
        finally:
            for item in (metadata, vectors, objects, queue):
                await item.aclose()
        return
    engine = build_engine(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    task = asyncio.create_task(
        {"publisher": run_publisher, "scheduler": run_scheduler, "worker": run_workers}[
            command
        ](engine, stop)
    )
    waiter = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait(
            (task, waiter), return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            await task
    finally:
        stop.set()
        task.cancel()
        waiter.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)
        await engine.aclose()
