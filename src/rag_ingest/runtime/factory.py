"""Build the default stack while allowing callers to inject custom adapters."""

from rag_ingest.adapters.postgres.connection import Database
from rag_ingest.adapters.postgres.metadata import PostgresMetadataStore
from rag_ingest.adapters.postgres.vectors import PgVectorStore
from rag_ingest.adapters.aws.s3 import S3ObjectStore
from rag_ingest.adapters.aws.sqs import SQSQueue
from rag_ingest.adapters.processing.text import TextExtractor, CharacterChunker
from rag_ingest.adapters.processing.embedding import OpenAIEmbedder
from dataclasses import replace
from rag_ingest.contracts.recognition import RecognizerCapabilities
from rag_ingest.contracts.storage import CleanupObjectStore
from rag_ingest.domain.pipeline import ComponentSpec
from .engine import IngestionEngine


def build_engine(settings, **overrides):
    """Use built-in S3/SQS/PG adapters unless an explicit replacement is supplied.

    Custom components can be passed as metadata=, objects=, uploads=, queue=,
    extractor=, chunker=, embedder=, vectors=, recognizer=, keyword= (a
    KeywordIndex) or reranker= (a Reranker). They must satisfy contracts. No bucket, queue, database, or index is provisioned.

    A recognizer's spec and capabilities are recorded in the returned engine's
    settings so every recipe built from them identifies the tool.
    """
    allowed = {
        "metadata",
        "objects",
        "uploads",
        "queue",
        "extractor",
        "chunker",
        "embedder",
        "vectors",
        "recognizer",
        "keyword",
        "reranker",
    }
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown adapter override: {', '.join(sorted(unknown))}")
    recognizer = overrides.get("recognizer")
    if recognizer is not None:
        if not isinstance(recognizer.spec, ComponentSpec) or not isinstance(
            recognizer.capabilities, RecognizerCapabilities
        ):
            raise ValueError(
                "Recognizer needs a ComponentSpec spec and RecognizerCapabilities"
            )
        required = (
            ("submit", "poll")
            if recognizer.capabilities.call_style == "job"
            else ("recognize",)
        )
        if not all(callable(getattr(recognizer, name, None)) for name in required):
            raise ValueError(
                "Recognizer does not implement the methods its call_style needs: "
                + ", ".join(required)
            )
        settings = replace(
            settings,
            recognizer_spec=recognizer.spec,
            recognizer_capabilities=recognizer.capabilities,
        )
    core = settings.core
    metadata = overrides.get("metadata") or PostgresMetadataStore(
        Database(core.metadata_dsn, pool_size=settings.metadata_pool_size())
    )
    objects = overrides.get("objects") or S3ObjectStore(
        region=core.s3_region,
        endpoint_url=core.s3_endpoint_url,
        public_endpoint_url=settings.public_s3_endpoint,
    )
    if not isinstance(objects, CleanupObjectStore):
        raise ValueError(
            "Object store must implement CleanupObjectStore "
            "(delete_prefix_versions, delete_all_key_versions); otherwise "
            "deleted documents' vectors and files would never be erased"
        )
    settings.pipeline()  # fail now, not at the first upload, if inconsistent
    uploads = overrides.get("uploads") or objects
    queue = overrides.get("queue") or SQSQueue(
        core.sqs_queue_url, region=core.sqs_region, endpoint_url=core.sqs_endpoint_url
    )
    keyword = overrides.get("keyword")
    if keyword is None and settings.keyword_index == "postgres":
        from rag_ingest.adapters.postgres.keyword import PostgresKeywordIndex

        # Same database and pool as the metadata store: the text lives there.
        database = getattr(metadata, "db", None) or Database(core.metadata_dsn)
        keyword = PostgresKeywordIndex(database, language=settings.keyword_language)
    if settings.search_hybrid and keyword is None:
        raise ValueError(
            "INGEST_SEARCH_HYBRID needs a keyword index: set INGEST_KEYWORD_INDEX=postgres"
            " or inject one with build_engine(keyword=...)"
        )
    reranker = overrides.get("reranker")
    if settings.search_rerank and reranker is None:
        raise ValueError(
            "INGEST_SEARCH_RERANK needs a reranker: build_engine(reranker=...)"
            " (none is bundled; see examples/http_reranker.py)"
        )
    embedder = overrides.get("embedder")
    if embedder is None:
        if settings.embedding_driver == "openai":
            embedder = OpenAIEmbedder(
                api_key=settings.embedding_api_key, base_url=settings.embedding_base_url
            )
        else:
            raise ValueError("A custom embedding driver requires an injected embedder")
    return IngestionEngine(
        metadata=metadata,
        objects=objects,
        uploads=uploads,
        queue=queue,
        extractor=overrides.get("extractor") or TextExtractor(),
        chunker=overrides.get("chunker") or CharacterChunker(),
        embedder=embedder,
        vectors=overrides.get("vectors")
        or PgVectorStore(
            Database(settings.vector_dsn, pool_size=settings.vector_pool_size()),
            **settings.vector_store_options(),
        ),
        recognizer=recognizer,
        keyword=keyword,
        reranker=reranker,
        settings=settings,
    )
