"""Runtime options for the ready-to-run service; core adapters remain injectable."""

import json
import os
from dataclasses import dataclass, field
from uuid import UUID
from rag_ingest.config import Settings
from rag_ingest.domain.limits import (
    MAX_BATCH_ITEMS,
    MULTIPART_MAX_PART_BYTES,
    MULTIPART_MIN_PART_BYTES,
)
from rag_ingest.domain import extraction_recipe
from rag_ingest.domain.pipeline import ComponentSpec, PipelineConfig


@dataclass(frozen=True, kw_only=True)
class ServiceSettings:
    """Use AWS S3/SQS and PostgreSQL/pgvector unless adapters are injected.

    API tokens map to trusted tenants and grant tenant-wide access. Production
    applications may supply their own authenticator for finer document ACLs.
    """

    core: Settings
    vector_dsn: str = field(repr=False)
    bucket: str
    public_s3_endpoint: str | None = None
    embedding_driver: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_encoding: str = "cl100k_base"
    # Runtime limits for one embedding call, not part of the recipe: raise the
    # timeout for slower or self-hosted models, the token budget to match the
    # provider's per-request limit.
    embedding_timeout_seconds: int = 30
    embedding_max_batch_tokens: int = 200_000
    embedding_api_key: str = field(default="", repr=False)
    embedding_base_url: str = "https://api.openai.com/v1"
    api_tokens: dict = field(default_factory=dict, repr=False)
    max_upload_bytes: int = 10 * 1024**3
    multipart_threshold: int = 64 * 1024**2
    multipart_part_bytes: int = 64 * 1024**2
    upload_seconds: int = 3600
    presign_seconds: int = 300
    max_source_disk_bytes: int = 10 * 1024**3
    max_extracted_characters: int = 20_000_000
    max_pages: int = 10000
    chunk_size: int = 1000
    chunk_overlap: int = 100
    batch_size: int = 64
    worker_concurrency: int = 2
    queue_visibility_seconds: int = 120
    max_uncertain_repeats: int = 2
    verify_concurrency: int = 2
    max_active_generations: int = 16
    retention_days: int = 7
    # Recognition recipe (fingerprinted) and runtime limits (not fingerprinted).
    extraction_mode: str = "text_only"
    unreadable_pages: str = "keep_empty"
    min_text_chars: int = 16
    render_dpi: int = 200
    strict_pdf: bool = False
    allow_signed_url: bool = False
    min_confidence: float | None = None
    recognition_concurrency: int = 4
    max_recognized_pages: int = 1000
    tenant_daily_recognized_pages: int = 0  # 0 means no daily limit
    recognition_poll_seconds: int = 15
    # Set by build_engine from the injected Recognizer; never from env.
    recognizer_spec: ComponentSpec | None = None
    recognizer_capabilities: object | None = None
    vector_target: str = "pgvector-default"
    db_pool_size: int = 0  # 0: derive from this process's concurrency
    # Refuse a file whose exact bytes a live document already has
    # (INGEST_ALLOW_DUPLICATES=true turns this off for the whole service).
    refuse_duplicates: bool = True
    # Files of failed/expired uploads are erased this long after expiry.
    abandoned_upload_grace_seconds: int = 3600
    # Per-tenant API limits (0 = unlimited), enforced by a token bucket shared
    # by every API process through PostgreSQL.
    api_uploads_per_minute: int = 600
    api_searches_per_minute: int = 120
    # Serve /docs and /openapi.json (off by default in production).
    api_docs: bool = False
    # Vector index (pgvector HNSW). m and ef_construction take effect when the
    # index is built (rag-ingest migrate); ef_search and max_scan_tuples per query.
    vector_index: str = "hnsw"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 200
    hnsw_max_scan_tuples: int = 20_000
    # Keyword index: "none" (detached) or "postgres" (built-in full-text search).
    # An injected KeywordIndex (build_engine(keyword=...)) replaces the built-in.
    keyword_index: str = "none"
    keyword_language: str = "english"
    # Hybrid search queries the keyword index too and fuses both rankings.
    search_hybrid: bool = False
    search_fusion: str = "rrf"  # "rrf" (ranks) or "weighted" (normalized scores)
    search_rrf_k: int = 60
    search_vector_weight: float = 1.0
    search_keyword_weight: float = 1.0
    search_keyword_timeout_seconds: int = 5
    # strict: a failing retriever fails the search; otherwise the other one answers
    # and the response lists the failed one under "degraded".
    search_strict: bool = False
    # Reranking: reorder the top candidates with an injected Reranker
    # (build_engine(reranker=...)); each request can turn it on or off.
    search_rerank: bool = False
    rerank_top_n: int = 50
    rerank_timeout_ms: int = 1500

    def validate_search(self):
        """Refuse inconsistent search settings at startup."""
        import math

        if self.keyword_index not in ("none", "postgres"):
            raise ValueError("INGEST_KEYWORD_INDEX must be none or postgres")
        if self.search_fusion not in ("rrf", "weighted"):
            raise ValueError("INGEST_SEARCH_FUSION must be rrf or weighted")
        if not 1 <= self.search_rrf_k <= 1000:
            raise ValueError("INGEST_SEARCH_RRF_K must be 1..1000")
        for name, value in (
            ("INGEST_SEARCH_VECTOR_WEIGHT", self.search_vector_weight),
            ("INGEST_SEARCH_KEYWORD_WEIGHT", self.search_keyword_weight),
        ):
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(name + " must be 0..100")
        if self.search_vector_weight + self.search_keyword_weight <= 0:
            raise ValueError("At least one search weight must be positive")
        if not 1 <= self.search_keyword_timeout_seconds <= 120:
            raise ValueError("INGEST_SEARCH_KEYWORD_TIMEOUT_SECONDS must be 1..120")
        if not 1 <= self.rerank_top_n <= 200:
            raise ValueError("INGEST_RERANK_TOP_N must be 1..200")
        if not 50 <= self.rerank_timeout_ms <= 60_000:
            raise ValueError("INGEST_RERANK_TIMEOUT_MS must be 50..60000")

    def vector_store_options(self):
        """Keyword arguments for PgVectorStore; one place for every caller."""
        return dict(
            dimensions=self.embedding_dimensions,
            target_id=self.vector_target,
            embedding_spec=self.pipeline().embedder,
            index=self.vector_index,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
            hnsw_max_scan_tuples=self.hnsw_max_scan_tuples,
        )

    def metadata_pool_size(self):
        """Connections the metadata pool needs for this process's concurrency.

        Per worker: its job's transaction, its lease heartbeat, and one per
        read-ahead load during index/publish. Plus upload verifiers,
        recognition pollers, and the scheduler, publisher, pruner and API.
        A smaller pool makes jobs queue for connections and fail after the
        pool timeout under load, so a configured value below this is refused.
        """
        from rag_ingest.runtime.engine import INDEX_READ_AHEAD

        required = (
            self.worker_concurrency * (2 + INDEX_READ_AHEAD)
            + self.verify_concurrency
            + self.recognition_concurrency
            + 4
        )
        if not self.db_pool_size:
            return required
        if self.db_pool_size < required:
            raise ValueError(
                f"INGEST_DB_POOL_SIZE={self.db_pool_size} is below the {required} "
                "connections this process's concurrency settings need"
            )
        return self.db_pool_size

    def vector_pool_size(self):
        """Each job holds at most one vector connection at a time."""
        return self.worker_concurrency + 2

    @classmethod
    def from_env(cls):
        """Read and validate service options without creating any infrastructure."""
        e = os.environ
        core = Settings.from_env()

        def integer(name, default):
            """Reject blank, noninteger, or nonpositive numeric deployment settings."""
            try:
                v = int(e.get(name, str(default)))
            except ValueError:
                raise ValueError(name + " must be an integer") from None
            if v <= 0:
                raise ValueError(name + " must be positive")
            return v

        def flag(name):
            """Read an explicit true/false setting; reject anything else."""
            value = e.get(name, "false").strip().lower()
            if value not in ("true", "false"):
                raise ValueError(name + " must be true or false")
            return value == "true"

        tokens = json.loads(e.get("INGEST_API_TOKENS", "{}"))
        if not isinstance(tokens, dict) or any(
            not isinstance(k, str) or len(k) < 24 for k in tokens
        ):
            raise ValueError(
                "API tokens must be a JSON object with tokens at least 24 characters long"
            )
        tokens = {k: UUID(v) for k, v in tokens.items()}
        result = cls(
            core=core,
            vector_dsn=e.get("INGEST_VECTOR_DSN") or core.metadata_dsn,
            bucket=core.s3_bucket,
            public_s3_endpoint=e.get("INGEST_S3_PUBLIC_ENDPOINT_URL") or None,
            embedding_driver=e.get("INGEST_EMBEDDING_DRIVER", "openai"),
            embedding_model=e.get("INGEST_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=integer("INGEST_EMBEDDING_DIMENSIONS", 1536),
            embedding_api_key=e.get(
                "INGEST_EMBEDDING_API_KEY", e.get("OPENAI_API_KEY", "")
            ),
            embedding_base_url=e.get(
                "INGEST_EMBEDDING_BASE_URL", "https://api.openai.com/v1"
            ),
            embedding_encoding=e.get("INGEST_EMBEDDING_ENCODING", "cl100k_base"),
            embedding_timeout_seconds=integer("INGEST_EMBEDDING_TIMEOUT_SECONDS", 30),
            embedding_max_batch_tokens=integer(
                "INGEST_EMBEDDING_MAX_BATCH_TOKENS", 200_000
            ),
            api_tokens=tokens,
            max_upload_bytes=integer("INGEST_MAX_UPLOAD_BYTES", 10 * 1024**3),
            multipart_threshold=integer(
                "INGEST_MULTIPART_THRESHOLD_BYTES", 64 * 1024**2
            ),
            multipart_part_bytes=integer("INGEST_MULTIPART_PART_BYTES", 64 * 1024**2),
            upload_seconds=integer("INGEST_UPLOAD_SECONDS", 3600),
            presign_seconds=integer("INGEST_PRESIGN_SECONDS", 300),
            max_source_disk_bytes=integer("INGEST_MAX_SOURCE_DISK_BYTES", 10 * 1024**3),
            max_extracted_characters=integer(
                "INGEST_MAX_EXTRACTED_CHARACTERS", 20_000_000
            ),
            max_pages=integer("INGEST_MAX_PAGES", 10000),
            chunk_size=integer("INGEST_CHUNK_SIZE", 1000),
            chunk_overlap=int(e.get("INGEST_CHUNK_OVERLAP", "100")),
            batch_size=integer("INGEST_EMBEDDING_BATCH_SIZE", 64),
            worker_concurrency=integer("INGEST_WORKER_CONCURRENCY", 2),
            queue_visibility_seconds=integer("INGEST_SQS_VISIBILITY_SECONDS", 120),
            vector_target=e.get("INGEST_VECTOR_TARGET", "pgvector-default"),
            max_uncertain_repeats=int(e.get("INGEST_MAX_UNCERTAIN_REPEATS", "2")),
            verify_concurrency=integer("INGEST_VERIFY_CONCURRENCY", 2),
            max_active_generations=integer("INGEST_MAX_ACTIVE_GENERATIONS", 16),
            retention_days=integer("INGEST_RETENTION_DAYS", 7),
            extraction_mode=e.get("INGEST_EXTRACTION_MODE", "text_only"),
            unreadable_pages=e.get("INGEST_UNREADABLE_PAGES", "keep_empty"),
            min_text_chars=int(e.get("INGEST_MIN_TEXT_CHARS", "16")),
            render_dpi=integer("INGEST_RENDER_DPI", 200),
            strict_pdf=flag("INGEST_STRICT_PDF"),
            allow_signed_url=flag("INGEST_ALLOW_SIGNED_URL"),
            min_confidence=(
                float(e["INGEST_MIN_CONFIDENCE"])
                if e.get("INGEST_MIN_CONFIDENCE")
                else None
            ),
            recognition_concurrency=integer("INGEST_RECOGNITION_CONCURRENCY", 4),
            max_recognized_pages=integer("INGEST_MAX_RECOGNIZED_PAGES", 1000),
            tenant_daily_recognized_pages=int(
                e.get("INGEST_TENANT_DAILY_RECOGNIZED_PAGES", "0")
            ),
            recognition_poll_seconds=integer("INGEST_RECOGNITION_POLL_SECONDS", 15),
            db_pool_size=int(e.get("INGEST_DB_POOL_SIZE", "0")),
            refuse_duplicates=not flag("INGEST_ALLOW_DUPLICATES"),
            abandoned_upload_grace_seconds=integer(
                "INGEST_ABANDONED_UPLOAD_GRACE_SECONDS", 3600
            ),
            api_uploads_per_minute=int(e.get("INGEST_API_UPLOADS_PER_MINUTE", "600")),
            api_searches_per_minute=int(e.get("INGEST_API_SEARCHES_PER_MINUTE", "120")),
            api_docs=flag("INGEST_API_DOCS"),
            vector_index=e.get("INGEST_VECTOR_INDEX", "hnsw"),
            hnsw_m=integer("INGEST_HNSW_M", 16),
            hnsw_ef_construction=integer("INGEST_HNSW_EF_CONSTRUCTION", 64),
            hnsw_ef_search=integer("INGEST_HNSW_EF_SEARCH", 200),
            hnsw_max_scan_tuples=integer("INGEST_HNSW_MAX_SCAN_TUPLES", 20_000),
            keyword_index=e.get("INGEST_KEYWORD_INDEX", "none"),
            keyword_language=e.get("INGEST_KEYWORD_LANGUAGE", "english"),
            search_hybrid=flag("INGEST_SEARCH_HYBRID"),
            search_fusion=e.get("INGEST_SEARCH_FUSION", "rrf"),
            search_rrf_k=integer("INGEST_SEARCH_RRF_K", 60),
            search_vector_weight=float(e.get("INGEST_SEARCH_VECTOR_WEIGHT", "1.0")),
            search_keyword_weight=float(e.get("INGEST_SEARCH_KEYWORD_WEIGHT", "1.0")),
            search_keyword_timeout_seconds=integer("INGEST_SEARCH_KEYWORD_TIMEOUT_SECONDS", 5),
            search_strict=flag("INGEST_SEARCH_STRICT"),
            search_rerank=flag("INGEST_SEARCH_RERANK"),
            rerank_top_n=integer("INGEST_RERANK_TOP_N", 50),
            rerank_timeout_ms=integer("INGEST_RERANK_TIMEOUT_MS", 1500),
        )
        result.validate_recognition()
        if not 600 <= result.abandoned_upload_grace_seconds <= 7 * 86400:
            raise ValueError(
                "INGEST_ABANDONED_UPLOAD_GRACE_SECONDS must be 600..604800"
            )
        if not 5 <= result.embedding_timeout_seconds <= 600:
            raise ValueError("INGEST_EMBEDDING_TIMEOUT_SECONDS must be 5..600")
        if not 1_000 <= result.embedding_max_batch_tokens <= 10_000_000:
            raise ValueError(
                "INGEST_EMBEDDING_MAX_BATCH_TOKENS must be 1000..10000000"
            )
        for name, value in (
            ("INGEST_API_UPLOADS_PER_MINUTE", result.api_uploads_per_minute),
            ("INGEST_API_SEARCHES_PER_MINUTE", result.api_searches_per_minute),
        ):
            if not 0 <= value <= 100_000:
                raise ValueError(name + " must be 0 (unlimited) or 1..100000")
        result.validate_search()
        # Same checks as the store, so a bad value stops startup, not the first search.
        from rag_ingest.adapters.postgres.vectors import PgVectorStore

        PgVectorStore(None, **result.vector_store_options())
        if not 0 <= result.db_pool_size <= 500:
            raise ValueError("INGEST_DB_POOL_SIZE must be 0 (derive) or 1..500")
        result.metadata_pool_size()  # refuse a pool smaller than the concurrency
        if result.verify_concurrency > 32:
            raise ValueError("INGEST_VERIFY_CONCURRENCY must be 1..32")
        if not 0 <= result.max_uncertain_repeats <= 10:
            raise ValueError("INGEST_MAX_UNCERTAIN_REPEATS must be 0..10")
        if result.embedding_driver != "openai":
            raise ValueError(
                "INGEST_EMBEDDING_DRIVER must be openai; inject a custom embedder in library mode"
            )
        if not 0 <= result.chunk_overlap < result.chunk_size or result.batch_size > MAX_BATCH_ITEMS:
            raise ValueError("Invalid chunk/batch settings")
        if result.presign_seconds > 3600 or result.queue_visibility_seconds > 43200:
            raise ValueError("Grant or visibility duration exceeds adapter bounds")
        if result.embedding_dimensions > 16000 or result.worker_concurrency > 64:
            raise ValueError("Dimension/concurrency limit exceeded")
        if not MULTIPART_MIN_PART_BYTES <= result.multipart_part_bytes <= MULTIPART_MAX_PART_BYTES:
            raise ValueError("Multipart part size must be 5 MiB..5 GiB")
        return result

    def extractor_options(self):
        """The document recipe's options; the recognizer block identifies the tool."""
        recognizer = None
        if self.extraction_mode != "text_only":
            if self.recognizer_spec is None or self.recognizer_capabilities is None:
                raise ValueError(
                    f"INGEST_EXTRACTION_MODE={self.extraction_mode} needs an injected "
                    "Recognizer (build_engine(settings, recognizer=...))"
                )
            recognizer = extraction_recipe.recognizer_block(
                self.recognizer_spec, self.recognizer_capabilities
            )
        return extraction_recipe.validate(
            {
                "mode": self.extraction_mode,
                "unreadable_pages": self.unreadable_pages,
                "min_text_chars": self.min_text_chars,
                "render_dpi": self.render_dpi,
                "strict_pdf": self.strict_pdf,
                "allow_signed_url": self.allow_signed_url,
                "min_confidence": self.min_confidence,
                "recognizer": recognizer,
                "normalize_text": True,
            }
        )

    def validate_recognition(self):
        """Check recognition settings at startup with actionable messages."""
        if self.extraction_mode not in extraction_recipe.MODES:
            raise ValueError(
                "INGEST_EXTRACTION_MODE must be one of "
                + ", ".join(extraction_recipe.MODES)
            )
        if self.unreadable_pages not in ("keep_empty", "fail"):
            raise ValueError("INGEST_UNREADABLE_PAGES must be keep_empty or fail")
        if not 0 <= self.min_text_chars <= 10_000:
            raise ValueError("INGEST_MIN_TEXT_CHARS must be 0..10000")
        if not 72 <= self.render_dpi <= 600:
            raise ValueError("INGEST_RENDER_DPI must be 72..600")
        if self.min_confidence is not None and not 0 <= self.min_confidence <= 1:
            raise ValueError("INGEST_MIN_CONFIDENCE must be between 0 and 1")
        if not 1 <= self.recognition_concurrency <= 64:
            raise ValueError("INGEST_RECOGNITION_CONCURRENCY must be 1..64")
        if self.tenant_daily_recognized_pages < 0:
            raise ValueError("INGEST_TENANT_DAILY_RECOGNIZED_PAGES must be >= 0")

    def pipeline(self):
        """Capture all output-affecting built-in choices as a durable recipe."""
        options = {
            "model": self.embedding_model,
            "dimensions": self.embedding_dimensions,
            "encoding": self.embedding_encoding,
        }
        return PipelineConfig(
            extractor=ComponentSpec(
                driver=extraction_recipe.DRIVER,
                implementation_version=extraction_recipe.VERSION,
                options_json=json.dumps(
                    self.extractor_options(), sort_keys=True, separators=(",", ":")
                ),
            ),
            chunker=ComponentSpec(
                driver="characters",
                implementation_version="1",
                options_json=json.dumps(
                    {"size": self.chunk_size, "overlap": self.chunk_overlap}
                ),
            ),
            embedder=ComponentSpec(
                driver=self.embedding_driver,
                implementation_version="1",
                options_json=json.dumps(options),
            ),
        )
