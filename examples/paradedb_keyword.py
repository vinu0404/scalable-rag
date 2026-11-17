"""Example KeywordIndex: real BM25 inside PostgreSQL with ParadeDB pg_search.

Not part of the package (it needs a vendor extension), but tested: the
keyword conformance kit, the whole hybrid pipeline and provisioning run in
tests/test_paradedb.py, and it was benchmarked at 300k chunks (README
section 13). Wire it in with:

    keyword = ParadeDBKeywordIndex(Database(settings.core.metadata_dsn))
    await keyword.provision()       # once, after rag-ingest migrate
    engine = build_engine(settings, keyword=keyword)   # and INGEST_SEARCH_HYBRID=true

Same table, same text, same transactions as the built-in full-text adapter;
only the index differs. A pg_search `bm25` index on search_chunks ranks with
BM25 (corpus-wide term rarity) and returns the top k without scoring every
matching row (a Top K scan), so large tenants stay fast. Measured at 150k
chunks per tenant in README section 13.

Needs the pg_search extension (ParadeDB images, or installed on your own
PostgreSQL; many managed services do not offer it).

Rows count as indexed when keyword_target names this target; PostgreSQL
keeps the BM25 index itself in step with every committed write, so a row is
searchable as soon as its transaction commits.
"""

import logging

import sqlalchemy as sa

from rag_ingest.adapters.postgres.keyword import MAX_QUERY_CHARACTERS, PostgresKeywordIndex
from rag_ingest.contracts.keyword import KeywordCapabilities, KeywordHit
from rag_ingest.logs import log_event

log = logging.getLogger(__name__)

INDEX_NAME = "ix_search_chunks_bm25"
# Languages pg_search can both stem and strip stopwords for.
LANGUAGES = frozenset({
    "danish", "dutch", "english", "finnish", "french", "german", "hungarian", "italian",
    "norwegian", "portuguese", "russian", "spanish", "swedish",
})


class ParadeDBKeywordIndex(PostgresKeywordIndex):
    """KeywordIndex backed by a pg_search BM25 index on search_chunks."""

    _INDEX_SET = "keyword_target = :target"
    _STALE = "keyword_target IS DISTINCT FROM :target"
    _READY = "(keyword_target = :target)"

    def __init__(self, database, *, language="english"):
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
        self.db = database
        self.language = language
        self._target = f"paradedb-bm25:{language}"
        self._capabilities = KeywordCapabilities(
            scoring="bm25", max_batch_items=1000, filters=frozenset({"document_ids"})
        )

    def _definition(self):
        # Filters and the tie-breaker are untokenized (literal) so they run inside
        # the index and keep the Top K scan; the text is stemmed, stopwords removed.
        return (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} ON search_chunks USING bm25"
            " ((vector_id::pdb.literal), tenant_id, document_id,"
            f" (content::pdb.simple('stemmer={self.language}',"
            f" 'stopwords_language={self.language}')), (keyword_target::pdb.literal))"
        )

    async def provision(self):
        """Explicit setup, once after rag-ingest migrate: extension and BM25 index.

        Built CONCURRENTLY so writes continue; a failed build leaves an invalid
        index, which is dropped and rebuilt. A table holds one BM25 index, so a
        different language is reported, not silently rebuilt.
        """
        async with self.db.engine.connect() as c:
            c = await c.execution_options(isolation_level="AUTOCOMMIT")
            try:
                # A build over existing rows can take minutes; the pool's limit is 15 s.
                await c.execute(sa.text("SET statement_timeout = 0"))
                await c.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_search"))
                row = (
                    await c.execute(
                        sa.text(
                            "SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) FROM pg_index i"
                            " WHERE i.indexrelid = to_regclass(:name)"
                        ),
                        dict(name=INDEX_NAME),
                    )
                ).first()
                if row is not None and not row[0]:
                    await c.execute(sa.text(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}"))
                    row = None
                if row is None:
                    await c.execute(sa.text(self._definition()))
                elif f"stemmer={self.language}" not in row[1]:
                    log_event(
                        log,
                        "keyword_index_settings_differ",
                        logging.WARNING,
                        configured=self.language,
                        hint=f"DROP INDEX CONCURRENTLY {INDEX_NAME}, then provision()"
                        " and reindex-keyword",
                    )
            finally:
                await c.invalidate()  # never return the unlimited session to the pool
        await self.doctor()

    async def doctor(self):
        """Refuse to start without pg_search or the BM25 index."""
        async with self.db.engine.connect() as c:
            found = (
                await c.execute(
                    sa.text(
                        "SELECT count(*) FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid"
                        " WHERE c.relname = :name AND i.indisvalid"
                    ),
                    dict(name=INDEX_NAME),
                )
            ).scalar_one()
        if not found:
            raise ValueError(
                f"BM25 index {INDEX_NAME} missing: install pg_search and call provision()"
            )

    async def search(self, *, tenant_id, query, k, document_ids=()):
        """This tenant's best BM25 matches for any of the query's words.

        `|||` tokenizes the text with the index's analyzer and ORs the words:
        it is never parsed as query syntax. Every filter runs inside the
        index, so only the top k rows are read.
        """
        if not isinstance(query, str) or not 1 <= len(query) <= MAX_QUERY_CHARACTERS:
            raise ValueError("Query must be 1..4000 characters")
        if not 1 <= k <= 1000 or len(document_ids) > 1000:
            raise ValueError("Invalid keyword search request")
        params = dict(query=query, tenant=tenant_id, target=self._target, k=k)
        scope = ""
        if document_ids:
            scope = " AND document_id = ANY(CAST(:docs AS uuid[]))"
            params["docs"] = [str(d) for d in document_ids]
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = await c.execute(
                sa.text(
                    "SELECT vector_id, document_id, generation_id, pdb.score(vector_id) AS score"
                    " FROM search_chunks WHERE content ||| :query AND tenant_id = :tenant"
                    " AND keyword_target = :target"
                    + scope
                    # The "C" tie-breaker matches the literal index field (keeps Top K).
                    + ' ORDER BY pdb.score(vector_id) DESC, vector_id COLLATE "C" LIMIT :k'
                ),
                params,
            )
            return tuple(
                KeywordHit(vector_id=v, document_id=d, generation_id=g, score=float(s))
                for v, d, g, s in rows
            )
