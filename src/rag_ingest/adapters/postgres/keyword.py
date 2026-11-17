"""Built-in keyword index: PostgreSQL full-text search over the chunk table.

No extra service and no second copy of the text: each chunk row in
search_chunks gets a tsvector (search_tsv) and the target it was built for
(keyword_target). A GIN index on (tenant_id, search_tsv) (needs btree_gin)
lets one tenant's matches be found inside the index.

Ranking is ts_rank_cd, which is not BM25 (no corpus-wide term rarity); in
hybrid search only the order matters, so it fuses well. For true BM25 inject
another KeywordIndex (OpenSearch, pg_search, ...).
"""

import re

import sqlalchemy as sa

from rag_ingest.contracts.errors import PermanentInputError
from rag_ingest.contracts.keyword import KeywordCapabilities, KeywordHit
from rag_ingest.contracts.vectors import VectorCheck, VectorReadiness

_LANGUAGE = re.compile(r"^[a-z_]{1,63}$")
MAX_QUERY_CHARACTERS = 4000


class PostgresKeywordIndex:
    """KeywordIndex over search_chunks; language is a PostgreSQL text-search config."""

    # What indexing a row sets, and when a row counts as indexed (subclasses differ).
    _INDEX_SET = "search_tsv = to_tsvector(CAST(:lang AS regconfig), content), keyword_target = :target"
    _STALE = "(keyword_target IS DISTINCT FROM :target OR search_tsv IS NULL)"
    _READY = "(keyword_target = :target AND search_tsv IS NOT NULL)"

    def __init__(self, database, *, language="english"):
        if not isinstance(language, str) or not _LANGUAGE.match(language):
            raise ValueError("language must be a text-search configuration name")
        self.db = database
        self.language = language
        self._target = f"postgres-fts:{language}"
        self._capabilities = KeywordCapabilities(
            scoring="tfidf", max_batch_items=1000, filters=frozenset({"document_ids"})
        )

    @property
    def target_id(self):
        return self._target

    @property
    def capabilities(self):
        return self._capabilities

    async def doctor(self):
        """Refuse a language PostgreSQL does not know (startup check)."""
        async with self.db.engine.connect() as c:
            found = (
                await c.execute(
                    sa.text("SELECT count(*) FROM pg_ts_config WHERE cfgname = :name"),
                    dict(name=self.language),
                )
            ).scalar_one()
        if not found:
            raise ValueError(f"PostgreSQL has no text-search configuration {self.language!r}")

    @staticmethod
    def _ids(items, limit):
        ids = [i.vector_id for i in items]
        if not 1 <= len(ids) <= limit or len(set(ids)) != len(ids):
            raise ValueError(f"Batch must hold 1..{limit} unique chunks")
        return ids

    async def index_batch(self, scope, *, items):
        """Build each row's tsvector from its stored text (idempotent).

        The text itself is never written here: it is already in the chunk
        row, which must hold exactly the text the caller indexed.
        """
        ids = self._ids(items, self._capabilities.max_batch_items)
        for item in items:
            scope.validate_item_ids(item.vector_id, item.document_id, item.generation_id)
        async with self.db.transaction(scope.tenant_id, lock=False) as (c, now):
            stored = dict(
                (
                    await c.execute(
                        sa.text(
                            "SELECT vector_id, text_sha256 FROM search_chunks"
                            " WHERE tenant_id = :tenant AND generation_id = :gen"
                            " AND vector_id = ANY(CAST(:ids AS text[]))"
                        ),
                        dict(tenant=scope.tenant_id, gen=scope.generation_id, ids=ids),
                    )
                ).all()
            )
            for item in items:
                if item.vector_id not in stored:
                    raise ValueError("Chunk row missing; the index stage writes it first")
                if stored[item.vector_id] != item.text_sha256:
                    raise PermanentInputError("Indexed text differs from the stored chunk")
            await c.execute(
                sa.text(
                    "UPDATE search_chunks SET " + self._INDEX_SET
                    + " WHERE tenant_id = :tenant AND generation_id = :gen"
                    " AND vector_id = ANY(CAST(:ids AS text[])) AND " + self._STALE
                ),
                dict(
                    lang=self.language,
                    target=self._target,
                    tenant=scope.tenant_id,
                    gen=scope.generation_id,
                    ids=ids,
                ),
            )

    async def verify_batch(self, scope, *, expected):
        ids = self._ids(expected, 10_000)
        async with self.db.transaction(scope.tenant_id, lock=False) as (c, now):
            rows = {
                vid: (digest, ready)
                for vid, digest, ready in await c.execute(
                    sa.text(
                        "SELECT vector_id, text_sha256, " + self._READY
                        + " FROM search_chunks WHERE tenant_id = :tenant"
                        " AND generation_id = :gen AND vector_id = ANY(CAST(:ids AS text[]))"
                    ),
                    dict(target=self._target, tenant=scope.tenant_id, gen=scope.generation_id, ids=ids),
                )
            }
        checks = []
        for e in expected:
            digest, ready = rows.get(e.vector_id, (None, False))
            readiness = (
                VectorReadiness.MISSING
                if digest is None or not ready
                else VectorReadiness.MISMATCH
                if digest != e.text_sha256
                else VectorReadiness.READY
            )
            checks.append(VectorCheck(vector_id=e.vector_id, readiness=readiness))
        return tuple(checks)

    async def search(self, *, tenant_id, query, k, document_ids=()):
        """Rows of this tenant matching any of the query's words, best first.

        plainto_tsquery reads the text as plain words (never as syntax, never
        a syntax error) and joins them with AND; a question would then only
        match chunks containing every word, which almost none do. The ANDs
        are turned into ORs ('simple' re-parses the already normalized
        words), and ts_rank_cd ranks chunks that cover more of the words,
        closer together, first. A query of stopwords only matches nothing.
        """
        if not isinstance(query, str) or not 1 <= len(query) <= MAX_QUERY_CHARACTERS:
            raise ValueError("Query must be 1..4000 characters")
        if not 1 <= k <= 1000 or len(document_ids) > 1000:
            raise ValueError("Invalid keyword search request")
        params = dict(lang=self.language, query=query, tenant=tenant_id, target=self._target, k=k)
        scope = ""
        if document_ids:
            scope = " AND sc.document_id = ANY(CAST(:docs AS uuid[]))"
            params["docs"] = [str(d) for d in document_ids]
        async with self.db.transaction(tenant_id, lock=False) as (c, now):
            rows = await c.execute(
                sa.text(
                    """WITH p AS (SELECT plainto_tsquery(CAST(:lang AS regconfig), :query) AS p),
                    q AS (SELECT CASE WHEN numnode(p.p) = 0 THEN p.p
                        ELSE to_tsquery('simple', replace(CAST(p.p AS text), ' & ', ' | '))
                        END AS q FROM p)
                    SELECT sc.vector_id, sc.document_id, sc.generation_id,
                        ts_rank_cd(sc.search_tsv, q.q) AS score
                    FROM search_chunks sc, q
                    WHERE numnode(q.q) > 0 AND sc.tenant_id = :tenant
                      AND sc.search_tsv @@ q.q AND sc.keyword_target = :target"""
                    + scope
                    + " ORDER BY score DESC, sc.vector_id LIMIT :k"
                ),
                params,
            )
            return tuple(
                KeywordHit(vector_id=v, document_id=d, generation_id=g, score=float(s))
                for v, d, g, s in rows
            )

    async def delete_generation(self, scope):
        async with self.db.transaction(scope.tenant_id, lock=False) as (c, now):
            await c.execute(
                sa.text(
                    "UPDATE search_chunks SET search_tsv = NULL, keyword_target = NULL"
                    " WHERE tenant_id = :tenant AND generation_id = :gen"
                ),
                dict(tenant=scope.tenant_id, gen=scope.generation_id),
            )

    async def aclose(self):
        """The database pool is shared with the metadata store; it closes it."""
