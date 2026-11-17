"""Add published builds to the keyword index (after enabling or changing it).

Walks every live document's active build, reads its chunk text from the chunk
table page by page, and indexes it. Indexing is idempotent, so the command can
be stopped and run again. A build that stops being live while it is being
indexed (deleted or replaced) has its entries removed again, so an external
engine is never left holding entries no cleanup will visit.
"""

import logging

from rag_ingest.contracts.keyword import KeywordDocument
from rag_ingest.contracts.vectors import VectorScope
from rag_ingest.logs import log_event

log = logging.getLogger(__name__)


async def reindex_keyword(engine, *, page=500):
    """{'builds': n, 'chunks': n} indexed."""
    keyword = engine.keyword
    if keyword is None:
        raise ValueError("No keyword index configured (INGEST_KEYWORD_INDEX)")
    size = min(page, keyword.capabilities.max_batch_items)
    builds = chunks = 0
    after = None
    while True:
        batch = await engine.metadata.live_builds(after=after, limit=100)
        if not batch:
            return {"builds": builds, "chunks": chunks}
        for tenant, document, generation in batch:
            scope = VectorScope(tenant_id=tenant, document_id=document, generation_id=generation)
            last = None
            while True:
                rows = await engine.metadata.search_chunk_page(
                    tenant_id=tenant, generation_id=generation, after=last, limit=size
                )
                if not rows:
                    break
                await keyword.index_batch(
                    scope,
                    items=tuple(
                        KeywordDocument(
                            vector_id=v,
                            document_id=d,
                            generation_id=generation,
                            text=text,
                            text_sha256=digest,
                        )
                        for v, d, text, digest in rows
                    ),
                )
                chunks += len(rows)
                last = rows[-1][0]
            if not await engine.metadata.is_live_build(
                tenant_id=tenant, document_id=document, generation_id=generation
            ):
                await keyword.delete_generation(scope)
                log_event(log, "keyword_reindex_build_retired", generation_id=str(generation))
                continue
            builds += 1
        after = batch[-1][:2]
