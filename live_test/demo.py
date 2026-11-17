"""Demo page and grounded chat for the live test (application code, not the package).

GET /          the upload + status + chat page (live_test/web/index.html)
POST /demo/chat  search this tenant's live documents, then ask an OpenAI chat
                 model to answer only from the numbered sources it is given.

Authentication is the same bearer token as the API (INGEST_API_TOKENS).
The chat model is DEMO_CHAT_MODEL (default gpt-4o-mini) and uses the
embedding API key and base URL already configured for the service.
"""

import hmac
import os
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import Depends, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_ingest.api.app import enforce_rate, search_chunks

PAGE = Path(__file__).parent / "web" / "index.html"
SOURCES = 8  # chunks given to the model per question
MAX_SOURCE_CHARS = 2000  # per chunk, so a prompt stays bounded

SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. "
    "Cite every claim with its source number in square brackets, like [2]. "
    "If the sources do not contain the answer, say you could not find it in "
    "the uploaded documents. Do not use outside knowledge."
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)


def add_demo_routes(app, engine, *, transport=None):
    """Attach the page and chat endpoint to an API app built by create_app."""
    settings = engine.settings
    model = os.environ.get("DEMO_CHAT_MODEL", "gpt-4o-mini")
    llm = httpx.AsyncClient(
        base_url=(settings.embedding_base_url or "https://api.openai.com/v1").rstrip("/")
        + "/",
        headers={"Authorization": "Bearer " + (settings.embedding_api_key or "")},
        timeout=httpx.Timeout(60, connect=5),
        transport=transport,
    )

    async def tenant(authorization: str | None = Header(default=None)):
        """Same rule as the API: the token decides the tenant, nothing else."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Bearer token required")
        token = authorization[7:].encode()
        for secret, candidate in settings.api_tokens.items():
            if hmac.compare_digest(token, secret.encode()):
                return candidate
        raise HTTPException(401, "Invalid credentials")

    @app.get("/", include_in_schema=False)
    async def page():
        return FileResponse(PAGE, media_type="text/html")

    @app.post("/demo/chat")
    async def chat(body: ChatRequest, t: UUID = Depends(tenant)):
        # Each question embeds a query and calls a paid model: same limit as search.
        await enforce_rate(engine, t, "searches", settings.api_searches_per_minute)
        found = (await search_chunks(engine, t, body.question, SOURCES))["results"]
        sources = [
            dict(
                n=n,
                document_id=r["document_id"],
                filename=r["filename"],
                pages=r["pages"],
                score=round(r["score"], 4),
                text=r["text"],
                also_in=r["also_in"],
            )
            for n, r in enumerate(found, start=1)
        ]
        if not sources:
            return dict(
                answer="No documents are searchable yet. Upload files and wait for Ready.",
                sources=[],
                model=None,
            )
        context = "\n\n".join(
            f"[{s['n']}] {s['filename']}"
            + (f", page {', '.join(map(str, s['pages']))}" if s["pages"] else "")
            + f"\n{s['text'][:MAX_SOURCE_CHARS]}"
            for s in sources
        )
        try:
            r = await llm.post(
                "chat/completions",
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": f"Sources:\n\n{context}\n\nQuestion: {body.question}",
                        },
                    ],
                },
            )
        except httpx.HTTPError:
            raise HTTPException(503, "Chat model unreachable") from None
        if r.status_code != 200:
            detail = r.json().get("error", {}).get("message", "") if r.content else ""
            raise HTTPException(502, f"Chat model error {r.status_code}: {detail[:200]}")
        answer = r.json()["choices"][0]["message"]["content"]
        return dict(answer=answer, sources=sources, model=model)

    return llm
