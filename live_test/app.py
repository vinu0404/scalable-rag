"""Run the service against real AWS with Textract injected as the Recognizer.

The built-in CLI cannot inject a recognizer, so this is how an application
wires one in. Roles run as separate processes so faults can be injected:

    .venv/bin/python -m live_test.app serve    # API + publisher + scheduler + demo page
    .venv/bin/python -m live_test.app worker   # workers (run 1 or more)

Live-test overrides are explicit below; everything else comes from .env.
"""

import argparse
import asyncio
import signal
from dataclasses import replace

import uvicorn

from rag_ingest.bootstrap import load_settings
from rag_ingest.logs import configure_logging

LIVE_OVERRIDES = dict(
    extraction_mode="recognize_unreadable",  # Textract only for pages that need it
    max_recognized_pages=40,  # per document
    tenant_daily_recognized_pages=300,  # hard Textract cost ceiling for the run
    multipart_threshold=8 * 1024**2,  # exercise real S3 multipart uploads
    multipart_part_bytes=5 * 1024**2,
    worker_concurrency=4,
    recognition_poll_seconds=5,
    keyword_index="postgres",  # built-in full-text keyword index
    search_hybrid=True,  # vector + keyword, fused
    search_rerank=True,  # reorder the top candidates with the LLM reranker
    rerank_top_n=20,
    rerank_timeout_ms=8000,  # an LLM is slower than a dedicated cross-encoder
)


def build():
    load_settings(env_file=".env")
    import os

    configure_logging(os.environ.get("INGEST_LOG_LEVEL", "INFO"), "text")
    from rag_ingest.runtime.factory import build_engine
    from rag_ingest.runtime.settings import ServiceSettings

    from live_test.openai_reranker import OpenAIReranker
    from live_test.textract_recognizer import TextractRecognizer

    settings = replace(ServiceSettings.from_env(), **LIVE_OVERRIDES)
    recognizer = TextractRecognizer(region=settings.core.s3_region)
    reranker = OpenAIReranker(api_key=settings.embedding_api_key)
    return build_engine(settings, recognizer=recognizer, reranker=reranker)


async def serve(engine):
    """API, publisher and scheduler (with verifier and Textract poller)."""
    from rag_ingest.api.app import create_app
    from rag_ingest.runtime.runners import run_publisher, run_scheduler

    from live_test.demo import add_demo_routes

    stop = asyncio.Event()
    app = create_app(engine, background=False)
    chat_client = add_demo_routes(app, engine)  # page at / and /demo/chat
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=engine.settings.core.api_port,
            log_config=None,
            access_log=False,
        )
    )
    roles = [
        asyncio.create_task(r(engine, stop)) for r in (run_publisher, run_scheduler)
    ]
    await server.serve()  # returns on Ctrl+C; its lifespan closes the engine
    stop.set()
    for task in roles:
        task.cancel()
    await asyncio.gather(*roles, return_exceptions=True)
    await chat_client.aclose()


async def worker(engine):
    from rag_ingest.runtime.runners import run_workers

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    task = asyncio.create_task(run_workers(engine, stop))
    waiter = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait((task, waiter), return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            await task
    finally:
        stop.set()
        task.cancel()
        waiter.cancel()
        await asyncio.gather(task, waiter, return_exceptions=True)
        await engine.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["serve", "worker"])
    role = parser.parse_args().role
    engine = build()
    asyncio.run(serve(engine) if role == "serve" else worker(engine))


if __name__ == "__main__":
    main()
