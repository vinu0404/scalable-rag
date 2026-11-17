"""Run the bundled API with the hypothetical company gateway adapter.

Set the normal infrastructure environment plus COMPANY_EMBEDDING_URL/TOKEN.
Run from the project root: python examples/custom_app.py
The gateway itself is an example contract; no real gateway is provided here.
"""

import os
from dataclasses import replace
import uvicorn
from company_embedder import CompanyEmbedder
from rag_ingest.bootstrap import load_settings
from rag_ingest.runtime.settings import ServiceSettings
from rag_ingest.runtime.factory import build_engine
from rag_ingest.api.app import create_app


def main():
    """Change the saved recipe and inject an adapter while keeping other defaults."""
    load_settings(env_file=".env")
    settings = replace(
        ServiceSettings.from_env(),
        embedding_driver="company_http",
        embedding_model="company-text-v2",
        embedding_encoding="utf8-bytes",
    )
    adapter = CompanyEmbedder(
        base_url=os.environ["COMPANY_EMBEDDING_URL"],
        token=os.environ["COMPANY_EMBEDDING_TOKEN"],
    )
    engine = build_engine(settings, embedder=adapter)
    uvicorn.run(
        create_app(engine, background=True),
        host=settings.core.api_host,
        port=settings.core.api_port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
