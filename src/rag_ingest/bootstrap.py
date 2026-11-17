"""Explicit configuration loading for application startup."""

from pathlib import Path

from dotenv import load_dotenv

from rag_ingest.config import Settings


def load_settings(*, env_file: str | Path | None = None) -> Settings:
    """Load an optional file into the environment, then validate settings.

    Call once at process startup, before creating clients or starting threads.
    Existing environment variables take precedence, including empty values.
    No file is searched for or loaded when env_file is None.
    File values are literal: ${VARIABLE} interpolation is disabled.
    Loading a file changes the process environment, even if validation fails.
    """
    if env_file is not None:
        path = Path(env_file).expanduser()
        # Opening explicitly surfaces missing/unreadable files to the caller.
        with path.open(encoding="utf-8") as stream:
            load_dotenv(stream=stream, override=False, interpolate=False)

    return Settings.from_env()
