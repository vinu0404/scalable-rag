"""Configurable, recoverable document ingestion for RAG applications."""

import logging

# A library never configures output; this silences the "no handler" fallback.
logging.getLogger(__name__).addHandler(logging.NullHandler())
