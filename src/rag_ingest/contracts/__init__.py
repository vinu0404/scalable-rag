"""Define the interfaces that ingestion adapters must implement.

The core uses these interfaces so users can replace storage, queues,
processing components, or vector stores without rewriting ingestion rules.
Concrete provider connections belong in adapters, not in this package.
"""
