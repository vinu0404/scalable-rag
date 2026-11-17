"""Provide concrete implementations of the ingestion contracts.

Adapters translate core operations into provider calls, such as PostgreSQL
transactions, S3 object operations, and SQS delivery. Keep provider-specific
SDK imports inside their adapter modules so optional dependencies stay optional.

Importing this package must not open connections, read credentials, run
migrations, or start workers. Application startup creates selected adapters.
"""
