# Validation record — 15 November 2026

This records checks actually executed while preparing this package. It does not certify a production deployment.

| Check | Result |
| --- | --- |
| Python unittest discovery | 46 tests discovered: **44 passed, 2 skipped** |
| Boto3 S3/SQS request handling using in-process Moto | Passed immutable-write replay/conflict, source version pinning, presigned grant fields, multipart completion, exact-prefix cleanup, queue visibility/redelivery/acknowledgement tests |
| Processing orchestration | Passed extraction through publication with emulated S3, actual parser subprocess, deterministic embeddings and explicit metadata/vector test doubles |
| Recovery and publication evidence | Passed checkpoint reuse and rejection of publication when an expected vector is missing |
| API authorization | Passed missing-token rejection, cross-tenant document isolation, owner access and rejection of caller-supplied tenant fields |
| Embedding adapters | Passed hypothetical company-gateway handling and mocked OpenAI request/result/index handling; no paid calls |
| Retry logic | Passed asynchronous Tenacity and durable retry-decision tests |
| Schema | PostgreSQL DDL compilation, tenant-aware foreign keys, scheduling/lease constraints, packaged migration consistency passed |
| Alembic | Offline upgrade SQL generation passed; no live migration was run here |
| Formatting/static checks | Ruff formatting and undefined/unused-name checks passed |
| Packaging | Python wheel built; installed-wheel CLI help smoke check passed; packaged migration modules present |
| Local environment example | Parsed and validated with explicit dotenv loading |
| PostgreSQL/pgvector integration | **Not run: two opt-in tests skipped** because no test DSN was supplied and this environment denies socket creation |
| Docker Compose / LocalStack | **Not run here** |
| Live AWS/OpenAI | **Not run here** |
| 10 GiB or 10,000-document benchmark | **Not run** |

The processing test doubles do not establish PostgreSQL transaction correctness. Moto and mocked HTTP tests do not establish real provider compatibility or IAM permissions. Run the README's dedicated disposable PostgreSQL integration test and a real local upload before using this stack beyond development.

## Reproduce

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

For the two real database tests, follow README section 12 and use only a disposable database whose name ends in `_test`. They reset metadata/vector tables. The local run/setup instructions are in README section 5.

Validation interpreter: Python 3.12. Adapter dependencies used during validation included boto3 1.44.3, Moto 5.2.5, SQLAlchemy 2.1.8, psycopg 3.3.9, Alembic 1.20.1, HTTPX 0.28.1, pypdf 6.21.4 and Tenacity 9.2.1. Dependency ranges in pyproject.toml are broader; pin versions validated in your own deployment.
