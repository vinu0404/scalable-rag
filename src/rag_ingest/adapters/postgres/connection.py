"""Own a bounded async PostgreSQL pool and short transactions."""

import logging
from contextlib import asynccontextmanager
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import (
    DBAPIError,
    InterfaceError,
    OperationalError,
    TimeoutError as PoolTimeoutError,
)
from rag_ingest.contracts.errors import (
    PermanentInputError,
    with_code,
    OperatorActionRequiredError,
    RetryableError,
    UncertainOutcomeError,
)
from rag_ingest.logs import log_event


log = logging.getLogger(__name__)
LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}
TLS_MODES = {"require", "verify-ca", "verify-full"}


def tls_required(url):
    """True when a remote connection insists on TLS (sslmode require or stronger).

    psycopg's default "prefer" silently falls back to an unencrypted
    connection, so only an explicit sslmode counts. Local hosts and Unix
    sockets are exempt.
    """
    host = url.host or ""
    if host in LOCAL_HOSTS or host.startswith("/"):
        return True
    return url.query.get("sslmode") in TLS_MODES


class Database:
    """Create connections explicitly; importing this module never connects.

    Version one serializes state-changing operations within each tenant using
    an advisory transaction lock. This favors correctness over maximum tenant
    throughput. No storage/model/network operation belongs inside that lock.
    """

    def __init__(self, dsn, *, pool_size=5):
        """Normalize the public DSN and configure connection/statement bounds."""
        url = sa.engine.make_url(dsn)
        if url.drivername not in ("postgres", "postgresql", "postgresql+psycopg"):
            raise ValueError("A PostgreSQL DSN is required")
        url = url.set(drivername="postgresql+psycopg")
        if not tls_required(url):
            log_event(
                log,
                "database_tls_not_required",
                logging.WARNING,
                host=url.host,
                hint="add ?sslmode=verify-full (or require) to the DSN",
            )
        self.engine = create_async_engine(
            url,
            pool_size=pool_size,
            max_overflow=pool_size,
            pool_timeout=10,
            pool_pre_ping=True,
            echo=False,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=15000 -c lock_timeout=10000",
            },
        )

    @asynccontextmanager
    async def transaction(self, tenant_id, *, lock=True):
        """Serialize one tenant's mutation and never hide an uncertain commit.

        A connection lost before COMMIT is sent leaves nothing committed, so it
        is retryable; only a failure while committing has an unknown outcome.
        """
        committing = False
        try:
            async with self.engine.connect() as conn:
                async with conn.begin():
                    if lock:
                        await conn.execute(
                            sa.text(
                                "SELECT pg_advisory_xact_lock(hashtextextended(:tenant, 0))"
                            ),
                            {"tenant": str(tenant_id)},
                        )
                    now = (
                        await conn.execute(sa.text("SELECT clock_timestamp()"))
                    ).scalar_one()
                    yield conn, now
                    committing = True
        except PoolTimeoutError:
            raise RetryableError("PostgreSQL connection pool is exhausted") from None
        except DBAPIError as exc:
            code = getattr(exc.orig, "sqlstate", None)
            if code in ("28000", "28P01", "42501"):
                raise OperatorActionRequiredError("PostgreSQL access failed") from None
            if code in ("40001", "40P01", "55P03", "57014"):
                raise RetryableError(
                    "PostgreSQL transaction was not completed"
                ) from None
            if not committing and isinstance(exc, (OperationalError, InterfaceError)):
                raise RetryableError(
                    "PostgreSQL connection failed before commit"
                ) from None
            if code and code.startswith("22"):
                # Data exception: the server rejected the values (for example a
                # NUL character in JSONB) and rolled back. Repeating cannot help.
                raise with_code(
                    PermanentInputError("PostgreSQL rejected data it cannot store"),
                    "unstorable_data",
                ) from None
            raise UncertainOutcomeError(
                "PostgreSQL transaction outcome needs reconciliation"
            ) from None

    async def aclose(self):
        """Return and close this adapter's pooled connections."""
        await self.engine.dispose()
