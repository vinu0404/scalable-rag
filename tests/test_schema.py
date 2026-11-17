"""Check packaged migration/schema consistency without connecting to PostgreSQL."""

import hashlib
import importlib
import json
import os
import unittest
from pathlib import Path

import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy.schema import CreateTable

import rag_ingest
from sqlalchemy.dialects import postgresql
from rag_ingest.adapters.postgres.schema import metadata


V0001_SHA256 = "db73ac63cc5dbb1cf8248a417583677fa2d3df41f3e7bd8251d2c5de00db3824"
DSN = os.environ.get("INGEST_TEST_METADATA_DSN")


class Schema(unittest.TestCase):
    def test_released_initial_revision_is_unchanged(self):
        """Released revisions are immutable; schema changes need a new revision."""
        revision = importlib.import_module(
            "rag_ingest.migrations.versions.v0001_initial"
        )
        digest = hashlib.sha256(json.dumps(revision.STATEMENTS).encode()).hexdigest()
        self.assertEqual(digest, V0001_SHA256)

    def test_revisions_form_one_chain(self):
        r1 = importlib.import_module("rag_ingest.migrations.versions.v0001_initial")
        r2 = importlib.import_module(
            "rag_ingest.migrations.versions.v0002_admission_retention"
        )
        self.assertEqual((r1.down_revision, r2.down_revision), (None, "0001"))
        r3, r4 = (
            importlib.import_module(f"rag_ingest.migrations.versions.{name}")
            for name in ("v0003_recognition", "v0004_batch_plans")
        )
        self.assertEqual((r3.down_revision, r4.down_revision), ("0002", "0003"))
        r5 = importlib.import_module(
            "rag_ingest.migrations.versions.v0005_duplicate_lookup"
        )
        self.assertEqual(r5.down_revision, "0004")
        r6 = importlib.import_module(
            "rag_ingest.migrations.versions.v0006_abandoned_upload_files"
        )
        self.assertEqual(r6.down_revision, "0005")
        r7 = importlib.import_module("rag_ingest.migrations.versions.v0007_search_chunks")
        self.assertEqual(r7.down_revision, "0006")
        r8 = importlib.import_module("rag_ingest.migrations.versions.v0008_keyword_index")
        self.assertEqual(r8.down_revision, "0007")

    @unittest.skipUnless(DSN, "Set INGEST_TEST_METADATA_DSN to a disposable *_test DB")
    def test_migrating_a_fresh_database_matches_schema(self):
        """Packaged migrations to head produce exactly schema.py (real Postgres)."""
        server = sa.engine.make_url(DSN).set(drivername="postgresql+psycopg")
        fresh = server.set(database="rag_ingest_migrations_test")
        admin = sa.create_engine(server, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(sa.text(f"DROP DATABASE IF EXISTS {fresh.database}"))
            c.execute(sa.text(f"CREATE DATABASE {fresh.database}"))
        try:
            config = Config()
            config.set_main_option(
                "script_location",
                str(Path(rag_ingest.__file__).parent / "migrations"),
            )
            config.attributes["dsn"] = fresh.render_as_string(hide_password=False)
            command.upgrade(config, "head")
            engine = sa.create_engine(fresh)
            with engine.connect() as c:
                diff = compare_metadata(MigrationContext.configure(c), metadata)
            engine.dispose()
            self.assertEqual(diff, [])
        finally:
            with admin.connect() as c:
                c.execute(sa.text(f"DROP DATABASE IF EXISTS {fresh.database}"))
            admin.dispose()

    def test_parent_foreign_keys_include_tenant(self):
        for table in metadata.tables.values():
            for constraint in table.foreign_key_constraints:
                self.assertIn("tenant_id", constraint.column_keys)
                self.assertTrue(constraint.deferrable)

    def test_scheduling_and_lease_constraints_are_present(self):
        self.assertIn("scheduling_done", metadata.tables["generations"].c)
        self.assertIn("reconciled_at", metadata.tables["jobs"].c)
        ddl = str(
            CreateTable(metadata.tables["jobs"]).compile(dialect=postgresql.dialect())
        )
        self.assertIn("lease_expires_at IS NOT NULL", ddl)
        self.assertIn("UNIQUE (tenant_id, generation_id, stage, shard)", ddl)
