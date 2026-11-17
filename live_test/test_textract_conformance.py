"""Run the package's Recognizer conformance kit against real AWS Textract.

Cost: about 3 Textract pages. The sample PDF is stored under conformance/ in
the test bucket and deleted afterwards.
Run: .venv/bin/python -m unittest live_test.test_textract_conformance -v
"""

import hashlib
import unittest
from uuid import uuid4

from rag_ingest.adapters.aws.s3 import S3ObjectStore
from rag_ingest.bootstrap import load_settings
from rag_ingest.runtime.settings import ServiceSettings
from rag_ingest.testing.conformance import RecognizerConformance

from live_test.textract_recognizer import TextractRecognizer

load_settings(env_file=".env")
SETTINGS = ServiceSettings.from_env()


class TextractConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
    poll_interval_seconds = 3

    def make_recognizer(self):
        return TextractRecognizer(region=SETTINGS.core.s3_region)

    async def store_sample(self, data, name):
        self.store = S3ObjectStore(region=SETTINGS.core.s3_region)
        async def blocks():
            yield data

        info = await self.store.put_immutable(
            bucket=SETTINGS.bucket, key=f"conformance/{uuid4()}/{name}", blocks=blocks(),
            expected_size_bytes=len(data), expected_sha256=hashlib.sha256(data).hexdigest(),
        )
        self.stored = info.ref
        return info.ref

    async def asyncTearDown(self):
        if getattr(self, "stored", None):
            await self.store.delete_all_key_versions(self.stored)
            await self.store.aclose()
        await super().asyncTearDown()


if __name__ == "__main__":
    unittest.main()
