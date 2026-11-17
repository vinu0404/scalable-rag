"""Verify the live run end to end and print a pass/fail report.

Reads Postgres, the real SQS queue, real S3 versions, pgvector, the saved
extraction outputs, and searches through the API. Writes
test_data/report.json.   .venv/bin/python -m live_test.verify
"""

import asyncio
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import boto3
import httpx
import sqlalchemy as sa

from live_test.load import API, RECORD, token
from rag_ingest.domain.enums import ProcessingStage as ST

TRUTH = Path("test_data/ground_truth.json")
EXPECTED_FAILURES = {  # file -> failure codes that are the correct outcome
    "edge-01-encrypted.pdf": {"input_or_pipeline_validation_failed"},
    "edge-02-corrupt.pdf": {"input_or_pipeline_validation_failed"},
    "edge-03-blank.pdf": {"no_extractable_text"},
}


def norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


async def main():
    from live_test.app import build

    engine = build()
    items = json.loads(RECORD.read_text())
    truth = json.loads(TRUTH.read_text())
    tenant = next(iter(engine.settings.api_tokens.values()))
    checks, report = [], {"documents": {}}

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # ---- documents: expected final state --------------------------------
    deleted = json.loads(Path("test_data/delete_test.json").read_text())["orig"]
    for item in items:
        if item.get("document_id") == deleted:  # fault-injection deletion
            item["final_state"] = "deleted"
            report["documents"][item["file"]] = {
                "state": "deleted",
                "code": None,
                "ok": True,
            }
            continue
        expected = EXPECTED_FAILURES.get(item["file"])
        state, code = item.get("final_state"), item.get("failure_code")
        ok = (
            (state == "failed" and code in expected)
            if expected
            else state == "published"
        )
        report["documents"][item["file"]] = {"state": state, "code": code, "ok": ok}
    bad = [f for f, d in report["documents"].items() if not d["ok"]]
    check("every document reached its expected outcome", not bad, ", ".join(bad))

    db = engine.metadata.db.engine
    async with db.connect() as c:
        q = lambda sql, **a: c.execute(sa.text(sql), a)  # noqa: E731
        stuck = (
            await q(
                "SELECT count(*) FROM jobs WHERE status IN "
                "('queued','running','retry_wait','waiting')"
            )
        ).scalar_one()
        check(
            "no job left queued/running/retrying/waiting", stuck == 0, f"{stuck} jobs"
        )
        unpublished = (
            await q("SELECT count(*) FROM outbox WHERE published_at IS NULL")
        ).scalar_one()
        check("outbox fully published", unpublished == 0, f"{unpublished} pending")
        report["job_states"] = dict(
            (
                await q(
                    "SELECT status || ':' || coalesce(failure_code,'-'), count(*) FROM jobs GROUP BY 1"
                )
            ).all()
        )
        rec = (await q("SELECT state, payload FROM recognition_jobs")).all()
        report["textract_jobs"] = dict(Counter(r.state for r in rec))
        pages_sent = sum(
            len(r.payload["fields"]["page_numbers"])
            if "fields" in r.payload
            else len(r.payload["page_numbers"])
            for r in rec
        )
        report["textract_pages_submitted"] = pages_sent
        dup = (
            await q(
                "SELECT count(*) - count(DISTINCT payload->'request_id') FROM recognition_jobs"
            )
        ).scalar_one()
        check("no Textract request submitted twice", dup == 0, f"{dup} duplicates")
        usage = (
            await q("SELECT coalesce(sum(pages),0) FROM recognition_usage")
        ).scalar_one()
        report["textract_pages_billed_budget"] = usage
        check(
            "Textract pages within the 300-page ceiling", usage <= 300, f"{usage} pages"
        )
        gone = (
            await q(
                "SELECT count(*) FROM rag_vectors WHERE document_id = CAST(:d AS uuid)",
                d=deleted,
            )
        ).scalar_one()
        check("deleted document: vectors physically erased", gone == 0, f"{gone} left")
        rows = (
            await q("""SELECT d.id, g.inventory, (SELECT count(*) FROM rag_vectors v
                 WHERE v.generation_id = g.id) AS vectors FROM documents d
                 JOIN generations g ON g.tenant_id = d.tenant_id AND g.id = d.active_id""")
        ).all()
        mismatched = [
            str(r.id) for r in rows if r.vectors != r.inventory["fields"]["chunk_count"]
        ]
        check(
            "pgvector: every published build has exactly its chunk count",
            not mismatched,
            ", ".join(mismatched),
        )
        report["vectors_total"] = sum(r.vectors for r in rows)

    # ---- SQS drained ----------------------------------------------------------
    sqs = boto3.client("sqs", region_name=engine.settings.core.sqs_region)
    attrs = sqs.get_queue_attributes(
        QueueUrl=engine.settings.core.sqs_queue_url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
        ],
    )["Attributes"]
    inflight = int(attrs["ApproximateNumberOfMessages"]) + int(
        attrs["ApproximateNumberOfMessagesNotVisible"]
    )
    check("SQS queue drained", inflight == 0, json.dumps(attrs))

    # ---- S3: pinned single-version sources, create-only outputs ---------------
    s3 = boto3.client("s3", region_name=engine.settings.core.s3_region)
    versions, marker = Counter(), {}
    while True:
        page = s3.list_object_versions(
            Bucket=engine.settings.bucket, Prefix=f"tenants/{tenant}/", **marker
        )
        for v in page.get("Versions", []):
            versions[v["Key"]] += 1
        if not page.get("IsTruncated"):
            break
        marker = {
            "KeyMarker": page["NextKeyMarker"],
            "VersionIdMarker": page["NextVersionIdMarker"],
        }
    overwritten = [k for k, n in versions.items() if n > 1]
    check(
        "S3: no object overwritten (one version per key)",
        not overwritten,
        ", ".join(overwritten[:5]),
    )
    report["s3_objects"] = len(versions)

    # ---- recognition: right pages, accurate text -------------------------------
    by_file = {i["file"]: i for i in items}
    scores, wrong_method = [], []
    for name, pages in truth.items():
        item = by_file.get(name)
        if not item or item.get("final_state") != "published":
            continue
        doc = await engine.metadata.get_document(
            tenant_id=tenant, document_id=item["document_id"]
        )
        jobs = await engine.metadata.list_generation_jobs(
            tenant_id=tenant, generation_id=doc.active_generation_id, stage=ST.EXTRACT
        )
        manifest = await engine.checkpoint_manifest(jobs[0])
        page_map = json.loads(
            await engine.read_output(engine._output(manifest, "page_map"))
        )
        text = (
            await engine.read_output(engine._output(manifest, "extracted_text"))
        ).decode()
        for entry in page_map["pages"]:
            expected = pages.get(str(entry["page"]))
            if expected is None:
                continue
            if entry["method"] != "recognized":
                wrong_method.append(f"{name} p{entry['page']}={entry['method']}")
                continue
            got = text[entry["start"] : entry["end"]]
            scores.append(
                difflib.SequenceMatcher(None, norm(expected), norm(got)).ratio()
            )
    check(
        "scanned pages were sent to Textract",
        not wrong_method,
        ", ".join(wrong_method[:5]),
    )
    if scores:
        report["recognition_similarity"] = {
            "pages": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
        }
        check(
            "Textract accuracy on known-text scans (mean similarity >= 0.95)",
            sum(scores) / len(scores) >= 0.95,
            json.dumps(report["recognition_similarity"]),
        )

    # ---- search through the API -------------------------------------------------
    queries = {
        "attention mechanism replacing recurrence in sequence transduction": "duplicate-of-text-01",
        "low-rank adaptation of large language models": "2106.09685",
        "shipment tracking code arrived at the regional warehouse": "scan-",
    }
    with httpx.Client(
        base_url=API, headers={"Authorization": "Bearer " + token()}, timeout=60
    ) as api:
        doc_to_file = {i.get("document_id"): i["file"] for i in items}
        for query, expected in queries.items():
            hits = api.post("/v1/search", json={"query": query, "limit": 5}).json()[
                "results"
            ]
            files = [doc_to_file.get(h["document_id"], "?") for h in hits]
            check(
                f"search: '{query[:40]}...' finds {expected}",
                any(expected in f for f in files[:3]),
                ", ".join(files[:3]),
            )
            check(
                f"search: deleted document absent for '{query[:30]}...'",
                all(h["document_id"] != deleted for h in hits),
            )
    await engine.aclose()

    report["checks"] = [{"check": n, "pass": ok, "detail": d} for n, ok, d in checks]
    Path("test_data/report.json").write_text(json.dumps(report, indent=1, default=str))
    print("\nLIVE TEST REPORT")
    for name, ok, detail in checks:
        print(
            f"  {'PASS' if ok else 'FAIL'}  {name}"
            + (f"  [{detail}]" if detail and not ok else "")
        )
    for key in (
        "job_states",
        "textract_jobs",
        "textract_pages_submitted",
        "textract_pages_billed_budget",
        "recognition_similarity",
        "vectors_total",
        "s3_objects",
    ):
        if key in report:
            print(f"  {key}: {report[key]}")
    print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
