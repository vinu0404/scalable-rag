"""Upload every PDF in test_data/pdfs through the real API and watch progress.

Idempotency keys derive from each file's name and bytes, so re-running this
script resumes instead of creating duplicates. Prints a progress table every
10 seconds and writes test_data/uploads.json (file -> upload/document ids).

    .venv/bin/python -m live_test.load [--timeout 1800]
"""

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from rag_ingest.bootstrap import load_settings

PDFS = Path("test_data/pdfs")
RECORD = Path("test_data/uploads.json")
API = "http://127.0.0.1:8080"


def token():
    load_settings(env_file=".env")
    return next(iter(json.loads(os.environ["INGEST_API_TOKENS"])))


def upload(path, api, storage):
    """One resumable upload: direct PUT or multipart parts, then complete."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    key = "live-" + hashlib.sha256((path.name + digest).encode()).hexdigest()[:40]
    body = {"filename": path.name, "size_bytes": len(data), "sha256": digest}
    response = api.post("/v1/uploads", json=body, headers={"Idempotency-Key": key})
    if response.is_error:
        return {"file": path.name, "error": f"initiate HTTP {response.status_code}"}
    up = response.json()
    result = {
        "file": path.name,
        "upload_id": up["upload_id"],
        "document_id": up["document_id"],
        "size": len(data),
        "multipart": bool(up["multipart"]),
    }
    if up["status"] != "uploading":
        return result
    if up["multipart"]:
        plan = up["multipart"]
        for number in range(1, plan["total_parts"] + 1):
            start = (number - 1) * plan["part_size_bytes"]
            part = data[start : start + plan["part_size_bytes"]]
            grant = api.post(
                f"/v1/uploads/{up['upload_id']}/parts/{number}",
                json={"sha256": hashlib.sha256(part).hexdigest()},
            ).json()
            r = storage.put(grant["url"], headers=grant["headers"], content=part)
            if r.is_error:
                return {**result, "error": f"part {number} HTTP {r.status_code}"}
        result["parts"] = plan["total_parts"]
    else:
        grant = up["grant"]
        r = storage.put(
            grant["url"],
            headers={**grant["headers"], "Content-Length": str(len(data))},
            content=data,
        )
        if r.is_error and r.status_code != 412:  # 412: already uploaded (create-only)
            return {**result, "error": f"PUT HTTP {r.status_code}"}
    r = api.post(f"/v1/uploads/{up['upload_id']}/complete")
    if r.is_error:
        return {**result, "error": f"complete HTTP {r.status_code}"}
    return result


def state(api, item):
    """Classify one document: uploading/verifying, waiting, processing, published, failed."""
    if "error" in item:
        return "upload_error", item["error"]
    upload = api.get(f"/v1/uploads/{item['upload_id']}")
    if upload.status_code == 404:
        return "deleted", ""
    up = upload.json()
    if up["status"] in ("failed", "expired"):
        return "upload_" + up["status"], ""
    if up["status"] != "accepted":
        return up["status"], ""
    doc = api.get(f"/v1/documents/{item['document_id']}")
    if doc.status_code == 404:
        return "deleted", ""
    doc = doc.json()
    jobs = doc["jobs"]
    failed = [j for j in jobs if j["status"] == "failed"]
    if failed:
        return "failed", failed[0]["failure_code"]
    if (
        doc["active_generation_id"]
        and doc["active_generation_id"] == doc["desired_generation_id"]
    ):
        return "published", ""
    if any(j["status"] == "waiting" for j in jobs):
        return "waiting_textract", ""
    if not jobs or all(j["status"] == "queued" and j["attempt"] == 0 for j in jobs):
        return "queued", ""
    return "processing", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    headers = {"Authorization": "Bearer " + token()}
    files = sorted(PDFS.glob("*.pdf"))
    started = time.monotonic()
    with (
        httpx.Client(base_url=API, headers=headers, timeout=120) as api,
        httpx.Client(timeout=300) as storage,
    ):
        with ThreadPoolExecutor(5) as pool:
            items = list(pool.map(lambda f: upload(f, api, storage), files))
        RECORD.write_text(json.dumps(items, indent=1))
        multipart = sum(1 for i in items if i.get("multipart"))
        print(
            f"uploaded {len(items)} files in {time.monotonic() - started:.0f}s "
            f"({multipart} via multipart); errors: {sum('error' in i for i in items)}",
            flush=True,
        )
        final = {
            "published",
            "failed",
            "upload_failed",
            "upload_expired",
            "upload_error",
            "deleted",
        }
        while True:
            try:
                states = {i["file"]: state(api, i) for i in items}
            except httpx.TransportError:
                print("API unreachable (restarting?); retrying in 10s", flush=True)
                time.sleep(10)
                continue
            counts = Counter(s for s, _ in states.values())
            elapsed = time.monotonic() - started
            print(
                f"[{elapsed:6.0f}s] "
                + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                flush=True,
            )
            if all(s in final for s, _ in states.values()) or elapsed > args.timeout:
                break
            time.sleep(10)
        for item in items:
            s, code = states[item["file"]]
            item["final_state"], item["failure_code"] = s, code
        RECORD.write_text(json.dumps(items, indent=1))
        print("\nfinal states:")
        for item in items:
            print(
                f"  {item['file']:40s} {item['final_state']:18s} {item['failure_code'] or ''}"
            )


if __name__ == "__main__":
    main()
