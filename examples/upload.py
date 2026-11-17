"""Upload a local file directly to S3, resume observed parts, then request verification.

Run: python examples/upload.py ./example.txt --token "$INGEST_DEMO_TOKEN"
Keep the printed request key to resume: repeat with --request-key THE_SAME_KEY.
Never print or persist signed URLs. Each part uses at most the planned size in RAM.
"""

import argparse
import hashlib
import time
from pathlib import Path
from uuid import uuid4
import httpx


def main():
    """Hash in bounded blocks and use fresh grants for incomplete transfer parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--token", required=True)
    parser.add_argument("--request-key", default=None)
    parser.add_argument(
        "--document-id", default=None, help="Replace an existing document"
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    key = args.request_key or str(uuid4())
    print("Request key:", key, flush=True)
    digest = hashlib.sha256()
    with args.file.open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    with (
        httpx.Client(
            base_url=args.api,
            headers={"Authorization": "Bearer " + args.token},
            timeout=60,
        ) as api,
        httpx.Client(timeout=300) as storage,
    ):

        def request(method, path, **kw):
            """Return only successful JSON; avoid logging raw response bodies or URLs."""
            response = api.request(method, path, **kw)
            if response.is_error:
                raise RuntimeError(
                    f"API {method} returned {response.status_code}; inspect document/upload status"
                )
            return response.json()

        body = {
            "filename": args.file.name,
            "size_bytes": args.file.stat().st_size,
            "sha256": digest.hexdigest(),
        }
        if args.document_id:
            body["document_id"] = args.document_id
        upload = request(
            "POST", "/v1/uploads", json=body, headers={"Idempotency-Key": key}
        )
        id = upload["upload_id"]
        doc = upload["document_id"]
        print("Upload:", id, "Document:", doc, flush=True)
        if upload["status"] == "uploading":
            if upload["multipart"]:
                plan = upload["multipart"]
                observed = {}
                after = None
                while True:
                    params = {"after": after} if after else {}
                    page = request("GET", f"/v1/uploads/{id}/parts", params=params)
                    observed.update(
                        {p["part_number"]: p["size_bytes"] for p in page["parts"]}
                    )
                    after = page["next_after"]
                    if after is None:
                        break
                with args.file.open("rb") as source:
                    for number in range(1, plan["total_parts"] + 1):
                        offset = (number - 1) * plan["part_size_bytes"]
                        size = min(plan["part_size_bytes"], body["size_bytes"] - offset)
                        if observed.get(number) == size:
                            continue
                        source.seek(offset)
                        data = source.read(size)
                        grant = request(
                            "POST",
                            f"/v1/uploads/{id}/parts/{number}",
                            json={"sha256": hashlib.sha256(data).hexdigest()},
                        )
                        response = storage.put(
                            grant["url"], headers=grant["headers"], content=data
                        )
                        if response.is_error:
                            raise RuntimeError(
                                f"Storage part {number} returned {response.status_code}; resume with the same request key"
                            )
                        print("Part completed:", number, flush=True)
            else:
                grant = upload["grant"]
                with args.file.open("rb") as source:
                    response = storage.put(
                        grant["url"],
                        headers={
                            **grant["headers"],
                            "Content-Length": str(body["size_bytes"]),
                        },
                        content=source,
                    )
                # A create-only grant can report an existing completed object after a lost response.
                if response.is_error and response.status_code != 412:
                    raise RuntimeError(
                        f"Storage returned {response.status_code}; retry with the same request key"
                    )
            request("POST", f"/v1/uploads/{id}/complete")
        end = time.monotonic() + args.poll_seconds
        while time.monotonic() < end:
            upload = request("GET", f"/v1/uploads/{id}")
            if upload["status"] in ("failed", "expired", "cancelled"):
                raise RuntimeError("Upload " + upload["status"])
            status = request("GET", f"/v1/documents/{doc}")
            states = [j["status"] for j in status["jobs"]]
            if any(s in ("failed", "cancelled") for s in states):
                raise RuntimeError(
                    "Processing stopped; inspect GET /v1/documents/" + doc
                )
            if (
                status["active_generation_id"]
                and status["active_generation_id"] == status["desired_generation_id"]
            ):
                print("Ready:", doc)
                return
            print("Upload:", upload["status"], "Jobs:", states, flush=True)
            time.sleep(2)
        print("Still processing. Inspect GET /v1/documents/" + doc)


if __name__ == "__main__":
    main()
