# Recoverable document ingestion for RAG

This project turns an uploaded PDF or UTF-8 text file into verified, searchable vectors. The goal is to keep accepted work recoverable when clients disconnect, queues deliver twice, workers crash, dependencies time out, or a document is replaced or deleted while processing is still running.

The package includes default **S3, SQS, PostgreSQL metadata, pgvector, local text extraction, character chunking, and OpenAI embedding adapters**. You can inject replacements. The local configuration uses OpenAI embeddings, so set `INGEST_EMBEDDING_API_KEY` before you start. The automated tests use an offline test embedder (`rag_ingest.testing.embedder`) and make no paid calls.

**Project status:** this is an initial implementation with automated tests and local setup instructions. It is not a claim of production certification or measured 10 GB/10,000-document capacity. See [Validation and limits](#validation-and-limits) before deploying.

## 1. Start with the problems

A simple `upload → extract → embed → insert` function works only while everything finishes successfully in one uninterrupted run. A reliable ingestion service must answer what happens after each interruption.

| Problem | What could go wrong | How this implementation handles it |
| --- | --- | --- |
| A large upload disconnects | Restarting a 10 GB file wastes time and bandwidth | Direct S3 multipart transfer, a durable part-size plan, provider part listing, and resumable part numbers |
| API response is lost | Client retries create two documents/builds | A tenant-scoped upload idempotency key binds the request contents to the original upload |
| A filename or client hash is untrustworthy | Wrong content is accepted or deduplication uses a fake identity | Inspect the completed object: stream its bytes to compute SHA-256, or use the full-object SHA-256 that storage itself verified at upload, and compare expected size/hash |
| The file changes after verification | A worker processes different bytes | Persist the exact S3 object version; workers read that version and recheck source integrity |
| Same bytes arrive twice | Duplicate content identities proliferate | Tenant-scoped SHA-256 uniqueness identifies one content artifact while retaining separate document/source identities |
| SQL commits but SQS send fails | Accepted work silently disappears | Create the job and outbox event in the same PostgreSQL transaction; publisher sends later |
| SQS accepts a message but its response is lost | Publisher sends again | Reuse the event identity; workers claim the durable SQL job rather than trusting message uniqueness |
| A queue delivers twice | Two workers might perform the same work | Current lease ID, worker identity, expiry, desired generation, and job status are checked transactionally |
| A worker crashes | Its job remains stuck forever | Expiring leases plus a scheduler that reconciles SQL state and recreates missing notifications |
| A worker keeps heartbeating | An unhealthy attempt could run forever | Lease renewal cannot extend the fixed attempt/job/generation deadlines |
| A provider is temporarily unavailable | Immediate retries overload it | Small Tenacity retries for known safe calls; durable delayed job retries with attempt/deadline bounds |
| A write times out after remote acceptance | Blind retries may duplicate side effects or billing | Classify an uncertain outcome separately; verify stable vector writes where possible; repeat lease-scoped, idempotent stage work a bounded number of times (`INGEST_MAX_UNCERTAIN_REPEATS`), then expose a terminal outcome for inspection |
| Half an embedding batch succeeds | Repeating everything wastes successful model work | Save the batch's successful embeddings in one immutable, self-checksummed segment before raising the unresolved failure; a retry embeds only what is missing |
| A saved checkpoint lies or references missing files | Recovery skips work incorrectly | Read the manifest and outputs; check exact hashes, sizes, source, recipe, and identities before reuse |
| Chunking crashes during fan-out | Some embedding batches are never scheduled | Seal an immutable complete inventory; scheduler releases its batches in bounded pages |
| Vector writes report success | Some records might not be queryable or might contain the wrong payload | Verify every expected vector ID, payload identity and stored float32 vector before publication |
| A replacement is half finished | Users see mixed old/new chunks | Keep the old generation active; publish the new generation by atomically changing one document pointer |
| An older build finishes late | It overwrites a newer replacement | Publication requires the generation still to be the desired generation |
| Deletion races with running work | A late worker makes deleted content visible again | Tombstone first, clear visibility, reject stale commits; physical cleanup follows a durable plan and grace period |
| Content is shared by multiple documents | Deleting one document breaks another | Cleanup checks live source references and freezes an artifact deletion plan under the tenant lock |
| Requests name another tenant | Documents or vector results leak | Resolve tenant from a trusted bearer-token mapping; scope metadata and vectors by tenant |
| Input is corrupt, encrypted, or requires OCR | A parser hangs or silently drops content | Killable parser subprocess, time/output limits, explicit failure; scanned/image pages are recognized by an injected `Recognizer` or recorded as `unrecognized`, never dropped silently |
| Workload is large | Unbounded tasks, memory, or connections exhaust the service | Fixed worker slots, bounded batch sizes/readers, finite transport timeouts and bounded SQL/SDK pools |

These mechanisms aim for **recoverable, at-least-once execution with guarded publication**. They do not promise exactly-once execution of every remote operation. For example, a model provider might bill a request whose response is lost.

## 2. The approach and why we built it in this order

1. **Domain records first.** Define document identity, source versions, generations, upload sessions, jobs, leases, artifacts, and outbox events. These are application concepts, not ORM models. Their constructors reject invalid local values.
2. **Contracts second.** Define what metadata, storage, queue, extraction, chunking, embedding, and vector adapters must guarantee. The core uses these operations without embedding AWS-specific request details.
3. **Durable orchestration third.** Put cross-record invariants in transactions: acceptance plus job plus outbox; lease checks plus completion; retry disposition; deletion and publication.
4. **Checkpoints and completeness next.** Save immutable outputs, manifests, a sealed generation inventory, and exact vector identities. Recovery must verify evidence before using it.
5. **Concrete adapters and runtime last.** Implement S3/SQS/PostgreSQL/pgvector and processing defaults, then wire the API, publisher, scheduler, and worker runners around those guarantees.

This order lets us ask “what must remain true after a crash?” before choosing a particular SDK call.

### Full upload and ingestion pipeline

```mermaid
flowchart TD
    C["Client: choose file"] --> A["API: authenticate and register upload"]
    A --> SQL["PostgreSQL: upload session and multipart plan"]
    A --> G["Short-lived PUT or part grants"]
    G --> T["Client uploads directly to S3"]
    T --> DONE["API: complete upload and pin object version"]
    DONE --> V["Scheduler: verify size, SHA-256 and type"]
    V --> BAD["Reject invalid upload"]
    V --> ACCEPT["One SQL transaction: source, generation, job, outbox"]
    ACCEPT --> PUB["Publisher: claim outbox and send"]
    PUB --> Q["SQS: job reference"]
    Q --> W["Worker: claim SQL lease"]
    W --> E["Extract text and page map"]
    E --> CH["Chunk and seal complete batch inventory"]
    CH --> B["Scheduler: release embedding batch jobs"]
    B --> EM["Embed batches and save successful vectors"]
    EM --> I["Index stable generation-scoped vector IDs"]
    I --> CHECK["Verify exact complete vector set"]
    CHECK --> ACTIVE["SQL: activate desired generation atomically"]
    ACTIVE --> SEARCH["Search: tenant plus live active generation"]
    W --> CP["S3 outputs and manifests; SQL checkpoints"]
    CP --> W
```

The diagram groups recurring queue dispatch into the worker path. Each stage/batch continuation is durable SQL work with its own notification, not an in-memory chain of function calls that disappears with the process.

### The four running roles

| Role | What it does | Why it exists |
| --- | --- | --- |
| API | Authenticate; register uploads; issue grants; complete transfer; expose status, view, replacement, deletion and search | Keep the request path separate from long processing |
| Publisher | Claim due outbox records; send to SQS; mark publication or reschedule | Bridge PostgreSQL and SQS without losing accepted work |
| Scheduler | Verify completed uploads in a bounded pool beside its loop; admit waiting builds up to a per-tenant cap; release due retries and inventory batches; reconcile expired jobs; process cleanup; prune expired history | Recover work even when the original process/message disappears |
| Worker | Receive a notification; claim job authority; execute/resume a stage; heartbeat; commit outcome; acknowledge | Perform bounded processing under durable ownership |

`serve` runs these roles together for local learning. Separate commands let deployments scale roles independently.

## 3. Important concepts with examples

### Document, source version, content artifact, generation

Suppose you upload `handbook.pdf`.

- **Document** is the logical item users view and search. It owns tenant, display name, access reference, active/desired generation and deletion state.
- **SourceVersion** records one accepted original file, including exact object version, size and SHA-256. It retains the document's citation/history identity.
- **ContentArtifact** identifies matching raw bytes inside a tenant. By default a second live document with the same bytes is refused as a duplicate; with `allow_duplicate`, two separate documents can refer to the same content identity without sharing their permissions or becoming the same document.
- **ProcessingGeneration** is a build of one source with one pipeline fingerprint. Changing the model or chunking recipe creates a different generation.

The default acceptance path detects raw duplicates using a trusted SHA-256 uniqueness constraint. It does **not** currently skip all extraction/embedding across different documents or normalize different files into equivalent text. Each document still builds its own generation-scoped outputs and vectors. That conservative behavior preserves provenance and authorization.

A source retains its own uploaded object version even if its artifact identity is shared. Matching bytes therefore do not imply that every duplicate physical upload is immediately removed.

### Pipeline fingerprint

A pipeline fingerprint is SHA-256 of canonical JSON describing the extractor, chunker, embedder, their implementation versions and output-affecting options.

For example, changing chunk size from `1000` to `1500`, model, dimensions, or adapter implementation version changes the fingerprint. API keys and endpoint secrets are not part of the saved recipe. Reusing old output requires the same recipe identity as well as matching source/input identities.

An endpoint's behavior must remain compatible with its saved implementation/model version. If you silently change an internal model behind the same name, the hash cannot detect that for you; use immutable model revisions and bump the recipe.

### Job lease versus SQS visibility

A **job lease** is PostgreSQL authority to change a job. SQS visibility only delays transport redelivery.

Example: worker A owns lease `L1`. Its SQS visibility extension fails, so worker B receives the message. B cannot claim the job while `L1` remains current. If A crashes and `L1` expires, a later worker can acquire a new lease `L2`. A cannot save a checkpoint or publish using the old `L1`.

Renewal uses the current lease ID and does not create a fresh budget. An attempt has an absolute deadline, each job has an absolute deadline, and the whole generation has an absolute deadline. Automatic execution eventually stops even if heartbeats continue.

### What happens in RETRY_WAIT?

```mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> RUNNING: acquire new lease
    RUNNING --> SUCCEEDED: guarded durable completion
    RUNNING --> RETRY_WAIT: retryable failure and remaining budget
    RETRY_WAIT --> QUEUED: scheduler reaches due time
    RUNNING --> FAILED: permanent, uncertain or exhausted
    QUEUED --> CANCELLED: deleted or superseded
    RUNNING --> CANCELLED: reconciliation after deletion or supersession
    SUCCEEDED --> [*]
    FAILED --> [*]
    CANCELLED --> [*]
```

When a job moves to `RETRY_WAIT`, its lease becomes `None`: no worker owns it while it waits. The job and its next-attempt time stay in PostgreSQL. The worker can acknowledge the old delivery after this durable outcome. When due, the scheduler atomically returns the job to `QUEUED` and adds an outbox notification. A worker then claims a **new** lease and increments the attempt count. Source, generation and existing fixed budgets are preserved.

Tenacity handles only short in-process retries where repeating the call is known safe. It does not replace this SQL scheduler, recover a dead process, or make unknown write outcomes safe.

### Why an outbox? Where is is_published?

The API writes a job and its notification intent in one PostgreSQL transaction. The publisher reads that **outbox record**, sends its `JobAvailable` event to SQS, then records `published_at`.

There is no `is_published` boolean. `published_at is None` means publication remains pending; a timestamp records confirmed send acceptance. The publisher uses its own expiring claim while sending. A lost send response may result in duplicate messages, which the SQL job claim handles.

A notification contains event ID, tenant ID and job ID. It does not contain the document bytes, full job details, credentials or a presigned source URL. The worker uses those IDs to read authoritative SQL state.

### Artifacts, manifests, checkpoints and inventory

| Record | Simple meaning | Example |
| --- | --- | --- |
| Output artifact | Saved bytes produced by a stage | Extracted UTF-8 text, page map, chunk batches, embedding segments |
| ManifestOutput | Description of one saved output | Bucket/key/version, SHA-256, size and format version |
| StageManifest | Expected outputs for one completed stage/batch | EXTRACT lists text and page-map outputs |
| StageCheckpoint | Durable SQL reference to the exact stage manifest | This job saved manifest M with hash H |
| GenerationInventory | Sealed full plan of chunk batches and expected identities | Batches covering chunk ordinals 0 through 199 |
| BatchPlan | One batch's entry copied from the sealed inventory into `scheduled_batches.plan` when its embedding job is scheduled | Embedding job 000003 reads this row instead of downloading the whole inventory |
| Search chunk row | The one stored copy of a chunk's text, page spans and text hash (`search_chunks`), written by the index stage | Search reads a hit's text here after checking its document is live |
| Index receipt | The index stage's record of what it wrote, with one digest per batch | Publication compares these digests with the stored rows |
| SourceSpan | Where a chunk came from in a saved extraction | Extraction hash, character offsets, optional PDF page |

A manifest is evidence of **what should exist**, not proof that storage actually contains those bytes. Recovery reads and verifies its referenced files. A checkpoint alone cannot authorize skipping corrupted or missing outputs.

SourceSpan offsets are Python character positions in the saved extracted text, with an inclusive start and exclusive end. They are not byte offsets into the PDF. The page map preserves page attribution for chunks crossing pages.

### Why vector acceptance is not publication

Suppose a batch writes 100 vectors and the vector database accepts the request. That proves request acceptance only. Before publication, the service verifies every expected vector identity, payload and actual stored vector, checks complete generation counts/ID digest, then transactionally activates the still-desired generation.

Publication does this without downloading the vectors again. The index stage records each batch's digest (vector ID, payload hash and exact float32 vector of every row) computed from the verified embedding segments, and publication asks the database for the same digest over the stored rows, a bounded group of batches per query. A batch whose digests differ is then checked row by row, so a changed vector still fails permanently and a missing one is still retried. Vector stores without the optional digest methods are always checked row by row.

### How search works

Each chunk's text is stored **once**, in the `search_chunks` table of the metadata database. Indexes hold only IDs: a `rag_vectors` row keeps the vector and its `payload_hash`, nothing else (the earlier layout also kept a second copy of the vector and the text as JSON, about 73% of every row). The index stage writes the chunk text first, then the vectors, and records a digest of each batch's text in its receipt; publication recomputes that digest from the stored text and refuses to publish if any chunk is missing or changed (`search_chunks_mismatch`).

A search embeds the query, asks the vector store for the nearest IDs (fetching four times the limit, more if needed), then reads text, file name and pages from `search_chunks` **only for hits whose document is live, visible and in the active build**, applying any filters in the same query. With filters that match at most 500 documents, only those documents' vectors are searched, so a narrow filter never misses results. Identical passages in different files are returned once, with the other places listed in `also_in`. Deleting a document hides it at once and its cleanup erases its chunk text with its vectors.

**Keyword and hybrid search.** Keyword search is an adapter like every other component (`contracts/keyword.py`): the built-in one (`INGEST_KEYWORD_INDEX=postgres`) is PostgreSQL full-text search on the chunk table, so it needs nothing you do not already run (a `tsvector` per row and a GIN index on `(tenant_id, search_tsv)`, which needs the `btree_gin` extension). Any BM25 engine can be injected instead: `examples/paradedb_keyword.py` (ParadeDB `pg_search`, same database, tested and benchmarked) and `examples/opensearch_keyword.py` show two and `rag_ingest.testing.keyword_conformance` checks any adapter. It is detached unless configured. Like vector stores, keyword engines return only IDs and scores, so the same live-document check applies. The built-in engine matches chunks containing *any* of the query's words and ranks those covering more of them, closer together, first (requiring every word would make question-style queries match almost nothing). In hybrid mode the query embedding and the keyword search run at the same time, each ranking keeps its own order, and Reciprocal Rank Fusion combines them; a passage found by both rises. User text is never parsed as query syntax. If the keyword engine fails or times out, vector results are returned with `degraded: ["keyword"]` (or the search fails with `INGEST_SEARCH_STRICT=true`); a programming error in an adapter is never hidden that way. Builds published before a keyword index was enabled or changed publish normally (logged as `keyword_index_stale`); `rag-ingest reindex-keyword` adds them, can be stopped and rerun, and removes entries of builds deleted while it runs. PostgreSQL's ranking (`ts_rank_cd`) is not BM25 and scores every matching row; the ParadeDB example ranks with BM25 and stays fast for large tenants (measured in section 13).

**Reranking.** A reranker (`contracts/rerank.py`) reads the question together with each candidate passage and scores how well it answers it: more accurate than either retriever, and slower, so it only reorders the top candidates. With reranking on, search gathers `INGEST_RERANK_TOP_N` candidates (more than the requested limit), cuts text to the reranker's declared limits, and requires exactly one finite score per candidate; the result's `score` becomes the reranker's score and `ranks` gains `rerank`. Any failure, timeout or invalid reply keeps the fused order with `degraded: ["rerank"]` (or fails the search with `INGEST_SEARCH_STRICT=true`); a programming error in the adapter is not hidden. None is bundled: `examples/http_reranker.py` speaks the Cohere-style and Hugging Face TEI rerank APIs, `live_test/openai_reranker.py` uses a chat model, and `rag_ingest.testing.rerank_conformance` checks any adapter.

**The vector index.** pgvector keeps an HNSW index (`rag_vectors_hnsw`, cosine distance), built by `rag-ingest migrate` with `CREATE INDEX CONCURRENTLY` so writes continue; a failed concurrent build leaves an invalid index, which the next `migrate` drops and rebuilds. All tenants share one table, so a plain index scan would mostly meet other tenants' vectors and could return too few results. Search therefore uses pgvector's **iterative scan**: the graph walk continues until `k` rows pass the tenant filter, up to `INGEST_HNSW_MAX_SCAN_TUPLES`. If it still returns fewer than `k`, the tenant is searched exactly: either it has fewer than `k` vectors (cheap), or the scan limit was reached, which logs `vector_search_scan_limit_reached` so the limit can be raised. Searches limited to documents (`document_ids`, or narrow filters) are always exact. Up to 2,000 dimensions the index is on `vector`; 2,001–4,000 use a `halfvec` expression index (ranking only); above 4,000 there is no index and search is exact. Measure recall and latency with `rag-ingest search-eval --samples 200 --k 10 --ef 40,100,200` and pick the smallest `INGEST_HNSW_EF_SEARCH` that meets your recall target **and** misses no query entirely (the report counts queries whose results share nothing with exact search). Measured numbers at 300,000 vectors are in section 13.

Publication verification never uses the index: it reads the stored float32 values and compares them exactly. Another vector adapter must implement its provider's query-readiness semantics; an accepted asynchronous write is insufficient.

## 4. Large-file upload: parts are different from text chunks

**Multipart parts** split file bytes for transport. **RAG chunks** split extracted text for embedding. They have different sizes and purposes.

For a **10 GiB file** and the default **64 MiB part size**, the plan has **160 parts**. The server sets the part size once and persists it. It chooses at least the configured size and increases it when needed to stay within 10,000 parts, rounding up to MiB boundaries. Non-final parts must meet S3's minimum size; the last can be smaller.

The example client transfers one part at a time, so it does not load a 10 GiB file into RAM. It reconciles existing provider parts on resume. You may implement bounded parallel transfer, but preserve the same plan, offsets and part numbers.

The sequence is:

1. Client sends name, expected size, optional expected SHA-256, and an idempotency key.
2. API durably registers the upload and returns a direct PUT grant or multipart plan.
3. Client requests part grants and uploads bytes directly to S3.
4. Client lists provider-confirmed parts; the service records observations.
5. Client requests completion. The service checks exact part numbers, ETags and sizes against provider truth, completes the object, then pins its version.
6. API responds `VERIFYING`. The scheduler verifies the completed version: it streams the bytes and computes a trusted SHA-256, or, for a single-PUT PDF with a declared SHA-256, uses the checksum S3 verified at upload (see below). A file whose bytes a live document already has is refused here as a duplicate.
7. Successful verification atomically creates the accepted source, generation, first job and outbox record.
8. Workers begin extraction.

**Verification happens after the completed file reaches S3 and before processing is accepted.** It normally reads and hashes every byte. One exception skips that second download: a PDF uploaded in a single PUT with its declared SHA-256 (`sha256` at initiation). S3 itself checks that checksum over every byte and refuses a mismatch (`BadDigest`), so verification reads S3's full-object SHA-256 with a HEAD request and only the first 4 KiB for the file-type check. Text files (full UTF-8 check), multipart uploads and uploads without a declared SHA-256 are always read in full. It is not a promise that an arbitrary 10 GiB PDF can be parsed within the configured parser/time/text limits. The default worker downloads a verified source to temporary disk, so provision enough disk per worker. Excessive extracted output, unsupported files, encrypted PDFs, and image-only PDFs fail explicitly.

Uploads that expire cannot be resumed under the same expired session. Start a new request key. A lifecycle rule aborts abandoned multipart transfers; see the cleanup limitations below for unused completed objects.

## 5. Run locally

### Prerequisites

- Python 3.11 or newer; Python 3.12 is the development validation interpreter.
- Docker with Compose for PostgreSQL/pgvector and LocalStack.
- Enough temporary disk for your source files. Start with a small `.txt` file.

Use a terminal inside this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
cp .env.local.example .env
docker compose up -d
```

On Windows, activate with `.venv\Scripts\activate`; use PowerShell environment syntax for later `export` commands.

Wait until `docker compose ps` shows PostgreSQL healthy and LocalStack running, then:

```bash
rag-ingest --env-file .env local-init
rag-ingest --env-file .env migrate
rag-ingest --env-file .env doctor
rag-ingest --env-file .env serve
```

`local-init` explicitly creates a local bucket and standard queue, enables bucket versioning, and adds an incomplete multipart abort lifecycle rule. It refuses non-loopback endpoints. If it prints a different queue URL, copy that exact URL into `.env` before continuing. The emulator's URL format can vary by version/configuration.

`migrate` applies the packaged Alembic metadata migration and explicitly creates/checks pgvector storage. Normal runtime startup does not create cloud resources, extensions or schema.

`doctor` checks metadata tables, vector dimensions, bucket access/versioning and queue access. It checks for an embedding key when using OpenAI, but sends no paid model call. It is not a full IAM write-permission or model-compatibility test.

`serve` starts API, publisher, scheduler and a bounded worker pool. Open [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) for the API reference.

### Upload your first file

In a second activated terminal:

```bash
export INGEST_DEMO_TOKEN='local-development-token-change-me-123456'
printf 'Reliable ingestion preserves work after a worker crashes.\n' > example.txt
python examples/upload.py example.txt --token "$INGEST_DEMO_TOKEN"
```

The script prints a request key, upload ID and document ID. It uploads directly to storage, requests verification and polls until the desired generation becomes active. Keep the request key if the client disconnects:

```bash
python examples/upload.py example.txt --token "$INGEST_DEMO_TOKEN" --request-key YOUR_PREVIOUS_KEY
```

Use the same file contents and filename. A changed request bound to an existing key is rejected. The script checks provider-observed part sizes before skipping them; the final full-file checksum verifies resumed bytes.

For replacement:

```bash
python examples/upload.py updated-handbook.pdf --token "$INGEST_DEMO_TOKEN" --document-id YOUR_DOCUMENT_ID
```

The old active generation remains searchable until the new generation passes verification and publication. The logical display name currently stays with the document when replacing it; its accepted source may have a different original filename.

### Inspect, view and search

Replace the document ID below:

```bash
curl -sS http://127.0.0.1:8080/v1/documents/YOUR_DOCUMENT_ID \
  -H "Authorization: Bearer $INGEST_DEMO_TOKEN"

curl -sS http://127.0.0.1:8080/v1/documents/YOUR_DOCUMENT_ID/view \
  -H "Authorization: Bearer $INGEST_DEMO_TOKEN"

curl -sS http://127.0.0.1:8080/v1/search \
  -H "Authorization: Bearer $INGEST_DEMO_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does recovery work?","limit":5}'
```

The view endpoint returns a short-lived URL to the exact original source version. Add `?download=true` for attachment disposition. Treat returned grants as secrets. The API does not proxy the entire original through application memory.

Search quality depends on the embedding model; the tests' offline embedder (`rag_ingest.testing.embedder`) gives meaningless vectors and is only injected in tests, never configured.

### Embeddings

`.env.local.example` uses OpenAI `text-embedding-3-small`. Before your first vector migration, put your key in `.env`:

```dotenv
INGEST_EMBEDDING_DRIVER=openai
INGEST_EMBEDDING_MODEL=text-embedding-3-small
INGEST_EMBEDDING_DIMENSIONS=1536
INGEST_EMBEDDING_ENCODING=cl100k_base
INGEST_EMBEDDING_API_KEY=YOUR_SECRET_KEY
```

`openai` is the only embedder you can select in configuration (any OpenAI-compatible endpoint via `INGEST_EMBEDDING_BASE_URL`). For another model, inject your own `Embedder` with `build_engine(embedder=...)` (section 9).

The tokenizer may download its encoding data on first use. For restricted deployments, prewarm its cache during image creation. Model API usage is billed by your provider.

An existing vector table cannot silently change dimensions (for example `vector(1536)` to `vector(3072)`). Use a separate fresh database for a different dimension, or implement a reviewed migration/target strategy. `migrate`/`doctor` reject dimension mismatch. Do not delete a database containing data you need.

### Run roles separately

For a deployment or a closer local simulation, use four terminals instead of `serve`:

```bash
rag-ingest --env-file .env api
rag-ingest --env-file .env publisher
rag-ingest --env-file .env scheduler
rag-ingest --env-file .env worker
```

Each command is a separate long-running process. Run the scheduler even when the queue appears healthy: it owns verification, admission, delayed retries, batch release, recovery, cleanup and history pruning.

**Capacity and deadlines.** An accepted build waits until the scheduler admits it (at most `INGEST_MAX_ACTIVE_GENERATIONS` per tenant, oldest first). Its job and generation deadlines start at admission, so a bulk load never fails just because documents waited for capacity. A build that passes its generation deadline is marked failed and frees its slot. Upload hashing runs in its own bounded pool, so a large upload never delays retries or recovery.

For real AWS, supply existing resources, remove the local endpoint overrides and dummy credentials, and use the standard AWS credential chain such as an instance/task role. Require an enabled versioned bucket and a **standard SQS queue**. The adapter intentionally rejects FIFO queue URLs. Provision IAM, encryption, lifecycle, backups, queue retention/DLQ and network policy through your infrastructure tooling.

Use HTTPS and your application's authentication at the deployment boundary. The sample static token map is useful for integration, but is not a complete enterprise identity or per-document authorization system.

### Demo page with upload status and chat (optional)

`live_test/` contains a local demo that runs against real AWS: a page at `http://127.0.0.1:8080` to pick files, watch each one move from upload through processing to Ready (or Failed / Duplicate), and ask questions answered only from your documents, with numbered sources you can open. Start it with `.venv/bin/python -m live_test.app serve` and `.venv/bin/python -m live_test.app worker`, paste a token from `INGEST_API_TOKENS`, and choose files. Browsers upload straight to S3, so the bucket needs the CORS rule in `live_test/web/s3-cors.json` for that origin. Chat uses `DEMO_CHAT_MODEL` (default `gpt-4o-mini`) with the configured embedding API key. This is demo application code, not part of the package.

## 6. Configuration

Existing environment variables win over dotenv values. A file is loaded only when you explicitly pass `--env-file`. Variable interpolation inside the file is disabled. Do not commit `.env`.

| Setting | Default / requirement | Purpose |
| --- | --- | --- |
| `INGEST_METADATA_DSN` | Required PostgreSQL URL | Authoritative durable metadata |
| `INGEST_VECTOR_DSN` | Metadata DSN | Vector storage; bundled search requires metadata and vectors in the same DB |
| `INGEST_S3_BUCKET` | Required | Sources, outputs and manifests |
| `INGEST_SQS_QUEUE_URL` | Required | Existing standard queue |
| `INGEST_S3_REGION`, `INGEST_SQS_REGION` | `ap-south-1` | Provider regions |
| `INGEST_S3_ENDPOINT_URL`, `INGEST_SQS_ENDPOINT_URL` | Unset | Optional compatible/emulator endpoints |
| `INGEST_S3_PUBLIC_ENDPOINT_URL` | Internal S3 endpoint | Host reachable by clients using signed grants |
| `INGEST_API_TOKENS` | JSON token→tenant UUID mapping | Trusted tenant-wide API authorization; tokens ≥24 chars |
| `INGEST_API_HOST`, `INGEST_API_PORT` | `0.0.0.0`, `8080` | API bind address |
| `INGEST_API_UPLOADS_PER_MINUTE`, `INGEST_API_SEARCHES_PER_MINUTE` | `600`, `120` (`0` = unlimited) | Per-tenant limits on starting uploads and on searches (each search embeds the query, a paid call). Shared by all API processes through PostgreSQL; over the limit the API answers `429` with `Retry-After` |
| `INGEST_API_DOCS` | `false` | Serve the interactive API docs and schema at `/docs`, `/redoc`, `/openapi.json` |
| `INGEST_EMBEDDING_DRIVER` | `openai` | `openai` (any OpenAI-compatible API). Other models: inject an `Embedder` with `build_engine(embedder=...)` |
| `INGEST_EMBEDDING_MODEL` | `text-embedding-3-small` | Saved model identity |
| `INGEST_EMBEDDING_DIMENSIONS` | `1536` | Vector dimension and database table type |
| `INGEST_EMBEDDING_ENCODING` | `cl100k_base` | Input tokenizer |
| `INGEST_EMBEDDING_API_KEY` | Falls back to `OPENAI_API_KEY` | Embedding secret; never saved in a recipe |
| `INGEST_EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible embedding API |
| `INGEST_EMBEDDING_TIMEOUT_SECONDS` | `30`, `5..600` | Time one embedding call may take, including a quick retry; each HTTP attempt gets two thirds of it. Raise for slower or self-hosted models. Not part of the recipe |
| `INGEST_EMBEDDING_MAX_BATCH_TOKENS` | `200000` | Tokens one embedding request may carry; match your provider's per-request limit. A batch over it fails as a permanent input error. Not part of the recipe |
| `INGEST_MAX_UPLOAD_BYTES` | 10 GiB | Accepted transfer size ceiling |
| `INGEST_MULTIPART_THRESHOLD_BYTES` | 64 MiB | Switch from whole PUT to multipart |
| `INGEST_MULTIPART_PART_BYTES` | 64 MiB | Persisted initial multipart part size |
| `INGEST_UPLOAD_SECONDS` | `3600` | Upload plus verification session lifetime |
| `INGEST_PRESIGN_SECONDS` | `300`, at most `3600` | Grant lifetime, also clamped to remaining session time |
| `INGEST_MAX_SOURCE_DISK_BYTES` | 10 GiB | Maximum source downloaded for parsing |
| `INGEST_MAX_EXTRACTED_CHARACTERS` | `20000000` | Bounded saved extraction text |
| `INGEST_MAX_PAGES` | `10000` | Parser page ceiling |
| `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP` | `1000`, `100` | Character chunking recipe |
| `INGEST_EMBEDDING_BATCH_SIZE` | `64`, at most `512` | Bounded embedding/checkpoint batches |
| `INGEST_WORKER_CONCURRENCY` | `2`, at most `64` | Number of in-flight jobs per worker process |
| `INGEST_ALLOW_DUPLICATES` | `false` | When false, a file whose exact bytes (SHA-256) a live document in the same tenant already has is refused: at `POST /v1/uploads` with HTTP 409 `duplicate_content` naming the existing document when `sha256` is declared, otherwise at verification (upload `failed`, `duplicate_of` set). A request can send `allow_duplicate: true` to keep a second copy. Deleted documents and failed builds never count |
| `INGEST_ABANDONED_UPLOAD_GRACE_SECONDS` | `3600`, `600..604800` | The scheduler erases the stored file (and aborts unfinished multipart parts) of uploads that ended `failed` or `expired` (refused duplicates, failed verification, abandoned transfers) this long after the session expired, then records `files_removed_at`. Accepted uploads are never touched |
| `INGEST_DB_POOL_SIZE` | `0` (derived) | Metadata connections per process. Derived as workers × 4 + verifiers + recognition + 4; a smaller value is refused at startup. Each pool may burst to 2×; size Postgres `max_connections` (or a pooler) for all processes |
| `INGEST_SQS_VISIBILITY_SECONDS` | `120` | Delivery invisibility, separate from SQL lease |
| `INGEST_MAX_UNCERTAIN_REPEATS` | `2`, `0..10` | Bounded repeats after an uncertain outcome or timeout; each may re-bill a model call; `0` makes them terminal |
| `INGEST_MAX_ACTIVE_GENERATIONS` | `16` | Builds running at once per tenant; others wait without a running deadline |
| `INGEST_VERIFY_CONCURRENCY` | `2`, `1..32` | Uploads hashed at once per scheduler process |
| `INGEST_RETENTION_DAYS` | `7` | History kept for published notifications, the operation ledger and quarantine; also how long `Idempotency-Key`s are honored |
| `INGEST_EXTRACTION_MODE` | `text_only` | `text_only`, `recognize_unreadable`, `recognize_with_images` or `recognize_all`; recognition modes need an injected `Recognizer` |
| `INGEST_UNREADABLE_PAGES` | `keep_empty` | Pages with visual content but no usable text when not recognized: record them as `unrecognized`, or `fail` the build |
| `INGEST_MIN_TEXT_CHARS`, `INGEST_RENDER_DPI`, `INGEST_STRICT_PDF` | `16`, `200`, `false` | Page classification and rendering; part of the recipe fingerprint |
| `INGEST_ALLOW_SIGNED_URL` | `false` | Let URL-fetching tools receive a short-lived read link |
| `INGEST_MIN_CONFIDENCE` | unset | Fail a build whose recognized pages report lower confidence |
| `INGEST_RECOGNITION_CONCURRENCY` | `4` | Recognition calls in flight per job (and polls per scheduler) |
| `INGEST_MAX_RECOGNIZED_PAGES` | `1000` | Billed pages allowed per document |
| `INGEST_TENANT_DAILY_RECOGNIZED_PAGES` | `0` (no limit) | Billed pages per tenant per UTC day |
| `INGEST_RECOGNITION_POLL_SECONDS` | `15` | Minimum interval between polls of job-style tools |
| `INGEST_VECTOR_TARGET` | `pgvector-default` | Logical vector target identity |
| `INGEST_VECTOR_INDEX` | `hnsw` | `hnsw` or `none` (exact search only) |
| `INGEST_HNSW_M`, `INGEST_HNSW_EF_CONSTRUCTION` | `16`, `64` | HNSW build parameters (ef_construction ≥ 2 × m). Applied when the index is built; to change them, `DROP INDEX CONCURRENTLY rag_vectors_hnsw` and run `migrate` (a mismatch is logged as `vector_index_settings_differ`) |
| `INGEST_HNSW_EF_SEARCH` | `200`, `1..1000` | Candidates examined per search: higher means better recall, slower search. At 300k clustered vectors, 100 left ~9% of a large tenant's queries with no correct result; 200 cut that to 0.5–2% for ~10–20 ms more. Tune with `rag-ingest search-eval` |
| `INGEST_HNSW_MAX_SCAN_TUPLES` | `20000` | How far an iterative scan may walk past other tenants' vectors before the exact search takes over |
| `INGEST_KEYWORD_INDEX` | `none` | `none` (detached) or `postgres` (built-in full-text search over the chunk table). An injected `KeywordIndex` (`build_engine(keyword=...)`) replaces the built-in. With one configured, every new build is written to it and verified before publishing |
| `INGEST_KEYWORD_LANGUAGE` | `english` | PostgreSQL text-search configuration (`simple` for mixed languages). Part of the keyword target: changing it needs `rag-ingest reindex-keyword` |
| `INGEST_SEARCH_HYBRID` | `false` | Default search mode: `false` = vector only, `true` = vector + keyword fused. Requires a keyword index. Each request can choose `mode` |
| `INGEST_SEARCH_FUSION`, `INGEST_SEARCH_RRF_K` | `rrf`, `60` | `rrf` combines ranks (`weight / (k + rank)`), robust to different score scales; `weighted` adds min-max normalized scores |
| `INGEST_SEARCH_VECTOR_WEIGHT`, `INGEST_SEARCH_KEYWORD_WEIGHT` | `1.0`, `1.0` | Relative weight of each ranking in fusion |
| `INGEST_SEARCH_KEYWORD_TIMEOUT_SECONDS` | `5` | Keyword search time limit; on timeout or failure the vector results are returned with `degraded: ["keyword"]` |
| `INGEST_SEARCH_STRICT` | `false` | `true`: a failing retriever fails the search instead of degrading |
| `INGEST_SEARCH_RERANK` | `false` | Reorder the top candidates with an injected `Reranker` (`build_engine(reranker=...)`; none is bundled). Each request can set `rerank` |
| `INGEST_RERANK_TOP_N` | `50`, `1..200` | Candidates gathered and reranked before cutting to the requested limit (also capped by the reranker's declared maximum) |
| `INGEST_RERANK_TIMEOUT_MS` | `1500` | Reranker time limit; on timeout, failure or an invalid reply the fused order is returned with `degraded: ["rerank"]` |
| `INGEST_LOG_FORMAT` | `json` | CLI log output: `json` (one object per line) or `text` |
| `INGEST_LOG_LEVEL` | `INFO` | `DEBUG` adds per-delivery acknowledgements |

Execution policy variables are listed in `.env.example`: maximum attempts, retry base/cap, job/attempt duration, lease and heartbeat durations, and generation deadline. They are copied into accepted work. Editing defaults does not reset previously accepted job budgets.

The vector target registry binds the configured embedding recipe to a target and rejects drift. Workers also reject a different embedding recipe from their accepted work. Do not change the configured model, target or dimensions while old generations are still being processed. This first version binds a runtime to one target/dimension; use a coordinated migration or separate stack for a change.

### Production settings: what to change

Fixed limits that are **not** settings (S3 multipart rules, sizes of files the service writes and reads back, batch and chunk caps) live in `src/rag_ingest/domain/limits.py`, each with what fixes it. Changing one needs a reviewed migration plan, because work saved under the old value must stay readable.

The defaults are safe for a laptop. For a production deployment with many users and thousands of uploads, review these. Everything else can stay at its default.

**Where to run**

| Change | Why |
| --- | --- |
| Run API, scheduler and workers in the **same AWS region as the S3 bucket and SQS queue** | Each stage makes several S3 and SQS calls. From a laptop each call takes ~400 ms; in-region it takes ~10–30 ms. This is the largest latency difference (measured: a small PDF took ~30 s from upload to Ready from a Mac, mostly waiting on the network). |
| Run `rag-ingest migrate` before rolling out new workers | Revisions 0004–0006 add the batch plan column, duplicate-lookup indexes and the abandoned-upload marker. Indexes are built `CONCURRENTLY`, so tables keep accepting writes. |

**Throughput**

| Setting | Default | Production guidance |
| --- | --- | --- |
| `INGEST_WORKER_CONCURRENCY` | `2` (max `64`) | Jobs in flight per worker process. Stages mostly wait on S3, SQS, OpenAI and the database, so 8–16 per process is reasonable. When every slot is busy, ready jobs wait (measured: 3–5 s waits with 4 slots shared by 6 documents). For more CPU (PDF parsing runs in a subprocess per job), add worker processes instead of raising this further. |
| `INGEST_VERIFY_CONCURRENCY` | `2` (max `32`) | Uploads verified at once per scheduler process. Clients that **do not** send `sha256` make the server download and hash every byte of each upload, so with many such uploads raise this to 8–16. Uploads that declare `sha256` (single PUT, PDF) verify with one HEAD and a 4 KiB read and barely use this. |
| `INGEST_RECOGNITION_CONCURRENCY` | `4` | Recognition calls in flight per job, and polls per scheduler. Raise only as far as your OCR provider's rate limit allows. |
| `INGEST_RECOGNITION_POLL_SECONDS` | `15` | How soon a finished job-style OCR result is picked up. Lower means faster resumes but more provider polls. |
| `INGEST_EMBEDDING_TIMEOUT_SECONDS` | `30` | Raise for a slower or self-hosted embedding model; each HTTP attempt gets two thirds of this. |
| `INGEST_EMBEDDING_MAX_BATCH_TOKENS` | `200000` | Set to your embedding provider's per-request token limit. |

**Database connections**

| Setting | Guidance |
| --- | --- |
| `INGEST_DB_POOL_SIZE` | Leave at `0`: it is derived from the concurrency settings above, and a configured value that is too small is refused at startup. Each pool may burst to 2×, so size PostgreSQL `max_connections` for **all** processes together: about processes × 2 × pool. With many processes, put a connection pooler in front and test it in staging first; tenant locks are transaction-scoped (`pg_advisory_xact_lock`). |

**Clients**

| Change | Why |
| --- | --- |
| Send `sha256` when starting every upload | A duplicate is refused before any bytes are uploaded (HTTP 409 `duplicate_content`), and a single-PUT PDF is verified without a second full download. Computing it is one line in any language (`hashlib.sha256`, `shasum -a 256`). Without it, duplicates are still caught, but only after the upload. |
| Browser clients: add an S3 CORS rule for **your own origin only** | Allow `PUT` with headers `content-type`, `if-none-match`, `x-amz-checksum-sha256`; see `live_test/web/s3-cors.json` for the shape. |

**Search database** (measured in section 13: 300,000 chunks of 1,536 dimensions)

| Change | Why |
| --- | --- |
| Give PostgreSQL memory for the vector index | 300k vectors take ~4.9 GB of table and ~2.3 GB of HNSW index (~24 KB per chunk in all, text included). Searches are fast only while the index is in memory: size RAM (`shared_buffers` plus OS cache) for the index of your largest expected data, roughly 8 GB per million 1,536-dimension chunks. |
| Set `maintenance_work_mem` (≥ 2 GB) and `max_parallel_maintenance_workers` at the database or role level before `rag-ingest migrate` builds the index over existing rows | Measured: 59–95 s for 300k vectors with 2 GB and 7 workers. With PostgreSQL's 64 MB default the graph does not fit in memory and the build is much slower. |
| Keep `INGEST_HNSW_EF_SEARCH` at `200` or higher, and check `search-eval` on your own data | See the configuration table: on clustered data, a lower value makes some queries miss entirely. Raising `m` or `ef_construction` did not fix that; only `ef_search` did. |
| Above ~20,000 chunks per tenant, use a BM25 engine for keyword search | The built-in `postgres` full-text index scores every matching row: measured p95 ~0.2 s at 20k chunks but ~1.2 s at 150k, and hybrid search waits for it. The ParadeDB example (`examples/paradedb_keyword.py`) measured 16.5 ms p95 at 150k on the same data, with better ranking; it needs the `pg_search` extension, which many managed services such as RDS do not offer. There, use an OpenSearch/Elasticsearch `KeywordIndex` (section 8). |

**Cost and data hygiene**

| Setting | Default | Guidance |
| --- | --- | --- |
| `INGEST_ALLOW_DUPLICATES` | `false` | Keep `false`, so identical files are not processed and billed twice. Clients that need a second copy send `allow_duplicate: true`. |
| `INGEST_MAX_RECOGNIZED_PAGES`, `INGEST_TENANT_DAILY_RECOGNIZED_PAGES` | `1000`, `0` (no limit) | Set a daily per-tenant page limit if OCR is billed per page. |
| `INGEST_ABANDONED_UPLOAD_GRACE_SECONDS` | `3600` | Files of refused, failed or expired uploads are erased this long after the upload session expires. The default is fine. |
| `INGEST_RETENTION_DAYS` | `7` | How long published notifications, the idempotency ledger and quarantine records are kept. |
| `INGEST_LOG_FORMAT` | `json` | Keep JSON for log aggregation. |

### Production security checklist

The package enforces tenant isolation, create-only storage writes, pinned source versions, lease checks and verified publication by itself. These remaining controls belong to your deployment.

**Identity and access**

- Replace the static token map with your identity provider. `create_app(authenticate=...)` takes a function that receives the bearer token and returns the tenant UUID; derive the tenant from a verified claim, never from the request:

  ```python
  async def authenticate(token):
      try:
          claims = verify_jwt(token, issuer=..., audience=...)  # your OIDC/JWT library
      except InvalidToken:
          return None                          # the API answers 401
      return UUID(claims["tenant_id"])         # anything but a UUID is also a 401

  app = create_app(settings=settings, authenticate=authenticate)
  ```

  Return `None` for a bad token; let real failures (identity provider unreachable) raise, so they surface as server errors instead of looking like bad credentials.
- Never put long-lived service tokens in browsers; browser clients should use short-lived user tokens. The static tokens in `INGEST_API_TOKENS` are tenant-wide.
- Serve the API only behind HTTPS (terminate TLS at your load balancer). The API binds `0.0.0.0` so containers can reach it; do not expose that port directly.
- Keep `INGEST_API_DOCS=false` in production, and set `INGEST_API_UPLOADS_PER_MINUTE` / `INGEST_API_SEARCHES_PER_MINUTE` to what one tenant may reasonably use. A gateway or WAF in front can add per-IP limits; the package's limits are per tenant.

**Database**

- Require TLS: add `?sslmode=verify-full&sslrootcert=...` (or at least `sslmode=require`) to every remote DSN. A remote DSN without it logs `database_tls_not_required` at startup, because the driver's default silently falls back to an unencrypted connection.
- Use two roles: one for `rag-ingest migrate` (creates tables, indexes and the `vector` extension) and a runtime role for the API, scheduler and workers that can only read and write rows. The services never change the schema.
- Enable automated backups and test a restore.

**S3**

- Keep Block Public Access on and versioning enabled (required).
- Use bucket default encryption (SSE-S3, or SSE-KMS with a key the service role may use). The service sends no encryption headers, so a bucket policy that rejects uploads without them would refuse its writes.
- Add a bucket policy that denies requests without TLS (`aws:SecureTransport` = false).
- Add a lifecycle rule `AbortIncompleteMultipartUpload` (for example after 7 days) for the one multipart case the service's own cleanup cannot see. Do not add expiry rules for objects under `tenants/`: accepted sources are pinned versions.
- CORS only for browser clients, only for your own origin, only `PUT` (see section 6, Clients).

**Least-privilege IAM for the service role**

- S3 on the bucket: `s3:GetObject`, `s3:GetObjectVersion`, `s3:PutObject` (also covers multipart create, upload part and complete), `s3:DeleteObject`, `s3:DeleteObjectVersion`, `s3:ListBucket`, `s3:ListBucketVersions`, `s3:ListMultipartUploadParts`, `s3:AbortMultipartUpload`.
- SQS on the queue: `sqs:SendMessage`, `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:ChangeMessageVisibility`, `sqs:GetQueueAttributes`.
- Plus whatever your recognizer needs (for example Textract) and nothing else. Prefer an instance or task role over access keys.

**Secrets**

- Load `INGEST_EMBEDDING_API_KEY`, `INGEST_API_TOKENS` and database passwords from a secrets manager or your orchestrator's secret store, not from a `.env` file on disk. `.gitignore` excludes `.env`; never commit it.
- Rotate any key that was ever pasted into a chat, ticket or log.

**Untrusted documents**

- Run workers in containers with CPU, memory and disk limits, no inbound network, and outbound access only to AWS, your model provider and the database. The parser subprocess is bounded in time and memory but is not a security sandbox.
- If you build chat on top of search, treat document text as untrusted input to the model: a document can contain instructions aimed at the model. Keep answers grounded in cited sources, as the demo in `live_test/` does.

**Supply chain**

- `requirements-lock.txt` records the exact dependency versions this release was tested with; install with `pip install -c requirements-lock.txt -e '.[all]'`. The Postgres image in `docker-compose.yml` is pinned by digest.
- Run `ruff check src tests live_test examples` in CI (the `[tool.ruff]` config in `pyproject.toml` includes the bandit security rules) and a dependency vulnerability scanner such as `pip-audit`.

## 7. API operations

Every `/v1` operation requires `Authorization: Bearer TOKEN`. The tenant is never accepted as a caller-controlled JSON field. `/healthz` is public liveness; authenticated `/readyz` checks SQL and bundled-role health.

| Method and path | Purpose |
| --- | --- |
| `POST /v1/uploads` | Register `{filename,size_bytes,sha256?,document_id?,allow_duplicate?,tags?,metadata?}` (labels only for a new document) with `Idempotency-Key`; return upload and transfer plan/grant. With `sha256`, identical content already held by a live document returns **409** `{"error":"duplicate_content","duplicate_of":{"document_id","filename"}}` and nothing is uploaded |
| `GET /v1/uploads/{id}` | Safe upload status: `uploading`, `verifying`, `accepted`, `failed` or `expired`. A copy refused at verification is `failed` with `duplicate_of` set |
| `POST /v1/uploads/{id}/parts/{number}` | Return fresh part grant; body `{}` or `{sha256}` |
| `GET /v1/uploads/{id}/parts?after=N&limit=1000` | Provider-confirmed part page and advancing cursor |
| `POST /v1/uploads/{id}/complete` | Complete/pin object; return `202` verification status |
| `GET /v1/documents/{id}` | Active/desired generation and safe job reports |
| `GET /v1/jobs/{id}` | Stage, attempt, state, next-attempt time and failure code |
| `GET /v1/documents/{id}/view?download=false` | Authorized version-pinned read grant |
| `POST /v1/documents/{id}/reprocess` | Explicit fresh generation/budget from accepted source; requires `Idempotency-Key` |
| `DELETE /v1/documents/{id}` | Tombstone now, physical cleanup later; requires `Idempotency-Key` |
| `POST /v1/search` | `{query,limit,filters?,mode?,rerank?}` (limit 1..100; `mode` = `vector`, `keyword` or `hybrid`; `rerank` overrides `INGEST_SEARCH_RERANK`). Filters: `document_ids`, `tags_any`, `tags_all`, `metadata` (exact values), `created_after`/`created_before` (with timezone). Returns distinct passages with `filename`, `text`, `score`, `pages`, `spans`, `also_in` (other files and pages with the identical passage) and `ranks` (rank in each retriever that found it); the response also has `mode` and `degraded` (retrievers that failed while the other answered) |
| `PATCH /v1/documents/{id}` | `{tags?,metadata?}` replaces a document's labels; omitted fields stay; takes effect for the next search |

Duplicate detection compares the exact bytes (SHA-256) with documents in the same tenant that are not deleted and whose published or in-progress build is not failed. Clients that send `sha256` are refused before uploading; without it the copy is uploaded, refused at verification, and its file erased later by the abandoned-upload cleanup. Send `allow_duplicate: true` to keep a second copy on purpose.

`reprocess` uses the current configured recipe and deliberately authorizes another build. It is different from redelivering a failed job notification. Inspect uncertain model outcomes before authorizing work that may repeat paid calls.

## 8. Supported services and extension points

The package ships default adapters for one stack (AWS S3, AWS SQS, PostgreSQL with pgvector, OpenAI embeddings). Every one of them sits behind a protocol in `src/rag_ingest/contracts/`, so a different service needs an adapter, not a fork. **Tested** below means exercised in this project against the real service; **should work** means the service is expected to meet the listed requirements but has not been tested here; run your own failure tests before relying on it.

### Object storage (sources, stage outputs, manifests)

| Option | Status | Notes |
| --- | --- | --- |
| AWS S3 | **Tested** (real bucket, us-east-1) | Built in: `S3ObjectStore`. Bucket versioning must be enabled |
| In-process S3 (`moto`) | **Tested** (automated suite) | Request handling only, not real S3 semantics |
| S3-compatible endpoints (LocalStack, self-hosted, other clouds) | Should work **only if** every requirement below holds | Set `INGEST_S3_ENDPOINT_URL` (and `INGEST_S3_PUBLIC_ENDPOINT_URL` for browser clients). LocalStack is in `docker-compose.yml` but not validated |
| Azure Blob, Google Cloud Storage native APIs, local disk | Needs an adapter | Implement `ObjectStore`, `UploadStore` and `CleanupObjectStore`; optionally `StorageVerifiedObjectStore` |

An S3-compatible service must support: object versioning and reads of an exact version; listing object versions; create-only writes (`If-None-Match: *` on PutObject and CompleteMultipartUpload); `x-amz-checksum-sha256` on PutObject; multipart uploads with listable parts; SigV4 presigned URLs. A service without versioning cannot be used for the default intake, which pins every accepted source to a version. "S3 compatible" alone does not guarantee these.

### Queue (job notifications)

| Option | Status | Notes |
| --- | --- | --- |
| AWS SQS standard queue | **Tested** | Built in: `SQSQueue`, including batch sends (`publish_batch`) |
| SQS FIFO queue | Not supported | Refused at startup; ordering is not needed and FIFO limits throughput |
| SQS-compatible endpoints (LocalStack, ElasticMQ) | Should work, untested | `INGEST_SQS_ENDPOINT_URL` |
| RabbitMQ, Kafka, Redis Streams, Google Pub/Sub, Azure Service Bus | Needs an adapter | Implement `Queue`: at-least-once delivery, a per-delivery receipt, acknowledge, extend visibility; optionally `BatchQueue` |

Delivery may be duplicated or lost; correctness comes from the PostgreSQL job state and outbox, not from the queue.

### Database (metadata and vectors)

| Option | Status | Notes |
| --- | --- | --- |
| PostgreSQL 16 + pgvector 0.8.7 | **Tested** | Built in: `PostgresMetadataStore` (metadata) and `PgVectorStore` (vectors, HNSW cosine index with exact verification) |
| Managed PostgreSQL with the pgvector extension (for example Amazon RDS / Aurora PostgreSQL) | Should work, untested | Needs JSONB, `pg_advisory_xact_lock`, `CREATE INDEX CONCURRENTLY`, the `sha256()` function and the `vector` type. Run `rag-ingest migrate` |
| Separate vector database (Qdrant, Pinecone, Weaviate, OpenSearch, Milvus) | Needs an adapter | Implement `VectorStore`, plus `SearchableVectorStore.search_hits` (IDs and scores only) to serve `/v1/search`; add `DigestVerifiableVectorStore` for fast publication checks. Results are always read from the chunk table in the metadata database after the live-document check, so any store works with the bundled search |
| Other metadata databases (MySQL, DynamoDB, MongoDB) | Not practical | `RuntimeMetadataStore` relies on transactional, lease-guarded state changes; reimplementing it is a large project |

### Text extraction and OCR

| Option | Status | Notes |
| --- | --- | --- |
| PDF with a text layer, UTF-8 `.txt` | **Tested**, built in | `TextExtractor` (pypdf), run in a killable subprocess |
| Scanned or image PDF pages | Supported through **your own** `Recognizer` | No OCR tool is bundled, by design. Works with page-image tools (Tesseract, vision models) and whole-document tools (Textract, Mistral OCR style), synchronous or job-based. Template: `examples/recognizer_template.py`; check it with `rag_ingest.testing.RecognizerConformance` |
| AWS Textract | **Tested** as an example adapter | `live_test/textract_recognizer.py`, not part of the package |
| Word, HTML, Markdown, CSV/Excel, images, e-mail, archives | Needs an `Extractor` | Not accepted by the built-in intake today |

Recognized text is stored per page with an optional confidence. Layout, tables and form fields are not kept as structure; they become plain text.

### Keyword search

| Option | Status | Notes |
| --- | --- | --- |
| PostgreSQL full-text search | **Tested**, built in | `INGEST_KEYWORD_INDEX=postgres`; ranking `ts_rank_cd` (not BM25) |
| OpenSearch / Elasticsearch (BM25) | Example adapter, request shapes tested; not run against a live cluster | `examples/opensearch_keyword.py`; run `KeywordIndexConformance` against your cluster |
| ParadeDB `pg_search` (BM25) | Example adapter, **tested**: conformance kit, the whole hybrid pipeline, and a 300k-chunk benchmark on `pg_search` 0.26.3 | `examples/paradedb_keyword.py`; same database as the chunks, no extra service; call `provision()` once to build its index |
| Typesense, Tantivy, Vespa | Needs a `KeywordIndex` adapter | Prove it with `rag_ingest.testing.keyword_conformance` |

**Which one to choose.** The default is `none`: keyword search is off and search is vector-only until you pick one. The package builds in only the engine that needs nothing extra; vendor engines are adapters you inject with `build_engine(keyword=...)`.

| Your situation | Use | Why |
| --- | --- | --- |
| Any PostgreSQL, tenants up to ~20,000 chunks | `INGEST_KEYWORD_INDEX=postgres` | Built in, works everywhere (RDS and Aurora too); p95 ~0.2 s at 20k chunks |
| PostgreSQL where you can install `pg_search` (self-hosted, ParadeDB image), larger tenants | `examples/paradedb_keyword.py` | True BM25; p95 16.5 ms at 150k chunks (section 13) |
| Managed PostgreSQL without `pg_search`, larger tenants | An OpenSearch/Elasticsearch `KeywordIndex` | e.g. Amazon OpenSearch Service; start from `examples/opensearch_keyword.py` |

With any of them, set `INGEST_SEARCH_HYBRID=true`: hybrid found the right passage more often than vector-only search, both in the live test and in the benchmark.

### Reranking

| Option | Status | Notes |
| --- | --- | --- |
| No reranker | Default | Fused (or single-retriever) order |
| Cohere-style or Hugging Face TEI rerank API (cross-encoders such as bge-reranker) | Example adapter, request shapes tested; not run against a live service | `examples/http_reranker.py` |
| An OpenAI chat model scoring passages | **Tested** live as application code | `live_test/openai_reranker.py`; slower and costlier per call than a cross-encoder |
| Voyage, Jina, Bedrock, a local model | Needs a `Reranker` adapter | Prove it with `rag_ingest.testing.rerank_conformance` |

### Chunking and embeddings

| Option | Status | Notes |
| --- | --- | --- |
| Character chunks with overlap | **Tested**, built in | `CharacterChunker`; `INGEST_CHUNK_SIZE`, `INGEST_CHUNK_OVERLAP` |
| Token, sentence, heading or semantic chunking | Needs a `Chunker` | Must be deterministic and keep page/character source spans |
| OpenAI embeddings (`text-embedding-3-small`, 1536 dimensions) | **Tested** | Built in: `OpenAIEmbedder` |
| OpenAI-compatible embedding servers | Should work if they accept `dimensions` and `encoding_format: "float"` and return `index` per item | `INGEST_EMBEDDING_BASE_URL`, `INGEST_EMBEDDING_TIMEOUT_SECONDS`, `INGEST_EMBEDDING_MAX_BATCH_TOKENS` |
| Azure OpenAI, Cohere, Voyage, Bedrock, local models | Needs an `Embedder` | Different URLs, auth or response shapes. Example: `examples/company_embedder.py` |

### Ways to run it

| Mode | How |
| --- | --- |
| HTTP API plus background roles | `rag-ingest serve` (all-in-one) or `api`, `publisher`, `scheduler`, `worker` as separate processes |
| Python library | `build_engine(settings, metadata=..., objects=..., queue=..., extractor=..., chunker=..., embedder=..., vectors=..., recognizer=...)` |
| Custom adapters | The CLI uses the built-in adapters only. To inject a recognizer or other adapter, start the roles from your own small program, as `live_test/app.py` and `examples/custom_app.py` do |

## 9. Build a custom adapter: a different service example

The example here is a **hypothetical internal company embedding gateway**, not S3, SQS, OpenAI or pgvector. Its API contract is defined for teaching; no actual gateway service is included.

Its request is `POST /encode` with model, ordered texts, dimensions and a token ceiling. Its response is an ordered vector list. For this example only, its documented tokenizer defines one UTF-8 byte as one token.

See the complete, runnable adapter code in [examples/company_embedder.py](examples/company_embedder.py). Its small steps are:

1. Verify that the saved pipeline selects `company_http`, implementation version `1`, and supported model/dimension/tokenizer options.
2. Enforce batch, input and token limits before making a request.
3. Use a bounded asynchronous HTTP client and outer request deadline.
4. Classify connection rejection, throttling, input errors, credential errors and unknown post-dispatch outcomes separately.
5. Require one result for every input; construct validated `EmbeddedChunk` records with original identity/provenance.
6. Validate exact input coverage, dimensions and pipeline fingerprint before returning.
7. Close its HTTP pool during engine shutdown.

The wiring in [examples/custom_app.py](examples/custom_app.py) changes the durable recipe and replaces only the embedder:

```python
from dataclasses import replace
from rag_ingest.runtime.factory import build_engine
from rag_ingest.runtime.settings import ServiceSettings
from company_embedder import CompanyEmbedder

settings = replace(
    ServiceSettings.from_env(),
    embedding_driver="company_http",
    embedding_model="company-text-v2",
    embedding_encoding="utf8-bytes",
)

engine = build_engine(
    settings,
    embedder=CompanyEmbedder(
        base_url="https://embedding.internal.example",
        token=company_token,  # Obtain from your secret manager.
    ),
)
```

The existing S3, SQS, PostgreSQL and pgvector defaults remain supplied by the factory. Configure the correct dimensions before provisioning the vector table. Run all worker processes with the custom factory too; changing only the API process would leave workers unable to execute the new saved recipe. The supplied `custom_app.py` uses combined local mode to keep that example self-contained.

Do not set `INGEST_EMBEDDING_DRIVER=company_http` in the built-in CLI: it intentionally accepts only its built-in drivers. A custom application constructs settings and injects its implementation as above.

### Recognizing scanned and image pages with your own tool

The package has **no built-in OCR or vendor dependency**. Any tool works: a local OCR model, a cloud document API, a multimodal LLM. You implement one small interface, `contracts.recognition.Recognizer`, and **declare what your tool can do**. The package adapts to that.

```python
class MyRecognizer:
    spec = ComponentSpec(driver="my_tool", implementation_version="1",
                         options_json='{"model": "v3-2026-01"}')   # pinned, no secrets
    capabilities = RecognizerCapabilities(
        unit="document",          # or "page_image": one rendered page per request
        call_style="sync",        # or "job": submit() now, poll() later
        inputs=frozenset({"bytes"}),  # path | bytes | storage_ref | signed_url
        page_selection=False, max_pages_per_request=50,
        max_bytes_per_request=20 * 1024**2, max_requests_per_second=5,
        output_format="markdown")
    async def recognize(self, request): ...   # sync tools
    async def submit(self, request): ...      # job tools
    async def poll(self, job): ...            # job tools; job.page_numbers maps positions
    async def aclose(self): ...

engine = build_engine(settings, recognizer=MyRecognizer(...))   # INGEST_EXTRACTION_MODE=recognize_unreadable
```

Start from [examples/recognizer_template.py](examples/recognizer_template.py), then run the conformance kit against your real tool:

```python
from rag_ingest.testing.conformance import RecognizerConformance
class MyConformance(RecognizerConformance, unittest.IsolatedAsyncioTestCase):
    def make_recognizer(self): return MyRecognizer(...)
```

**What the package does for every tool:**

| Concern | Behavior |
| --- | --- |
| Which pages | Per recipe mode. A page needs recognition when it has visual content (images, or a large drawing stream) and no usable native text (fewer than `INGEST_MIN_TEXT_CHARS` visible characters, or a garbage text layer). `recognize_with_images` also includes text pages that contain images; `recognize_all` includes every PDF page. |
| No silent drops | Every page records its method in the page map: `text`, `recognized`, `blank` or `unrecognized`. `unrecognized` pages are kept empty with a warning, or fail the build (`INGEST_UNREADABLE_PAGES=fail`). |
| Fitting the tool | `page_image` tools get rendered pages within `max_image_pixels`/`max_bytes_per_request` (PNG, else JPEG, else smaller). `document` tools get the whole file, the file plus a page list (`page_selection`), or a page-subset PDF; requests are split to the page and byte limits and halved until they fit. Results map back to original page numbers and must cover exactly the requested pages. |
| Inputs | The most private form the tool accepts: `path`, then `bytes`, then `storage_ref` (the exact pinned object version; subsets are stored under the build's prefix), then `signed_url` only with `INGEST_ALLOW_SIGNED_URL=true`. |
| Cost | Every finished page is cached per build, so retries never re-bill it. `INGEST_MAX_RECOGNIZED_PAGES` caps a document; `INGEST_TENANT_DAILY_RECOGNIZED_PAGES` caps a tenant per UTC day (idempotent per request). `max_requests_per_second` is enforced by a token bucket shared by all processes in PostgreSQL. |
| Job-style tools | After `submit()` the provider job id is recorded durably, then the stage job is parked as `waiting`: no worker, no lease, no attempt spent. The scheduler polls, caches finished pages, and re-queues the job when nothing is pending. Only expired or retryable jobs are resubmitted, at most 3 times per page. Waiting is bounded by the job deadline (`INGEST_MAX_JOB_SECONDS`). Deleted or superseded documents abandon their provider jobs. |
| Sync tools | Calls run concurrently; finished pages are saved before any error. Near the attempt deadline no new calls start and progress continues in the next attempt. |
| Identity | The tool's `spec` and output-affecting capabilities are part of the recipe fingerprint: changing the tool or model makes a new build. A worker whose recognizer differs from the build's recipe refuses it (`operator_action_required`). |
| Failures | Your errors map to job outcomes: `RateLimitedError`/`RetryableError` retry with backoff, `UncertainOutcomeError` gets bounded repeats, `PermanentInputError` fails the build (`recognition_failed`), `OperatorActionRequiredError` stops the role. Results that do not match the request fail as `recognizer_contract_violation`. |
| Privacy | Documents leave your infrastructure only through the recognizer you inject. Request records carry tenant and document IDs (never content) so an adapter can choose per-tenant credentials or regions. |

Run **workers and the scheduler** with the same recognizer: job-style tools are polled by the scheduler. Page rendering needs `pip install 'rag-ingest[ocr]'` (pypdfium2 and Pillow); a missing extra stops the role with an actionable message. Text-only PDFs and `.txt` files never call the recognizer.

### Replacing other components

`build_engine` accepts `metadata`, `objects`, `uploads`, `queue`, `extractor`, `chunker`, `embedder` and `vectors` overrides. Protocols and records are in `contracts/` and `domain/`.

| Replacement | Required behavior |
| --- | --- |
| Queue | Bounded receive; current receipt handles; explicit acknowledgement and visibility; classified send uncertainty |
| Object/upload store | Bounded streams; exact immutable references; create-only output writes; multipart part truth; signing expiry; pinned versions in the bundled intake path |
| Extractor | Killable/bounded parsing; deterministic ordered pages; explicit unsupported/corrupt outcomes; a page method and `needs_recognition` per page |
| Recognizer | Declared capabilities; exactly the requested pages; contract error categories; pinned model in `spec`; see the conformance kit |
| Chunker | Deterministic ordinal/text/source spans; compatible saved recipe; bounded output |
| Embedder | Exact per-input coverage; finite vectors and dimensions; correct tokenizer; classified partial/uncertain outcomes |
| Vector store | Tenant/document/generation namespaces; stable immutable IDs; exact verification/query readiness; scoped idempotent deletion |
| Metadata | All guarded atomic transitions plus idempotency, authoritative time, lease CAS behavior and durable outcomes |

The `RuntimeMetadataStore` protocol extends the introductory `MetadataStore` with service-level operations: upload initiation/status, saved recipe reads, sealed inventory scheduling, recovery, quarantine, cleanup plans, and authorized search integration. Implement both protocol surfaces when replacing metadata, and inspect runtime calls for exact integration requirements. A dictionary with `get()`/`set()` is insufficient. The factory does not provide an automatic adapter-conformance certification.

The built-in janitor needs version-aware cleanup, so an injected object store must also implement `contracts.storage.CleanupObjectStore` (`delete_prefix_versions`, `delete_all_key_versions`). `build_engine` rejects a store without them, because deleted documents' vectors and files would otherwise never be erased. The generic `ObjectStore.delete()` alone cannot enumerate a generation's versions. If retention is managed elsewhere, implement them deliberately as documented no-ops.

“S3 compatible” is not proof of support for conditional multipart completion, signed checksum/length headers, exact version reads, or version enumeration. Run compatibility and failure tests against your chosen provider. Never disable required invariants just to make an endpoint pass a superficial upload test.

## 10. Code map and recommended reading order

```text
src/rag_ingest/
  domain/                Validated application records and canonical manifests
  domain/limits.py       Fixed limits shared by writers and readers (not settings)
  contracts/             Adapter protocols, limits and shared error categories
  adapters/aws/          Default S3 object/transfer and SQS transport implementations
  adapters/postgres/     Transactional metadata, safe typed codec, schema and pgvector
  adapters/processing/   Killable PDF parser/renderer, character chunks, embedders
  runtime/settings.py    Validated service defaults and saved recipe construction
  runtime/factory.py     Default wiring and explicit component overrides
  runtime/engine.py      Upload verification, duplicate refusal and stage execution
  runtime/recognition.py Recognizing pages through an injected Recognizer
  runtime/embedding_segments.py  Compact, self-verifying embedding files
  runtime/search.py      Search pipeline: vector hits, live check, filters, collapse
  runtime/search_eval.py Recall@k and latency of the vector index vs exact search
  runtime/keyword_reindex.py  Add published builds to a newly enabled keyword index
  contracts/keyword.py   The KeywordIndex interface (BM25 or any keyword engine)
  adapters/postgres/keyword.py  Built-in keyword index (PostgreSQL full-text search)
  contracts/rerank.py    The Reranker interface (cross-encoders, rerank APIs, models)
  domain/search.py       Search filters, results and the chunk-text digest
  runtime/runners.py     Publisher, scheduler, leases, workers and cleanup loops
  api/app.py             Authenticated HTTP interface and local combined lifecycle
  testing/               Conformance kits (recognizer, keyword index, reranker), fakes, offline test embedder, sample PDFs
  migrations/            Packaged Alembic revisions 0001-0008
  logs.py                Structured, secret-safe logging
  cli.py                 Setup, doctor and process commands
examples/
  upload.py              Direct/resumable local upload client
  company_embedder.py    Custom adapter for a different hypothetical service
  custom_app.py          Wiring that adapter into the default stack
  recognizer_template.py Plugging in your own OCR or document-recognition tool
  paradedb_keyword.py    BM25 keyword index with ParadeDB pg_search (tested)
  opensearch_keyword.py  BM25 keyword index on OpenSearch/Elasticsearch
  http_reranker.py       Cohere-style or TEI reranking service
bench/
  search_scale.py        Search benchmark at 300k vectors (disposable database only)
live_test/               Live test against real AWS (not part of the package)
  app.py                 API, scheduler and workers with Textract injected
  demo.py, web/          Upload/status/chat demo page and grounded chat endpoint
  textract_recognizer.py Example Recognizer for AWS Textract
  load.py, verify.py     Bulk upload through the API and an end-to-end report
```

For learning, start with `domain/documents.py`, then `jobs.py`, `policy.py`, `outbox.py` and `uploads.py`. Read the relevant contract before its concrete adapter. Then follow `engine.initiate_upload() → complete_upload() → verify_upload()`. Finally read one processing stage and its runner. Docstrings explain what the operation does and why its checks exist.

## 11. Recovery, replacement and deletion

### Recovery rules

- A notification is a hint to inspect a durable job. Queue acknowledgement is not the source of truth.
- A valid checkpoint is reusable only after reading and verifying its manifest and outputs.
- Failed/lost ownership cannot commit stage transitions or activate a generation.
- Expired jobs get bounded reconciliation passes; unpublished/missing notifications are repaired from SQL.
- Safe short retries and durable retries share finite time/attempt limits; restarting a process does not create a fresh budget.
- A sealed chunk inventory makes batch completeness explicit. Scheduling cursor records prevent missing or duplicated logical batch jobs.
- **Embedding segments: vectors are written once.** Each attempt writes its new embeddings into one immutable segment (`embedding-cache/<batch>/NNN.seg`): a small JSON header with chunk identities, recipe and dimensions, the vectors as float32 (exactly what pgvector stores and publication verifies), and a SHA-256 trailer. The segment is both the progress cache (a retry reads earlier segments and embeds only what is missing, so nothing is billed twice) and the stage output: the stage manifest commits the segments that cover the batch by pinned S3 version, SHA-256 and size, and only the current lease holder can checkpoint it. Index and publish read through the manifest, verify every segment, and require every chunk exactly once. Segments are create-only; a concurrent attempt takes the next number, and a later object version under a committed key cannot change what readers see. A damaged or foreign uncommitted segment is skipped with an `embedding_cache_entry_ignored` warning and re-embedded; a committed segment that fails its pinned hash fails the build as corrupt. A first attempt costs one write of about 0.5 MiB for 64 × 1,536 dimensions (about 4 MiB of JSON before); only retries check for earlier segments. Cross-document model-result caching is not implemented.

### Replacement

A new accepted source changes `desired_generation_id`, leaving `active_generation_id` unchanged. The new build owns a separate vector namespace. Once all output is verified and publication still matches desired state, one transaction switches the active pointer and saves retirement cleanup intent for the old active generation. Retired active outputs have a 24-hour grace.

Concurrent intake uses the document baseline saved at upload initiation. If another source becomes desired first, the older conflicting upload cannot silently overwrite it at acceptance; start a new explicit replacement if that was intended.

### Deletion

Deletion immediately tombstones the document and clears active/desired visibility. Bundled search joins live document state and cannot return tombstoned content. View signing also rejects it. Running work cannot publish afterward.

Physical erasure is asynchronous. The janitor waits at least one hour plus a safety margin, or a longer configured maximum attempt duration, then checks leases, source references and generation visibility. It freezes a durable cleanup plan before deleting generation vectors, S3 generation-object versions and safe-to-delete source keys. Shared canonical bytes are retained while a live document still references them. Artifact state prevents acceptance from racing with an authorized deletion.

Previously issued presigned URLs remain usable until they expire unless your storage policy revokes them. Logical deletion is immediate in this application; global physical erasure, backup expiry, provider replication and external caches need separate deployment retention policy.

## 12. Operations and troubleshooting

| Symptom | Check and response |
| --- | --- |
| `local-init` rejects endpoints | Use explicit `http://localhost:4566` S3/SQS endpoints; it must not provision live AWS |
| Queue URL errors | Use the exact URL printed by LocalStack `local-init`; restart roles after editing `.env` |
| Port 5432/4566 is in use | Change Compose's host mapping and matching `.env` endpoints/DSN |
| Missing or old schema | Run `migrate` with the same metadata DSN as the service |
| Vector dimension mismatch | Use the correct database/recipe or a reviewed migration; do not silently overwrite existing vectors |
| Upload remains `uploading` | Finish transfer and call `/complete`; inspect multipart provider parts and grant expiry |
| Upload remains `verifying` | Ensure scheduler is running; check storage access, expiry and full-file read time; raise `INGEST_VERIFY_CONCURRENCY` if many large uploads queue |
| Job stays `queued` after acceptance | The build is waiting for admission: the tenant already has `INGEST_MAX_ACTIVE_GENERATIONS` builds running. Its deadline has not started; it is admitted oldest-first as builds finish |
| Upload fails verification | Compare expected size/SHA-256/type with actual original; new file needs a new request key |
| Job is `retry_wait` | Inspect `next_attempt_at`; scheduler must release it; there is intentionally no worker lease while waiting |
| Job never starts | Check publisher, scheduler and workers; queue/SQL credentials; outbox pending state |
| Job fails `outcome_requires_inspection` | A remote outcome or timeout stayed uncertain after the allowed repeats; inspect provider output/billing before explicit reprocessing |
| Job shows `uncertain_outcome_repeat` | An uncertain outcome or timeout is being repeated; each repeat may re-bill a model call (`uncertain_outcome_repeat` log events count them) |
| Job fails `operator_action_required` | Fix credentials, endpoint, model access or schema; restart the stopped role and explicitly authorize recovery as needed |
| Scanned pages come out empty (`unrecognized`) | No recognizer is configured: inject a `Recognizer` and set `INGEST_EXTRACTION_MODE`; set `INGEST_UNREADABLE_PAGES=fail` to reject such documents instead |
| Job shows `waiting` / `awaiting_recognition` | A job-style recognizer is processing; the scheduler polls it. Ensure the scheduler runs with the same recognizer |
| PDF is large but extraction fails | Upload capacity differs from parser memory/text/time capacity; increase limits only after resource tests |
| Search has irrelevant results in local mode | Artificial test vectors have no semantics; configure a real embedding model |
| Search fails with a separate vector DSN | Bundled SQL search joins `documents`; use the same DB or implement an authorized search path for your topology |
| Deleted object still physically exists | Check cleanup grace, scheduler, live shared references, provider retention and failed cleanup attempts |

### Logging

Logs are structured around operation/job IDs and safe categories. They avoid raw document bodies, tokens, DSNs, signed URLs and provider response dumps.

- **CLI roles** (`api`, `serve`, `worker`, `publisher`, `scheduler`) configure output from `INGEST_LOG_FORMAT` and `INGEST_LOG_LEVEL`. The API uses the same format as the workers.
- **Levels mean something.** `INFO` is normal progress (`job_started`, `upload_accepted`, `cleanup_completed`). `WARNING` is recoverable trouble (`job_failed`, `uncertain_outcome_repeat`, `*_dependency_pending`, `lease_renewal_pending`). `ERROR` needs attention (`job_unexpected_error`, `worker_unexpected_error`, `cleanup_blocked_unsupported_store`, `background_role_failed`, `command_failed`). Alert on `ERROR`, and on sustained `WARNING` rates.
- **Exceptions** include the stack (file, line, function) and every exception type in the chain. Only this package's own error messages are printed, because their text is fixed; third-party messages can contain DSNs, signed URLs or document text, so they are omitted.
- **As a library**, the package only emits records under the `rag_ingest` logger hierarchy and installs no handlers. Configure logging in your application, or call `rag_ingest.logs.configure_logging(level="INFO", fmt="json")`. Each record also carries `record.event` and `record.fields` for your own structured formatter. Quarantine stores bounded malformed-message evidence in SQL, so protect that table as operational data. It is not exposed through the API.

For an explicit new build after inspection:

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/documents/YOUR_DOCUMENT_ID/reprocess \
  -H "Authorization: Bearer $INGEST_DEMO_TOKEN" \
  -H 'Idempotency-Key: reviewed-reprocess-001'
```

For logical deletion:

```bash
curl -sS -X DELETE http://127.0.0.1:8080/v1/documents/YOUR_DOCUMENT_ID \
  -H "Authorization: Bearer $INGEST_DEMO_TOKEN" \
  -H 'Idempotency-Key: delete-001'
```

**Upgrades.** A release reads rows written by older releases (record fields are only ever added, with defaults; `tests/fixtures/records_v0_1_0.json` guards this). An older release refuses rows written by a newer one with `RecordVersionError` and stops that role without changing the row or failing its job, because writes save whole records and would otherwise erase fields it does not know. Upgrade every role before new rows are written, and do not roll back past a release that has written newer records.

Schema changes ship as new Alembic revisions; released revisions never change. Run `rag-ingest migrate` before starting upgraded roles. Revision `0002` adds build admission, an outbox job index and retention timestamps, and marks existing builds as admitted so in-flight work continues unchanged. Its index builds briefly block writes to those tables, so on a large existing outbox run it during low traffic. Custom metadata stores must also implement `admit_generations` and `prune_history`, and `claim_job` must refuse jobs of builds that have not been admitted. Revision `0003` adds the `waiting` job status and recognition job, rate-limit and usage tables. Revision `0004` adds each scheduled batch's plan. Revision `0005` adds the duplicate-lookup indexes and `0006` the abandoned-upload marker; their indexes build `CONCURRENTLY`, so writes continue during the upgrade. Revision `0008` adds the keyword columns of `search_chunks` and their GIN index (and the `btree_gin` extension). Revision `0007` adds `search_chunks` (one copy of each chunk's text), document `tags`, `metadata` and `created_at` columns with their indexes, and makes batch plans required. Builds indexed before `0007` have no chunk rows and cannot publish or be searched: reprocess them. Custom metadata stores must also implement `get_batch_plan`, `abandoned_uploads`, `mark_upload_files_removed`, `put_search_chunks`, `search_chunk_digests`, `delete_search_chunks`, `filter_documents`, `hydrate_search_hits` and `update_document_labels`, and refuse duplicates in `initiate_upload` and `accept_verified_upload` as the PostgreSQL store does.

Use database backups and a restore test before operational deployment. Review indexes/query plans and lock/pool contention under your actual tenant/workload distribution. Version one serializes metadata mutations per tenant with transaction advisory locks, prioritizing correctness; it is not a benchmarked maximum-throughput design.

## 13. Validation and limits

Install test dependencies and run the automated suite:

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Without a database the suite runs the socket-free tests: retry decisions, real boto3 calls intercepted by in-process Moto, parser subprocess behavior, the embedding segment format, API tenant isolation, logging, settings and shared limits, and the custom adapter's result/error handling. Moto verifies our request/response handling; it is not a substitute for live AWS semantics, IAM, latency, throttling or outages.

For a real PostgreSQL/pgvector integration run, create a **dedicated disposable database whose name ends in `_test`**. The integration suite resets its metadata/vector tables and must never point at a database you need:

```bash
docker compose exec postgres createdb -U ingest ingestion_test
export INGEST_TEST_METADATA_DSN='postgresql://ingest:ingest_local_password@localhost:5432/ingestion_test'
python -m unittest discover -s tests -p 'test_postgres_integration.py' -v
```

The ParadeDB example's tests (`tests/test_paradedb.py`) need their own `*_test` database on a PostgreSQL with `pg_search` and pgvector, for example a ParadeDB container: `export INGEST_TEST_PARADEDB_DSN=postgresql://...:5434/ingestion_test`. They are skipped without it.

With that DSN set, run the whole suite (`python -m unittest discover -s tests`) to include the PostgreSQL tests: migrations to the latest revision match `schema.py`; the full pipeline, crash windows and recovery; recognition with a fake tool; duplicate refusal (including two identical uploads at once); abandoned-upload cleanup; publication digests; batched vector writes; batch plans; and the demo chat endpoint. They use in-process AWS emulation and deterministic embeddings, so no paid request is made. The PostgreSQL tests are skipped unless the dedicated test DSN is supplied.

### What was and was not verified here

Verified in this project:

- The full automated suite against PostgreSQL 16 with pgvector 0.8.7: all tests pass; 2 are skipped because a job-polling check does not apply to synchronous recognizers.
- A live run against real AWS S3, SQS and Textract with OpenAI `text-embedding-3-small`: 50 PDFs (text, scanned, mixed, encrypted, corrupt, blank, duplicate) uploaded through the API, every check in `live_test/verify.py` passing.
- The demo page in Chrome uploading straight to S3 (CORS), with duplicate refusal, Textract pages and grounded chat answers with sources.
- Per-stage latency on real AWS, measured before and after the optimizations (call and byte counts per stage).

Not verified:

- Running inside AWS (all live runs were from a laptop, so absolute timings include ~400 ms per S3/SQS call), and ingestion load beyond tens of documents (the search benchmark below loads rows directly; it does not exercise upload, extraction or embedding).
- Search with real embeddings at scale: the 300k benchmark uses synthetic clustered vectors and pseudo-word text.
- S3-, SQS- or PostgreSQL-compatible services other than those listed as tested in section 8, LocalStack included.
- Multipart uploads from a browser (files over the multipart threshold were tested through the API only).
- Failure injection against real AWS beyond the live test's crash and retry cases.

### Search at scale: measured

`bench/search_scale.py` loads a disposable database and measures the package's own code paths: `PgVectorStore.search_hits`, `PostgresKeywordIndex.search`, and the whole `runtime.search.run` pipeline (retrieval, fusion, liveness and filter checks, collapsing). Only the query-embedding call is replaced.

```bash
python bench/search_scale.py load    --dsn postgresql://USER:PASS@HOST/ingestion_bench
python bench/search_scale.py measure --dsn postgresql://USER:PASS@HOST/ingestion_bench
```

**Data.** 300,000 chunks of 1,536 dimensions in 6,000 documents. Tenants are skewed: one with 150,000 chunks, five with 20,000, fifty with 1,000. Vectors are clustered in a hierarchy (a shared direction, 200 topics, one centroid per document, noise per chunk), so a document's chunks are close and all tenants share the same regions of space, the hard case for a filtered index. Text is ~120 pseudo-words per chunk with a Zipf-like vocabulary. Each query is a stored chunk's vector plus noise, with four of that chunk's words. Uniform random vectors were not used: they behave nothing like embeddings.

**Machine.** MacBook (10 cores, 16 GB) running Docker (8 GB); PostgreSQL 16.16 and pgvector 0.8.7 with `shared_buffers=2GB` and `maintenance_work_mem=2GB`. HNSW `m=16`, `ef_construction=64`. Latencies are warm-cache and include the Python client.

| Load step | Result |
| --- | --- |
| HNSW build over 300k existing rows (`rag-ingest migrate` path) | 59–95 s (3,200–5,100 vectors/s) |
| Keyword vectors + GIN index | 16 s + 9 s |
| Inserts with the index live, 512 per transaction | 195 rows/s on one connection; 840–940 rows/s with 8 |
| Sizes | table 4.9 GB, HNSW 2.3 GB, chunk text 1.1 GB, keyword index 96 MB |

Vector search, recall@10 against exact search, by tenant size:

| Tenant | Exact search p50 | `ef_search=40` | `100` | `200` |
| --- | --- | --- | --- | --- |
| 150k chunks | 765 ms | 0.85, p50 28 ms | 0.89–0.92, p50 40 ms | 0.96–0.99, p50 36 ms, p95 102 ms |
| 20k chunks | 88 ms | 0.91 | 0.91–0.95 | 0.92–1.00, p95 53 ms |
| 1k chunks | 18 ms | 1.00, p50 7 ms | 1.00 | 1.00 |

Misses are all-or-nothing: a query either finds its neighbours or its graph walk ends in an unrelated cluster (scores ~0.15 where the true neighbours score ~0.85). With 200 fresh queries on the large tenant, 17 missed entirely at `ef_search=100` and 1–4 at 200; it was the same without the tenant filter, and rebuilding with `ef_construction=200` or `m=32` did not remove it. That is why the default is now 200 and `search-eval` reports misses. Real embeddings may be less sharply clustered than this data, so measure your own with `search-eval`.

Whole pipeline (`limit=10`), and whether the query's source chunk came back in the top 10:

| Tenant | vector | keyword | hybrid |
| --- | --- | --- | --- |
| 150k chunks | p50 67 / p95 129 ms, 95% | p50 270 / p95 1,153 ms, 43% | p50 206 / p95 1,144 ms, 96% |
| 20k chunks | p50 58 / p95 107 ms, 92% | p50 81 / p95 160 ms, 54% | p50 84 / p95 168 ms, 96% |
| 1k chunks | p50 12 / p95 45 ms, 100% | p50 14 / p95 24 ms, 80% | p50 13 / p95 17 ms, 100% |

Filters on the 150k tenant (hybrid): a tag on 10 documents (exact path) p50 35 ms; a tag on 277 documents p50 85 / p95 265 ms; a metadata filter matching every document p50 259 ms (it is as slow as unfiltered hybrid because of keyword search). 16 concurrent clients over mixed tenants: 30 hybrid queries/s, p50 252 / p95 1,311 ms, again dominated by keyword search on the large tenant.

What this means:

- Vector search scales well across tenant sizes: the iterative scan keeps small tenants complete and fast beside a large one, with no scan-limit fallbacks at this size.
- The `postgres` keyword index is the bottleneck at large tenants. It ranks with `ts_rank_cd` over every row matching any query word. The ParadeDB BM25 example removes that bottleneck; see below.
- Not measured: millions of vectors (at 1,536 dimensions, ~8 GB of index per million chunks, so plan memory first), real embeddings, cold caches, and a reranker's added latency (one network call, bounded by `INGEST_RERANK_TIMEOUT_MS`).
- Found and fixed by this benchmark: `migrate` could not build the vector index over existing rows, because the pool's 15 s statement timeout cancelled `CREATE INDEX CONCURRENTLY` on every attempt. The build now runs without that limit on a connection that is discarded afterwards.


**Keyword engines compared** (same 300k data and queries, one ParadeDB 0.26.3 container with pgvector 0.8.6; run with `measure --keyword postgres|paradedb`):

| Keyword search, k=40 | `postgres` full-text: p50 / p95 | found in top 10 | `paradedb` BM25: p50 / p95 | found in top 10 |
| --- | --- | --- | --- | --- |
| 150k-chunk tenant | 280 / 1,193 ms | 44% | **9.5 / 16.5 ms** | 100% |
| 20k-chunk tenant | 84 / 225 ms | 51% | 5.7 / 7.3 ms | 100% |
| 1k-chunk tenant | 7.6 / 24 ms | 81% | 4.5 / 5.6 ms | 100% |

| | `postgres` | `paradedb` |
| --- | --- | --- |
| Index build over 300k rows | tsvectors 17 s + GIN 14 s | 6 s |
| Index size | 200 MB GIN + a tsvector column per row | 86 MB |
| Narrow tag filter on the large tenant (hybrid) | p50 37 ms | p50 11 ms |
| 16 clients, hybrid, mixed tenants | 14 queries/s, p95 2,048 ms | 26 queries/s, p95 920 ms |

BM25 is about 70 times faster at p95 on the large tenant, because the tenant and build filters run inside the index and a Top K scan reads only the best rows. It also ranks better, because it weighs a word by how rare it is. The 100% is flattered by how the benchmark builds queries (two of the chunk's rarest words, exactly what BM25 rewards), so treat it as "BM25 finds distinctive terms", not as the accuracy on real questions. With BM25, hybrid search on a large tenant is limited by vector search, not by keywords. In this container, vector search itself was slower than in the first run (large tenant p50 ~400 ms against 67 ms): the HNSW index missed the cache 14% of the time because table and index (7.2 GB) did not fit beside ParadeDB in the 8 GB Docker VM. That is a direct measurement of why the vector index must fit in memory.

### Deliberate limits and remaining work

- Supported built-in inputs are PDFs and UTF-8 `.txt`. Scanned and image pages need an injected `Recognizer` (no tool is bundled; a Textract example is in `live_test/`). Recognized text is kept as plain text per page: tables, layout and form fields are not preserved as structure. Office formats, image uploads, archives and antivirus scanning require explicit extensions.
- The parser subprocess has time/output bounds and a 2 GiB address-space limit where Unix supports it. This is not a security sandbox. Use container resource controls and isolation for untrusted documents.
- File upload support up to the configured 10 GiB ceiling does not imply every such file can finish processing under default limits. No large-file or 10,000-document load benchmark was executed here.
- Duplicate refusal compares exact bytes within a tenant. Files that differ by even one byte, or that share only some pages, are separate documents, so their identical passages can both appear in search results. Text-normalization equivalence and per-tenant billing quotas are not implemented.
- Successful same-generation embeddings are cached; unknown model-call outcomes may still have been billed. There is no universal exactly-once billing guarantee.
- Files of uploads that ended failed or expired (refused duplicates, failed verification, abandoned transfers), including unfinished multipart parts, are erased by the scheduler after `INGEST_ABANDONED_UPLOAD_GRACE_SECONDS`. Two cases remain: a multipart upload started at the provider just before a crash, before its plan was saved (add an S3 lifecycle rule `AbortIncompleteMultipartUpload`), and outputs of superseded builds that never became active. Do not configure broad source-key expiry that deletes accepted pinned versions.
- Cleanup covers authorized deleted documents and retired active generations. Restore/backup retention, legal holds and replicated copies remain deployment concerns.
- The bundled static token policy is tenant-wide. Finer per-document ACLs need an application policy and an authorized search implementation; a custom authenticator alone does not add row-level ACL semantics.
- The bundled vector store uses one table dimension. Its HNSW index was benchmarked at 300,000 1,536-dimension vectors ("Search at scale" above), not at millions. Search has no paging. Passages that overlap without being identical are not collapsed. Multi-model targets need extensions.
- The package has no frontend (`live_test/web/` is a local demo page), no API rate limiting or per-tenant request quotas (only recognizer calls share a PostgreSQL token bucket), no centralized metrics backend, no bundled recognition tool, no migration downgrade and no cloud infrastructure deployment.
- Container image tags in Compose are convenience defaults; LocalStack uses `latest` because compatibility has not been validated here. Pin tested image digests and dependency versions before deployment.

Use this implementation as a concrete starting point, run the local integration flow, and keep its invariants while extending adapters and operational controls.

## Provider reference material

- [AWS S3 conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)
- [AWS CompleteMultipartUpload](https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html)
- [OpenAI embeddings API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create)

The built-in embedding adapter uses a conservative 8,191-token per-input ceiling and a 200,000-token batch ceiling, below the published API maxima. A different provider/model may impose lower limits and needs its own adapter policy.
