"""Fixed limits shared by more than one part of the package.

These are not settings. Each is fixed by a storage or provider rule, or by a
file format that one component writes and another reads back: if the two
sides disagreed, saved work would become unreadable. Change one only with a
reviewed migration plan. Deployment-specific knobs live in ServiceSettings.
"""

MiB = 1024**2

# ---- formats we write and read back (writer and reader must agree) ---------
# Any stage artifact (chunk batch, segment, page analysis, extracted text).
MAX_ARTIFACT_BYTES = 128 * MiB
# A stage manifest or checkpointed batch manifest.
MAX_MANIFEST_BYTES = 1 * MiB
# A sealed generation inventory.
MAX_INVENTORY_BYTES = 2 * MiB
# Batches one inventory may list, and chunks one generation may have.
MAX_INVENTORY_BATCHES = 4096
MAX_GENERATION_CHUNKS = 1_000_000
# Jobs one build may have: one embed job per batch, plus extract, chunk,
# index and publish.
MAX_GENERATION_JOBS = MAX_INVENTORY_BATCHES + 4
# The extraction page map (page offsets of the extracted text).
MAX_PAGE_MAP_BYTES = 8 * MiB
# The index stage's receipt (one digest per batch).
MAX_INDEX_RECEIPT_BYTES = 16 * MiB

# ---- batch shape (settings check, embedding call and vector adapter) -------
# Chunks per embedding / vector batch; INGEST_EMBEDDING_BATCH_SIZE is capped here.
MAX_BATCH_ITEMS = 512
# Characters one chunk may have when chunked and when embedded.
MAX_CHUNK_CHARACTERS = 100_000
# Characters one extracted page may have.
MAX_PAGE_CHARACTERS = 2_000_000
# Vector IDs per publication digest query (engine and vector adapter).
MAX_DIGEST_IDS = 8192

# ---- S3 multipart rules ----------------------------------------------------
MULTIPART_MIN_PART_BYTES = 5 * MiB
MULTIPART_MAX_PART_BYTES = 5 * 1024**3
MULTIPART_MAX_PARTS = 10_000

# ---- intake API contract -----------------------------------------------------
MAX_FILENAME_CHARS = 255
MAX_REQUEST_KEY_CHARS = 200
# Bytes read from the start of a file to check its type (e.g. "%PDF-").
FILE_TYPE_PREFIX_BYTES = 4096

# ---- internal I/O ------------------------------------------------------------
# Block size for streaming stored objects.
READ_BLOCK_BYTES = 1 * MiB

# ---- document labels (tags and metadata, used by search filters) -------------
MAX_TAGS = 32
MAX_TAG_CHARS = 64
MAX_METADATA_KEYS = 32
MAX_METADATA_KEY_CHARS = 64
MAX_METADATA_VALUE_CHARS = 256
