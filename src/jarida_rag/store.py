import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config, lexical
from .chunk import chunk_all, chunk_document
from .pdf import read_pages
from .sources import drive

COLUMNS = [
    "id", "text", "embed_text", "doc_id", "issue", "date", "lang",
    "title", "text_number", "text_type", "amends", "article", "page_start", "page_end",
]
CANDIDATES = 100  # nearest chunks fetched before the per-text cap and the reranker


def load_embedder():
    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    embedder.max_seq_length = 1024
    if config.EMBED_DEVICE.startswith("cuda"):
        embedder.half()
    return embedder


def load_reranker():
    # full precision: fp16 was 4x faster but ranked the answer chunk lower (evidence_mrr -0.03 on every variant)
    return CrossEncoder(config.RERANKER_MODEL, device=config.EMBED_DEVICE, max_length=1024)


def embed(embedder, texts):
    return embedder.encode(texts, batch_size=config.EMBED_BATCH, normalize_embeddings=True)


def connect():
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set: point it to a Postgres database with the pgvector extension")
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as db:
        db.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # Neon's free database sleeps after ~5 idle minutes: the pool checks the connection and reopens it if needed
    return ConnectionPool(
        config.DATABASE_URL,
        kwargs={"row_factory": dict_row, "autocommit": True},
        configure=register_vector,
        check=ConnectionPool.check_connection,
        min_size=1,
        max_size=1,
        open=True,
    )


def create_chunks_table(db, dimensions):
    db.execute(f"""
        CREATE TABLE IF NOT EXISTS {config.TABLE} (
            id uuid PRIMARY KEY,
            text text, embed_text text, doc_id text, issue text, date date, lang text,
            title text, text_number text, text_type text, amends text, article text, page_start int, page_end int,
            embedding vector({dimensions})
        )""")
    db.execute(f"CREATE INDEX IF NOT EXISTS {config.TABLE}_embedding_idx ON {config.TABLE} USING hnsw (embedding vector_cosine_ops)")
    db.execute(f"CREATE INDEX IF NOT EXISTS {config.TABLE}_doc_id_idx ON {config.TABLE} (doc_id)")


def insert_chunks(db, chunks, vectors):
    rows = [[chunk[column] for column in COLUMNS] + [vector] for chunk, vector in zip(chunks, vectors)]
    placeholders = ", ".join(["%s"] * (len(COLUMNS) + 1))
    db.cursor().executemany(f"INSERT INTO {config.TABLE} ({', '.join(COLUMNS)}, embedding) VALUES ({placeholders})", rows)


def build_index(pool, embedder):
    """Rebuild the table from the PDFs of data/manifest.csv: the fixed corpus of the evaluation notebooks."""
    chunks = chunk_all()
    vectors = embed(embedder, [chunk["embed_text"] for chunk in chunks])
    # one transaction: if the connection drops mid-insert, the previous table stays instead of a partial one
    with pool.connection() as db, db.transaction():
        db.execute(f"DROP TABLE IF EXISTS {config.TABLE}")
        create_chunks_table(db, len(vectors[0]))
        insert_chunks(db, chunks, vectors)
    return len(chunks)


def update_index(pool, embedder):
    """Index the stored documents that are new or were indexed under another config.INDEX_VERSION.

    A document is downloaded from Drive only when it needs indexing. One whose chunks went missing
    (build_index dropped the table) counts as not indexed.
    """
    with pool.connection() as db:
        create_chunks_table(db, embedder.get_sentence_embedding_dimension())
        documents = db.execute(f"""
            SELECT * FROM documents
            WHERE status = 'stored' AND (
                indexed_version IS DISTINCT FROM %s
                OR indexed_chunks > (SELECT count(*) FROM {config.TABLE} WHERE doc_id = documents.sha256)
            )
            ORDER BY date DESC""", [config.INDEX_VERSION]).fetchall()
    indexed = 0
    for document in documents:
        # one PDF that can't be had or read (network, Drive not shared, broken file) must not stop the others;
        # OSError covers curl_cffi's network errors, RuntimeError the Drive check and pdfium's PdfiumError
        try:
            index_document(pool, embedder, document)
            indexed += 1
        except (RuntimeError, OSError) as error:
            print(f"skipped {document['path']}: {error}")
    return indexed


def index_document(pool, embedder, document):
    pages = read_pages(drive.download(document["drive_id"]), document["filename"])
    meta = {"doc_id": document["sha256"], "issue": document["issue"], "date": document["date"], "lang": "ar"}
    chunks = chunk_document(pages, meta)
    vectors = embed(embedder, [chunk["embed_text"] for chunk in chunks]) if chunks else []
    with pool.connection() as db, db.transaction():
        db.execute(f"DELETE FROM {config.TABLE} WHERE doc_id = %s", [document["sha256"]])
        insert_chunks(db, chunks, vectors)
        # an unreadable scan is recorded too (0 chunks), so it isn't downloaded again on every run
        db.execute(
            "UPDATE documents SET indexed_version = %s, indexed_chunks = %s WHERE url = %s",
            [config.INDEX_VERSION, len(chunks), document["url"]],
        )


def load_lexical(pool):
    # shortcut: BM25 in memory over every chunk; move to a database BM25 (e.g. ParadeDB pg_search) past ~100k chunks
    with pool.connection() as db:
        return lexical.build(db.execute(f"SELECT {', '.join(COLUMNS)} FROM {config.TABLE}").fetchall())


def search(pool, embedder, question, reranker=None, lexical_index=None):
    return best_hits(question, candidates(pool, embedder, question, lexical_index), reranker)


def candidates(pool, embedder, question, lexical_index=None):
    """The closest chunks, fused with BM25 if given: the pool the per-text cap and the reranker then work on."""
    vector = embed(embedder, [question])[0]
    # <=> is cosine distance; shortcut: a fixed pool of candidates, filter by metadata in SQL when the corpus grows
    with pool.connection() as db:
        # the HNSW index returns at most ef_search rows (default 40), whatever the LIMIT
        db.execute(f"SET hnsw.ef_search = {CANDIDATES}")
        hits = db.execute(
            f"SELECT {', '.join(COLUMNS)} FROM {config.TABLE} ORDER BY embedding <=> %s LIMIT {CANDIDATES}",
            [vector],
        ).fetchall()
    return lexical.fuse(hits, lexical.search(lexical_index, question, CANDIDATES)) if lexical_index else hits


def candidates_in_memory(chunks, vectors, embedder, question, lexical_index=None):
    """Same candidates without a database, to compare variants quickly (notebooks/retrieval.ipynb)."""
    vector = embed(embedder, [question])[0]
    closest = (vectors @ vector).argsort()[::-1][:CANDIDATES]  # vectors are normalized: dot product = cosine
    hits = [chunks[i] for i in closest]
    return lexical.fuse(hits, lexical.search(lexical_index, question, CANDIDATES)) if lexical_index else hits


def cap_per_text(hits):
    """At most MAX_CHUNKS_PER_TEXT chunks per legal text, so one long decree can't fill the reranker's pool."""
    results = []
    per_text = {}
    for hit in hits:
        text_key = (hit["doc_id"], hit["title"])
        per_text[text_key] = per_text.get(text_key, 0) + 1
        if per_text[text_key] <= config.MAX_CHUNKS_PER_TEXT:
            results.append(hit)
    return results


def best_hits(question, hits, reranker=None):
    """From the candidates: the per-text cap, the reranker on the first RERANK_CANDIDATES, then the top K."""
    results = cap_per_text(hits)
    if reranker:
        candidates = results[:config.RERANK_CANDIDATES]
        scores = reranker.predict([(question, hit["embed_text"]) for hit in candidates])
        best_first = scores.argsort()[::-1]
        results = [candidates[i] for i in best_first]

    return results[:config.TOP_K]
