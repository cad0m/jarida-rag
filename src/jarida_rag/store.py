import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .chunk import chunk_all

COLUMNS = ["id", "text", "embed_text", "doc_id", "issue", "date", "lang", "title", "text_number", "article", "page_start", "page_end"]


def load_embedder():
    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    embedder.max_seq_length = 1024
    if config.EMBED_DEVICE.startswith("cuda"):
        embedder.half()
    return embedder


def load_reranker():
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


def build_index(pool, embedder):
    chunks = chunk_all()
    vectors = embed(embedder, [chunk["embed_text"] for chunk in chunks])
    rows = [[chunk[column] for column in COLUMNS] + [vector] for chunk, vector in zip(chunks, vectors)]
    placeholders = ", ".join(["%s"] * (len(COLUMNS) + 1))

    with pool.connection() as db:
        db.execute(f"DROP TABLE IF EXISTS {config.TABLE}")
        db.execute(f"""
            CREATE TABLE {config.TABLE} (
                id uuid PRIMARY KEY,
                text text, embed_text text, doc_id text, issue text, date date, lang text,
                title text, text_number text, article text, page_start int, page_end int,
                embedding vector({len(vectors[0])})
            )""")
        db.cursor().executemany(f"INSERT INTO {config.TABLE} ({', '.join(COLUMNS)}, embedding) VALUES ({placeholders})", rows)
        db.execute(f"CREATE INDEX ON {config.TABLE} USING hnsw (embedding vector_cosine_ops)")
    return len(chunks)


def search(pool, embedder, question, reranker=None):
    vector = embed(embedder, [question])[0]
    # <=> is cosine distance; shortcut: a fixed pool of 100 candidates, filter by metadata in SQL when the corpus grows
    with pool.connection() as db:
        hits = db.execute(
            f"SELECT {', '.join(COLUMNS)} FROM {config.TABLE} ORDER BY embedding <=> %s LIMIT 100",
            [vector],
        ).fetchall()
    return best_hits(question, hits, reranker)


def search_in_memory(chunks, vectors, embedder, question, reranker=None):
    """Same search without a database, to compare chunking variants quickly (notebooks/chunking_lab.ipynb)."""
    vector = embed(embedder, [question])[0]
    closest = (vectors @ vector).argsort()[::-1][:100]  # vectors are normalized: dot product = cosine
    return best_hits(question, [chunks[i] for i in closest], reranker)


def best_hits(question, hits, reranker=None):
    """From candidates ordered by similarity: a few chunks per legal text, optionally reranked, top K."""
    # keep at most a few chunks per legal text, so one long decree can't fill the whole context
    results = []
    per_text = {}
    for hit in hits:
        text_key = (hit["doc_id"], hit["title"])
        per_text[text_key] = per_text.get(text_key, 0) + 1
        if per_text[text_key] <= config.MAX_CHUNKS_PER_TEXT:
            results.append(hit)

    if reranker:
        candidates = results[:config.RERANK_CANDIDATES]
        scores = reranker.predict([(question, hit["embed_text"]) for hit in candidates])
        best_first = scores.argsort()[::-1]
        results = [candidates[i] for i in best_first]

    return results[:config.TOP_K]
