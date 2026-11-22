from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from . import config
from .chunk import chunk_all


def load_embedder():
    embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    embedder.max_seq_length = 1024
    if config.EMBED_DEVICE.startswith("cuda"):
        embedder.half()
    return embedder


def embed(embedder, texts):
    return embedder.encode(texts, batch_size=config.EMBED_BATCH, normalize_embeddings=True)


def open_store():
    return QdrantClient(path=str(config.QDRANT_DIR))


def build_index(client, embedder):
    chunks = chunk_all()
    vectors = embed(embedder, [chunk["text"] for chunk in chunks])

    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        config.COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    client.upsert(
        config.COLLECTION,
        points=[
            models.PointStruct(id=chunk["id"], vector=vector.tolist(), payload=chunk)
            for chunk, vector in zip(chunks, vectors)
        ],
    )
    return len(chunks)


def search(client, embedder, question, limit=config.TOP_K):
    vector = embed(embedder, [question])[0]
    # shortcut: a fixed candidate pool; switch to Qdrant's query_points_groups when the corpus grows
    hits = client.query_points(config.COLLECTION, query=vector.tolist(), limit=limit * 20).points

    # keep a few chunks per legal text, so one long decree can't fill the whole context
    results = []
    per_text = {}
    for hit in hits:
        key = (hit.payload["doc_id"], hit.payload["title"])
        per_text[key] = per_text.get(key, 0) + 1
        if per_text[key] <= config.MAX_CHUNKS_PER_TEXT:
            results.append(hit.payload)
    return results[:limit]
