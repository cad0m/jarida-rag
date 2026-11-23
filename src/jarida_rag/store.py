from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config
from .chunk import chunk_all


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


def search(client, embedder, question, reranker=None):
    vector = embed(embedder, [question])[0]
    # shortcut: a fixed pool of 100 candidates; switch to Qdrant's query_points_groups when the corpus grows
    hits = client.query_points(config.COLLECTION, query=vector.tolist(), limit=100).points

    # keep at most a few chunks per legal text, so one long decree can't fill the whole context
    results = []
    per_text = {}
    for hit in hits:
        text_key = (hit.payload["doc_id"], hit.payload["title"])
        per_text[text_key] = per_text.get(text_key, 0) + 1
        if per_text[text_key] <= config.MAX_CHUNKS_PER_TEXT:
            results.append(hit.payload)

    if reranker:
        candidates = results[:config.RERANK_CANDIDATES]
        scores = reranker.predict([(question, hit["text"]) for hit in candidates])
        best_first = scores.argsort()[::-1]
        results = [candidates[i] for i in best_first]

    return results[:config.TOP_K]
