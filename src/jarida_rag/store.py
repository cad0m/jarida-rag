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
    hits = client.query_points(config.COLLECTION, query=vector.tolist(), limit=limit).points
    return [hit.payload for hit in hits]
