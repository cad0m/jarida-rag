"""The HTTP API the frontend calls, built around models that are already loaded (notebooks/serve.ipynb).

GET /health -> {"status", "documents"}; POST /ask {"question"} -> {"answer", "sources"}
"""
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import rag

SOURCE_FIELDS = ["issue", "date", "title", "text_number", "article", "page_start", "text"]


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(pool, embedder, reranker, tokenizer, model, lexical_index=None):
    app = FastAPI(title="Jarida")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    one_at_a_time = threading.Lock()  # a single LLM on the GPU: concurrent questions would run it out of memory

    @app.get("/health")
    def health():
        with pool.connection() as db:
            indexed = db.execute("SELECT count(*) AS n FROM documents WHERE indexed_chunks > 0").fetchone()["n"]
        return {"status": "ok", "documents": indexed}

    @app.post("/ask")
    def ask(body: Question):
        with one_at_a_time:
            reply, hits = rag.answer(body.question.strip(), pool, embedder, tokenizer, model, reranker, lexical_index)
        return {"answer": reply, "sources": sources(pool, hits)}

    return app


def sources(pool, hits):
    """The hits as the frontend shows them, each with the Drive id that opens its PDF."""
    with pool.connection() as db:
        rows = db.execute(
            "SELECT sha256, drive_id FROM documents WHERE sha256 = ANY(%s)", [[hit["doc_id"] for hit in hits]]
        ).fetchall()
    drive_ids = {row["sha256"]: row["drive_id"] for row in rows}
    return [{**{field: hit[field] for field in SOURCE_FIELDS}, "drive_id": drive_ids.get(hit["doc_id"])} for hit in hits]
