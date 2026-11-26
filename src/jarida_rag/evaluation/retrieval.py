import csv
import time

import mlflow
import pandas as pd

from .. import config
from ..rag import source
from .scoring import contains

OUT_OF_SCOPE = "OUT_OF_SCOPE"  # `expected` of questions the gazette can't answer


def read_questions():
    with open(config.QUESTIONS, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def log_run(run_name, stage, retriever, metrics, tables):
    """`retriever` names the search used: dense or hybrid, with or without the reranker ("hybrid+rerank")."""
    with mlflow.start_run(run_name=run_name, tags={"stage": stage}):
        mlflow.log_params({
            "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "embed_model": config.EMBED_MODEL,
            "retriever": retriever,
            "reranker_model": config.RERANKER_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "top_k": config.TOP_K,
            "max_chunks_per_text": config.MAX_CHUNKS_PER_TEXT,
            "chunk_header": config.CHUNK_HEADER,
            "unwrap_lines": config.UNWRAP_LINES,
            "split_preamble": config.SPLIT_PREAMBLE,
        })
        mlflow.log_metrics(metrics)
        for name, table in tables.items():
            mlflow.log_table(table, name)


def evaluate(search, run_name, retriever):
    """`search(question)` returns the top chunks (from Postgres or from memory). For every answerable question:
    doc_hit       = a chunk of the right legal text is in the top K
    evidence_rank = position of the first chunk holding the evidence quote (0 = not retrieved)"""
    rows = []
    for question in read_questions():
        if question["expected"] == OUT_OF_SCOPE:
            continue
        assert question["evidence"], f"no evidence quote (every chunk would contain it): {question['question']}"

        start = time.time()
        hits = search(question["question"])
        seconds = time.time() - start

        # checked against what the LLM reads: the source label and the chunk text
        has_evidence = [contains(f"{source(hit)} {hit['text']}", question["evidence"]) for hit in hits]
        evidence_rank = has_evidence.index(True) + 1 if any(has_evidence) else 0

        references = [r for r in question["reference"].split("|") if r]  # multi-text questions: any of them counts
        if references:
            is_right_text = [hit["issue"] == question["issue"] and any(r in hit["title"] for r in references) for hit in hits]
        else:
            is_right_text = has_evidence  # texts without a decree number (the CESE opinion): judged by the quote

        rows.append({
            "question": question["question"],
            "reference": question["reference"],
            "retrieved": " | ".join(hit["text_number"] or hit["title"][:30] for hit in hits),
            "doc_hit": any(is_right_text),
            "evidence_rank": evidence_rank,
            "seconds": seconds,
        })

    table = pd.DataFrame(rows)
    metrics = {
        "doc_hit_rate": table["doc_hit"].mean(),
        "evidence_hit_rate": (table["evidence_rank"] > 0).mean(),
        # rewards ranking the chunk that holds the answer first, not just any chunk of the right text
        "evidence_mrr": (1 / table["evidence_rank"].replace(0, float("inf"))).mean(),
        "avg_search_seconds": table["seconds"].mean(),
    }
    log_run(run_name, "retrieval", retriever, metrics, {"retrieval.json": table})
    return table, metrics
