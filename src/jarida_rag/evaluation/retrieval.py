import csv
import time

import mlflow
import pandas as pd

from .. import config
from ..rag import source
from ..store import best_hits, cap_per_text
from .scoring import contains, miss_stage

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


def evidence_rank(hits, evidence):
    """1-based position of the first chunk holding the quote, checked on what the LLM reads (label + text); 0 = absent."""
    found = [contains(f"{source(hit)} {hit['text']}", evidence) for hit in hits]
    return found.index(True) + 1 if any(found) else 0


def evaluate(candidates, reranker, run_name, retriever):
    """`candidates(question)` returns the pool before the cap (store.candidates or store.candidates_in_memory).
    For every answerable question:
    doc_hit       = a chunk of the right legal text is in the top K
    evidence_rank = position of the first chunk holding the evidence quote (0 = not retrieved)
    miss_stage    = where that chunk was lost: not in candidates, cut by the cap, outside the rerank pool, or ranked low"""
    rows = []
    for question in read_questions():
        if question["expected"] == OUT_OF_SCOPE:
            continue
        assert question["evidence"], f"no evidence quote (every chunk would contain it): {question['question']}"

        start = time.time()
        pool = candidates(question["question"])
        hits = best_hits(question["question"], pool, reranker)
        seconds = time.time() - start

        ranks = [evidence_rank(stage, question["evidence"]) for stage in (pool, cap_per_text(pool), hits)]
        references = [r for r in question["reference"].split("|") if r]  # multi-text questions: any of them counts
        if references:
            doc_hit = any(hit["issue"] == question["issue"] and hit["text_number"] in references for hit in hits)
        else:
            doc_hit = ranks[-1] > 0  # texts without a decree number (the CESE opinion): judged by the quote

        rows.append({
            "question": question["question"],
            "reference": question["reference"],
            "retrieved": " | ".join(hit["text_number"] or hit["title"][:30] for hit in hits),
            "doc_hit": doc_hit,
            "candidate_rank": ranks[0],
            "capped_rank": ranks[1],
            "evidence_rank": ranks[2],
            "miss_stage": miss_stage(*ranks, reranked=reranker is not None),
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
