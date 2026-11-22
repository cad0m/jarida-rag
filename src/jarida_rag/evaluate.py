import csv

import mlflow
import pandas as pd

from . import config
from .store import search


def evaluate(client, embedder, run_name):
    with open(config.QUESTIONS, encoding="utf-8", newline="") as file:
        questions = [row for row in csv.DictReader(file) if row["text_number"]]

    results = []
    for row in questions:
        hits = search(client, embedder, row["question"])
        found = [hit["issue"] == row["issue"] and hit["text_number"] == row["text_number"] for hit in hits]
        results.append({
            "question": row["question"],
            "expected": row["text_number"],
            "retrieved": ", ".join(hit["text_number"] for hit in hits),
            "rank": found.index(True) + 1 if True in found else None,
        })
    ranks = [result["rank"] for result in results]
    table = pd.DataFrame(results)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "embed_model": config.EMBED_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
        })
        mlflow.log_metrics({
            "hit_rate": sum(rank is not None for rank in ranks) / len(ranks),
            "mrr": sum(1 / rank for rank in ranks if rank) / len(ranks),
            "num_chunks": client.count(config.COLLECTION).count,
            "num_questions": len(ranks),
        })
        mlflow.log_table(table, "results.json")
    return table
