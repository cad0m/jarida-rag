import csv
import time

import mlflow
import pandas as pd
import ragas
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

from . import config
from .rag import answer, source
from .store import search

# shortcut: keyword match to spot "the documents don't say"; check answers.json if refusal_accuracy looks wrong
REFUSAL_WORDS = ["does not contain", "do not contain", "no information", "not mentioned",
                 "لا يحتوي", "لا تحتوي", "لا يتضمن", "لا تتضمن", "لا توجد", "غير متوفر"]


def read_questions():
    with open(config.QUESTIONS, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def contains(text, phrase):
    # PDF line breaks fall anywhere, so compare with single spaces
    return " ".join(phrase.split()) in " ".join(text.split())


def log_run(run_name, reranker_used, metrics, tables):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "embed_model": config.EMBED_MODEL,
            "reranker": config.RERANKER_MODEL if reranker_used else "none",
            "chunk_size": config.CHUNK_SIZE,
            "top_k": config.TOP_K,
            "max_chunks_per_text": config.MAX_CHUNKS_PER_TEXT,
        })
        mlflow.log_metrics(metrics)
        for name, table in tables.items():
            mlflow.log_table(table, name)


def retrieval_eval(client, embedder, run_name, reranker=None):
    """For every answerable question:
    rank     = position of the first chunk from the right legal text
    evidence = whether the chunk holding the answer itself was retrieved"""
    rows = []
    for question in read_questions():
        if not question["expected"]:
            continue

        start = time.time()
        hits = search(client, embedder, question["question"], reranker)
        seconds = time.time() - start

        right_text = [hit["issue"] == question["issue"] and question["expected"] in hit["title"] for hit in hits]
        rows.append({
            "question": question["question"],
            "expected": question["expected"],
            "retrieved": " | ".join(hit["text_number"] or hit["title"][:30] for hit in hits),
            "rank": right_text.index(True) + 1 if any(right_text) else None,
            "evidence_found": any(contains(hit["text"], question["evidence"]) for hit in hits) if question["evidence"] else None,
            "seconds": seconds,
        })

    table = pd.DataFrame(rows)
    ranks = table["rank"].astype(float)  # None (not found) becomes NaN
    metrics = {
        "hit_rate": ranks.notna().mean(),
        "mrr": (1 / ranks).fillna(0).mean(),
        "evidence_recall": table["evidence_found"].dropna().astype(bool).mean(),
        "avg_search_seconds": table["seconds"].mean(),
    }
    log_run(run_name, bool(reranker), metrics, {"retrieval.json": table})
    return table


def answer_questions(client, embedder, tokenizer, model, reranker=None):
    """Answers the questions that have a reference answer, plus the ones the gazette can't answer."""
    answers = []
    for question in read_questions():
        should_refuse = not question["expected"]
        if not question["reference"] and not should_refuse:
            continue

        start = time.time()
        reply, hits = answer(question["question"], client, embedder, tokenizer, model, reranker)
        answers.append({
            "question": question["question"],
            "reference": question["reference"],
            "answer": reply,
            # exactly what the LLM saw, citation labels included, so the judge can check the citations too
            "contexts": [f"{source(hit)} {hit['text']}" for hit in hits],
            "should_refuse": should_refuse,
            "refused": any(word in reply.lower() for word in REFUSAL_WORDS),
            "seconds": time.time() - start,
        })
    return answers


def ragas_eval(answers, judge, judge_embeddings, run_name, reranker_used):
    # A correct "not found" has no claims for RAGAS to check, so refusals are measured apart
    golden = [item for item in answers if not item["should_refuse"]]
    dataset = ragas.EvaluationDataset.from_list([
        {
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item["contexts"],
            "reference": item["reference"],
        }
        for item in golden
    ])
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]

    result = ragas.evaluate(
        dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        run_config=RunConfig(max_workers=1, timeout=900),  
        raise_exceptions=False, 
    )
    scores = result.to_pandas()
    names = [metric.name for metric in metrics]

    answers_table = pd.DataFrame(answers).drop(columns="contexts")
    summary = {name: scores[name].mean() for name in names}
    summary["nan_rate"] = scores[names].isna().mean().mean()
    summary["refusal_accuracy"] = (answers_table["refused"] == answers_table["should_refuse"]).mean()
    summary["avg_answer_seconds"] = answers_table["seconds"].mean()

    log_run(run_name, reranker_used, summary, {"ragas.json": scores, "answers.json": answers_table})
    return scores
