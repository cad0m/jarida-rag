import csv
import re
import time

import mlflow
import pandas as pd
import ragas
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

from . import config
from .rag import NOT_FOUND_AR, NOT_FOUND_EN, answer, source

OUT_OF_SCOPE = "OUT_OF_SCOPE"  # `expected` of questions the gazette can't answer
RAGAS_TYPES = ("attribute",)  # single-fact questions: about 15, which keeps the judge run near 80 minutes


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
            "chunk_header": config.CHUNK_HEADER,
            "unwrap_lines": config.UNWRAP_LINES,
            "split_preamble": config.SPLIT_PREAMBLE,
        })
        mlflow.log_metrics(metrics)
        for name, table in tables.items():
            mlflow.log_table(table, name)


def retrieval_eval(search, run_name, reranker_used):
    """`search(question)` returns the top chunks (from Postgres or from memory). For every answerable question:
    rank          = position of the first chunk from the right legal text
    evidence_rank = position of the first chunk holding the answer phrase itself (0 = not retrieved)"""
    rows = []
    for question in read_questions():
        if question["expected"] == OUT_OF_SCOPE:
            continue

        start = time.time()
        hits = search(question["question"])
        seconds = time.time() - start

        # checked against what the LLM reads: the source label and the chunk text
        has_answer = [contains(f"{source(hit)} {hit['text']}", question["evidence"]) for hit in hits]
        evidence_rank = has_answer.index(True) + 1 if any(has_answer) else 0

        references = [r for r in question["reference"].split("|") if r]  # multi-text questions: any of them counts
        if references:
            is_right_text = [hit["issue"] == question["issue"] and any(r in hit["title"] for r in references) for hit in hits]
        else:
            is_right_text = has_answer  # texts without a decree number (the CESE opinion): judged by the quote
        rank = is_right_text.index(True) + 1 if any(is_right_text) else None

        rows.append({
            "question": question["question"],
            "reference": question["reference"],
            "retrieved": " | ".join(hit["text_number"] or hit["title"][:30] for hit in hits),
            "rank": rank,
            "evidence_rank": evidence_rank,
            "seconds": seconds,
        })

    table = pd.DataFrame(rows)
    ranks = table["rank"].astype(float)  # None (not found) becomes NaN
    evidence_ranks = table["evidence_rank"].dropna().astype(float)
    metrics = {
        "hit_rate": ranks.notna().mean(),
        "mrr": (1 / ranks).fillna(0).mean(),
        "evidence_recall": (evidence_ranks > 0).mean(),
        # finer than mrr: rewards ranking the chunk that holds the answer, not just any chunk of the right text
        "evidence_mrr": (1 / evidence_ranks.replace(0, float("inf"))).mean(),
        "avg_search_seconds": table["seconds"].mean(),
    }
    log_run(run_name, reranker_used, metrics, {"retrieval.json": table})
    return metrics


def has_fact(reply, fact):
    if re.fullmatch(r"[\d.,]+", fact):
        # a number counts only as a whole number: "2" must not match inside "2.26.28"
        return re.search(rf"(?<![\d.,]){re.escape(fact)}(?![\d]|[.,]\d)", reply) is not None
    return contains(reply, fact)


def has_key_facts(reply, key_fact):
    """Every "&" part of the key fact appears in the answer, any of its "|" alternatives counting."""
    reply = re.sub(r"\[[^\]]*\]", " ", reply)  # citation labels are full of numbers
    return all(any(has_fact(reply, alternative.strip()) for alternative in part.split("|")) for part in key_fact.split("&"))


def answer_questions(pool, embedder, tokenizer, model, reranker=None):
    """Answers the RAGAS questions (one fact to extract) and the ones the gazette can't answer."""
    answers = []
    for question in read_questions():
        should_refuse = question["expected"] == OUT_OF_SCOPE
        if question["type"] not in RAGAS_TYPES and not should_refuse:
            continue

        start = time.time()
        reply, hits = answer(question["question"], pool, embedder, tokenizer, model, reranker)
        answers.append({
            "question": question["question"],
            "expected": question["expected"],
            "answer": reply,
            # exactly what the LLM saw, citation labels included, so the judge can check the citations too
            "contexts": [f"{source(hit)} {hit['text']}" for hit in hits],
            "should_refuse": should_refuse,
            "refused": NOT_FOUND_EN in reply or NOT_FOUND_AR in reply,
            # deterministic correctness: RAGAS faithfulness can't tell a misread number from the right one
            "facts_ok": None if should_refuse else has_key_facts(reply, question["key_fact"]),
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
            "reference": item["expected"],
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
    summary["fact_accuracy"] = answers_table["facts_ok"].dropna().astype(bool).mean()
    summary["avg_answer_seconds"] = answers_table["seconds"].mean()

    log_run(run_name, reranker_used, summary, {"ragas.json": scores, "answers.json": answers_table})
    return scores
