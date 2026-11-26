import time

import pandas as pd
import ragas
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.run_config import RunConfig

from ..rag import NOT_FOUND_AR, NOT_FOUND_EN, answer, source
from .retrieval import OUT_OF_SCOPE, log_run, read_questions
from .scoring import has_key_facts

RAGAS_TYPES = ("attribute",)  # single-fact questions: about 15, which keeps the judge run near 40 minutes


def answer_questions(pool, embedder, tokenizer, model, reranker=None, lexical_index=None):
    """Answers the single-fact questions and the ones the gazette can't answer."""
    answers = []
    for question in read_questions():
        should_refuse = question["expected"] == OUT_OF_SCOPE
        if question["type"] not in RAGAS_TYPES and not should_refuse:
            continue

        start = time.time()
        reply, hits = answer(question["question"], pool, embedder, tokenizer, model, reranker, lexical_index)
        answers.append({
            "question": question["question"],
            "expected": question["expected"],
            "answer": reply,
            # exactly what the LLM saw, citation labels included, so the judge can check the citations too
            "contexts": [f"{source(hit)} {hit['text']}" for hit in hits],
            "should_refuse": should_refuse,
            # the fixed sentence and nothing else: a refusal followed by an answer is still a hallucination
            "refused": reply.strip() in (NOT_FOUND_EN, NOT_FOUND_AR),
            # deterministic correctness: RAGAS faithfulness can't tell a misread number from the right one
            "facts_ok": None if should_refuse else has_key_facts(reply, question["key_fact"]),
            "seconds": time.time() - start,
        })
    return answers


def evaluate(answers, run_name, retriever):
    """Fact and refusal accuracy: deterministic, so a prompt change is measured in minutes, without the judge."""
    table = pd.DataFrame(answers).drop(columns="contexts")
    metrics = {
        "fact_accuracy": table["facts_ok"].dropna().astype(bool).mean(),
        "refusal_accuracy": (table["refused"] == table["should_refuse"]).mean(),
        "avg_answer_seconds": table["seconds"].mean(),
    }
    log_run(run_name, "generation", retriever, metrics, {"answers.json": table})
    return table, metrics


def ragas_eval(answers, judge, judge_embeddings, run_name, retriever):
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
    # retrieval quality is measured exactly by evaluation/retrieval.py, so only the answer is judged here
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1)]

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

    summary = {name: scores[name].mean() for name in names}
    summary["nan_rate"] = scores[names].isna().mean().mean()
    log_run(run_name, "ragas", retriever, summary, {"ragas.json": scores})
    return scores
