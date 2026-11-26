import time

import pandas as pd
import ragas
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig

from ..rag import NOT_FOUND_AR, NOT_FOUND_EN, answer, source
from .retrieval import OUT_OF_SCOPE, log_run, read_questions
from .scoring import has_key_facts

RAGAS_TYPES = ("attribute",)  # single-fact questions: the judge only grades these 15, which keeps it near 25 minutes


def answer_questions(pool, embedder, tokenizer, model, reranker=None, lexical_index=None):
    """Answers every question of the golden set, the out-of-scope ones included."""
    answers = []
    for question in read_questions():
        should_refuse = question["expected"] == OUT_OF_SCOPE
        start = time.time()
        reply, hits = answer(question["question"], pool, embedder, tokenizer, model, reranker, lexical_index)
        answers.append({
            "question": question["question"],
            "type": question["type"],
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
    """Fact and refusal accuracy: deterministic, so a prompt change is measured without the judge."""
    table = pd.DataFrame(answers).drop(columns="contexts")
    answerable = table[~table["should_refuse"]]
    metrics = {
        "fact_accuracy": answerable["facts_ok"].astype(bool).mean(),
        "refusal_accuracy": table[table["should_refuse"]]["refused"].mean(),
        "false_refusal_rate": answerable["refused"].mean(),  # answerable questions the model declined
        "avg_answer_seconds": table["seconds"].mean(),
    }
    log_run(run_name, "generation", retriever, metrics, {"answers.json": table})
    return table, metrics


def ragas_eval(answers, judge, run_name, retriever):
    # A correct "not found" has no claims for RAGAS to check, so refusals are measured apart
    golden = [item for item in answers if item["type"] in RAGAS_TYPES and not item["should_refuse"]]
    dataset = ragas.EvaluationDataset.from_list([
        {
            "user_input": item["question"],
            "response": item["answer"],
            "retrieved_contexts": item["contexts"],
            "reference": item["expected"],
        }
        for item in golden
    ])
    # retrieval is measured exactly by evaluation/retrieval.py and correctness by key facts: the judge checks grounding
    result = ragas.evaluate(
        dataset,
        metrics=[Faithfulness()],
        llm=LangchainLLMWrapper(judge),
        run_config=RunConfig(max_workers=1, timeout=900),
        raise_exceptions=False,
    )
    scores = result.to_pandas()
    summary = {"faithfulness": scores["faithfulness"].mean(), "nan_rate": scores["faithfulness"].isna().mean()}
    log_run(run_name, "ragas", retriever, summary, {"ragas.json": scores})
    return scores
