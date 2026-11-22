import csv
import time

import mlflow
import pandas as pd
import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig
from transformers import pipeline

from . import config
from .rag import answer, load_llm
from .store import search

# shortcut: keyword match for "the documents don't say"; check the answers table if refusal_ok looks wrong
REFUSAL_MARKERS = ["does not contain", "do not contain", "no information", "not mentioned", "لا يحتوي", "لا تحتوي", "لا يتضمن", "لا تتضمن", "لا توجد", "غير متوفر"]


def read_questions():
    with open(config.QUESTIONS, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def flat(text):
    # PDF line breaks fall anywhere, so compare texts with single spaces
    return " ".join(text.split())


def evaluate(client, embedder, run_name, reranker=None):
    """Retrieval only, on every answerable question.
    hit: a chunk of the right legal text is retrieved. evidence: the chunk holding the answer itself is retrieved."""
    results = []
    for row in read_questions():
        if not row["expected"]:
            continue
        start = time.time()
        hits = search(client, embedder, row["question"], reranker)
        seconds = time.time() - start

        found = [hit["issue"] == row["issue"] and row["expected"] in hit["title"] for hit in hits]
        results.append({
            "question": row["question"],
            "expected": row["expected"],
            "retrieved": " | ".join(hit["text_number"] or hit["title"][:30] for hit in hits),
            "rank": found.index(True) + 1 if True in found else None,
            "evidence_found": any(flat(row["evidence"]) in flat(hit["text"]) for hit in hits) if row["evidence"] else None,
            "seconds": seconds,
        })
    table = pd.DataFrame(results)
    ranks = [result["rank"] for result in results]
    evidence = [result["evidence_found"] for result in results if result["evidence_found"] is not None]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "embed_model": config.EMBED_MODEL,
            "reranker": config.RERANKER_MODEL if reranker else "none",
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "max_chunks_per_text": config.MAX_CHUNKS_PER_TEXT,
        })
        mlflow.log_metrics({
            "hit_rate": sum(rank is not None for rank in ranks) / len(ranks),
            "mrr": sum(1 / rank for rank in ranks if rank) / len(ranks),
            "evidence_recall": sum(evidence) / len(evidence),
            "avg_search_seconds": table["seconds"].mean(),
            "num_chunks": client.count(config.COLLECTION).count,
            "num_questions": len(ranks),
        })
        mlflow.log_table(table, "retrieval.json")
    return table


def answer_questions(client, embedder, tokenizer, model, reranker=None):
    """Answers the golden set (questions with a reference) and the no-answer questions."""
    answers = []
    for row in read_questions():
        should_refuse = not row["expected"]
        if not (row["reference"] or should_refuse):
            continue
        start = time.time()
        reply, hits = answer(row["question"], client, embedder, tokenizer, model, reranker)
        answers.append({
            "question": row["question"],
            "reference": row["reference"],
            "answer": reply,
            "contexts": [hit["text"] for hit in hits],
            "should_refuse": should_refuse,
            "refused": any(marker in reply.lower() for marker in REFUSAL_MARKERS),
            "seconds": time.time() - start,
        })
    return answers


def load_judge():
    tokenizer, model = load_llm(config.JUDGE_MODEL, config.JUDGE_DEVICE, torch.float32)
    # return_full_text=False: otherwise the prompt is echoed into the JSON that RAGAS parses
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.JUDGE_MAX_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    # pass the tokenizer: without it ChatHuggingFace loads the default model's chat template, silently wrong
    judge_llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=generator, model_id=config.JUDGE_MODEL), tokenizer=tokenizer)
    judge_embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL, model_kwargs={"device": config.EMBED_DEVICE})
    return judge_llm, judge_embeddings


def ragas_eval(answers, judge_llm, judge_embeddings, run_name, reranker_used):
    # a correct "not found" has no claims to check, so RAGAS would score good behaviour as 0: refusals are scored apart
    golden = [item for item in answers if item["reference"] and not item["should_refuse"]]
    dataset = EvaluationDataset.from_list([
        {"user_input": item["question"], "response": item["answer"], "retrieved_contexts": item["contexts"], "reference": item["reference"]}
        for item in golden
    ])
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]
    result = ragas_evaluate(
        dataset,
        metrics=metrics,
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        # one call at a time: parallel calls on one GPU time out and come back as NaN
        run_config=RunConfig(max_workers=1, timeout=900),
        raise_exceptions=False,
    )
    scores = result.to_pandas()
    names = [metric.name for metric in metrics]

    refusals = [item["refused"] == item["should_refuse"] for item in answers]
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "embed_model": config.EMBED_MODEL,
            "reranker": config.RERANKER_MODEL if reranker_used else "none",
            "top_k": config.TOP_K,
            "golden_questions": len(golden),
        })
        mlflow.log_metrics({name: scores[name].mean() for name in names})
        mlflow.log_metrics({
            # NaN means the judge failed on that row: a high rate makes the means above meaningless
            "nan_rate": scores[names].isna().to_numpy().mean(),
            "refusal_accuracy": sum(refusals) / len(refusals),
            "avg_answer_seconds": sum(item["seconds"] for item in answers) / len(answers),
        })
        mlflow.log_table(scores, "ragas.json")
        mlflow.log_table(pd.DataFrame(answers).drop(columns="contexts"), "answers.json")
    return scores
