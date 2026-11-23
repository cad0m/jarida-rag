import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from . import config
from .store import search

SYSTEM_PROMPT = """You are a legal assistant specialized in the Moroccan Official Gazette (الجريدة الرسمية).

Rules:
1. Answer only from the provided context. Never use outside knowledge.
2. Answer in the language of the question: formal Arabic for Arabic questions, English for English questions.
3. Copy decree numbers, law numbers and dates exactly as written in the context. In English answers, translate مرسوم as "decree", قرار as "order", ظهير شريف as "dahir" and قانون as "law".
4. After each fact, cite its source exactly as written in the context, like this: [BO 7470, 2.24.830, المادة 5, p. 8].
5. If several legal texts in the context are relevant, name each one separately.
6. If the context does not contain the answer, say so clearly in the language of the question. Do not guess.
7. The context comes from PDF extraction and may contain small spelling errors in Arabic. Ignore them, but never change numbers.
8. Be concise and direct."""


def load_llm(model_id=config.LLM_MODEL, device=config.LLM_DEVICE, compute_dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        dtype=compute_dtype,  # the layers 4-bit doesn't cover (embeddings, norms) would otherwise load in float32
        device_map={"": device},
    )
    return tokenizer, model


def source(hit):
    parts = [f"BO {hit['issue']}", hit["text_number"], f"المادة {hit['article']}" if hit["article"] else "", f"p. {hit['page_start']}"]
    return "[" + ", ".join(part for part in parts if part) + "]"


def generate(tokenizer, model, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    set_seed(0)
    # Qwen3's recommended settings without thinking; greedy decoding makes it repeat itself
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer(question, client, embedder, tokenizer, model, reranker=None):
    with mlflow.start_span(name="answer") as span:
        span.set_inputs({"question": question})

        with mlflow.start_span(name="retrieve") as retrieve_span:
            retrieve_span.set_attribute("reranker", bool(reranker))
            hits = search(client, embedder, question, reranker)
            retrieve_span.set_outputs([f"{source(hit)} {hit['title'][:80]}" for hit in hits])

        context = "\n\n".join(f"{source(hit)} {hit['text']}" for hit in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        with mlflow.start_span(name="generate") as generate_span:
            generate_span.set_inputs({"prompt": messages[1]["content"]})
            reply = generate(tokenizer, model, messages)
            generate_span.set_outputs(reply)

        span.set_outputs(reply)
    return reply, hits
