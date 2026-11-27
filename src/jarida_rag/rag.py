import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from . import config
from .store import search

# Fixed sentences, so a refusal can be detected exactly instead of guessed from keywords
NOT_FOUND_EN = "The gazette documents provided do not contain this information."
NOT_FOUND_AR = "لا تتضمن وثائق الجريدة الرسمية المقدمة هذه المعلومة."

SYSTEM_PROMPT = f"""You are a legal assistant specialized in the Moroccan Official Gazette (الجريدة الرسمية).

Rules:
1. Answer only from the provided context. Never use outside knowledge.
2. Answer in the language of the question, even though the context is in Arabic: formal Arabic for Arabic questions, English for English questions.
3. Copy decree numbers, law numbers and dates exactly as written in the context. In English answers, translate مرسوم as "decree", قرار as "order", ظهير شريف as "dahir" and قانون as "law".
4. When you refer to a legal text, name its type and number in the sentence itself (for example "decree 2.24.830"), copied from the label of the passage you use. Never write an article or text number that is not in that label or passage.
5. After each fact, cite its source exactly as written in the context, like this: [BO 7470, 2.24.830, المادة 5, p. 8].
6. If several legal texts in the context are relevant, name each one separately.
7. Before saying a text is in force, check whether another text in the context amends, completes or repeals it (بتغيير، بتتميم، بنسخ، ينسخ، تنسخ). If one does, say which text changed it and that the newer text applies. Keep the two apart: "Article 2 of decree 2.25.761 repeals Article 5 of decree 2.11.89".
8. If the context does not contain the answer, reply with exactly this sentence and nothing else, even if you know the answer from elsewhere:
   English: {NOT_FOUND_EN}
   Arabic: {NOT_FOUND_AR}
9. The context comes from PDF extraction and may contain small spelling errors in Arabic. Ignore them, but never change numbers.
10. Be concise and direct."""


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


def answer(question, pool, embedder, tokenizer, model, reranker=None, lexical_index=None):
    with mlflow.start_span(name="answer") as span:
        span.set_inputs({"question": question})

        with mlflow.start_span(name="retrieve") as retrieve_span:
            hits = search(pool, embedder, question, reranker, lexical_index)
            retrieve_span.set_outputs([source(hit) for hit in hits])

        context = "\n\n".join(f"{source(hit)} {hit['text']}" for hit in hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        with mlflow.start_span(name="generate"):
            reply = generate(tokenizer, model, messages)

        span.set_outputs(reply)
    return reply, hits
