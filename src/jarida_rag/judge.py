import asyncio

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

from . import config
from .rag import load_llm


class LocalChatModel(ChatHuggingFace):
    """RAGAS only calls the judge asynchronously, and ChatHuggingFace refuses that for a local model.
    This runs the normal call in a thread instead."""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)


def load_judge():
    tokenizer, model = load_llm(config.JUDGE_MODEL, config.JUDGE_DEVICE, torch.float32)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.JUDGE_MAX_TOKENS,
        do_sample=False,
        return_full_text=False,  # only the new text, which RAGAS parses as JSON
    )
    judge = LocalChatModel(llm=HuggingFacePipeline(pipeline=generator, model_id=config.JUDGE_MODEL), tokenizer=tokenizer)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL, model_kwargs={"device": config.EMBED_DEVICE})
    return judge, embeddings
