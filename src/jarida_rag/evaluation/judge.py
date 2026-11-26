import asyncio

import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

from .. import config
from ..rag import load_llm


class LocalChatModel(ChatHuggingFace):
    """RAGAS only calls the judge asynchronously, and ChatHuggingFace refuses that for a local model.
    This runs the normal call in a thread instead."""

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await asyncio.to_thread(self._generate, messages, stop, **kwargs)


def load_judge():
    # fp16 compute is about 3x faster than fp32 on a T4; fall back to torch.float32 if nan_rate rises
    tokenizer, model = load_llm(config.JUDGE_MODEL, config.JUDGE_DEVICE, torch.float16)
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.JUDGE_MAX_TOKENS,
        do_sample=False,
        return_full_text=False,  # only the new text, which RAGAS parses as JSON
    )
    return LocalChatModel(llm=HuggingFacePipeline(pipeline=generator, model_id=config.JUDGE_MODEL), tokenizer=tokenizer)
