import re

from .. import config

ALEF_VARIANTS = str.maketrans("أإآ", "ااا")  # "أمحمد" and "امحمد" are the same name
NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def contains(text, phrase):
    # PDF line breaks fall anywhere, so compare with single spaces
    return " ".join(phrase.split()) in " ".join(text.split())


def normalize(text):
    """Lowercase, small English numbers as digits, one alef, a decimal point for "39,94",
    and no Arabic article: "للتجهيز" (ل + ال) still holds "تجهيز"."""
    text = re.sub(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", lambda match: NUMBER_WORDS[match.group()], text.lower())
    text = re.sub(r"(?<=\d),(?=\d)", ".", text).translate(ALEF_VARIANTS)
    return re.sub(r"(?<!\S)ال", "", text)


def has_fact(reply, fact):
    reply, fact = normalize(reply), normalize(fact)
    if re.fullmatch(r"[\d.,]+", fact):
        # a number counts only as a whole number: "2" must not match inside "2.26.28"
        return re.search(rf"(?<![\d.,]){re.escape(fact)}(?![\d]|[.,]\d)", reply) is not None
    if not re.search("[ء-ي]", fact):
        # Latin facts are whole words: "no" must not match inside "not"; Arabic prefixes (و، ب، لل) attach, so no boundary there
        return re.search(rf"\b{re.escape(' '.join(fact.split()))}\b", " ".join(reply.split())) is not None
    return contains(reply, fact)


def has_key_facts(reply, key_fact):
    """Every "&" part of the key fact appears in the answer, any of its "|" alternatives counting."""
    reply = re.sub(r"\[[^\]]*\]", " ", reply)  # citation labels are full of numbers
    return all(any(has_fact(reply, alternative.strip()) for alternative in part.split("|")) for part in key_fact.split("&"))


def miss_stage(candidate_rank, capped_rank, final_rank, reranked):
    """Where the evidence chunk was lost (ranks are 1-based, 0 = absent): tells coverage problems from ranking ones."""
    if final_rank:
        return "found"
    if not candidate_rank:
        return "not in candidates"
    if not capped_rank:
        return "cut by per-text cap"
    if reranked and capped_rank > config.RERANK_CANDIDATES:
        return "outside rerank pool"
    return "ranked below top K"
