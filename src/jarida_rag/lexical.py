"""BM25 over the chunks, fused with the dense ranking: exact decree numbers and names that embeddings blur."""
import re

from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"\d[\d./]*\d|\w+")  # "2.24.830" stays one token
DIACRITICS = re.compile("[ً-ْ]")
LETTER_VARIANTS = str.maketrans("أإآىة", "ااايه")
# the article and its attached prepositions; a bare "و" is a root letter too often ("وزير") to strip
ARTICLE = re.compile(r"^(?:وال|بال|فال|كال|لل|ال)(?=\w{2})")


def tokenize(text):
    text = DIACRITICS.sub("", text.lower()).translate(LETTER_VARIANTS)
    return [ARTICLE.sub("", token) for token in TOKEN.findall(text)]


def build(chunks):
    return BM25Okapi([tokenize(chunk["embed_text"]) for chunk in chunks]), chunks


def search(index, question, n):
    bm25, chunks = index
    scores = bm25.get_scores(tokenize(question))
    return [chunks[i] for i in scores.argsort()[::-1][:n] if scores[i] > 0]


def fuse(*rankings, k=60):
    """Reciprocal rank fusion: a chunk ranked high by either retriever rises, whatever the score scales."""
    scores, hits = {}, {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0) + 1 / (k + rank)
            hits[hit["id"]] = hit
    return [hits[i] for i in sorted(scores, key=scores.get, reverse=True)]
