import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = Path(os.environ.get("JARIDA_PDF_DIR", REPO_DIR / "data"))
WORK_DIR = Path(os.environ.get("JARIDA_WORK", REPO_DIR / "work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = REPO_DIR / "data" / "manifest.csv"
QUESTIONS = REPO_DIR / "data" / "questions.csv"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(WORK_DIR / 'mlflow.db').as_posix()}")
DATABASE_URL = os.environ.get("DATABASE_URL")  # Postgres + pgvector (Neon); never written in the code
TABLE = "chunks"

# The running header sits above 91% of the page height. Measured on 7467/7470/7497: header boxes start at 0.918,
# body text ends at 0.905, so the margin is thin; recheck when adding issues from other years or layouts.
HEADER_LINE = 0.91
MIN_PAGE_CHARS = 50  # fewer characters: a scan, map or figure
MIN_ARABIC_RATIO = 0.5  # below: a table or French page, kept but flagged
MIN_DOCUMENT_ARABIC_RATIO = 0.3  # below, for the median page: the whole issue is unreadable
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 30
# Chunking switches, compared in notebooks/chunking_lab.ipynb
CHUNK_HEADER = "short"  # "short": one-line real title on every chunk; "first_200": the old 200-character prefix
UNWRAP_LINES = True  # join PDF line wraps so chunks end on sentences, not mid-line
SPLIT_PREAMBLE = True  # chunk the preamble ("بناء على ...") apart from the articles
TOP_K = 5
MAX_CHUNKS_PER_TEXT = 2
RERANK_CANDIDATES = 20

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DEVICE = "cuda:0"
EMBED_BATCH = 16
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

LLM_MODEL = "Qwen/Qwen3-8B"
LLM_DEVICE = "cuda:1"
MAX_NEW_TOKENS = 768

# A different model family from the answering LLM, so it doesn't grade its own answers.
# The judge runs after everything else is freed: transformers pipelines move their model to GPU 0 anyway,
# so the judge goes there on purpose and the RAGAS embeddings use GPU 1.
JUDGE_MODEL = "google/gemma-4-12B-it"
JUDGE_DEVICE = "cuda:0"
JUDGE_EMBED_DEVICE = "cuda:1"
JUDGE_MAX_TOKENS = 2048  # faithfulness lists every claim as JSON; 1024 cut long answers mid-JSON
