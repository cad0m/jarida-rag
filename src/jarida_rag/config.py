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

HEADER_LINE = 0.91
MIN_PAGE_CHARS = 50  # fewer characters: a scan, map or figure
MIN_ARABIC_RATIO = 0.5  # below: a table or French page, kept but flagged
MIN_DOCUMENT_ARABIC_RATIO = 0.3  # below, for the median page: the whole issue is unreadable
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 30

# Chosen with notebooks/chunking_lab.ipynb (runs lab-v3-* on DagsHub); the other values are kept to reproduce the comparison
CHUNK_HEADER = "short"  # "short": a one-line title on every chunk; "first_200": the first 200 characters on later chunks
UNWRAP_LINES = True  # join PDF line wraps so chunks end on sentences, not mid-line
SPLIT_PREAMBLE = False  # True: the preamble ("بناء على ...") becomes its own chunks; it lowered evidence_mrr
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

JUDGE_MODEL = "google/gemma-4-12B-it"
JUDGE_DEVICE = "cuda:0"
JUDGE_EMBED_DEVICE = "cuda:1"
JUDGE_MAX_TOKENS = 2048  # faithfulness lists every claim as JSON; 1024 cut long answers mid-JSON
