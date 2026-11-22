import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = Path(os.environ.get("JARIDA_PDF_DIR", REPO_DIR / "data"))
WORK_DIR = Path(os.environ.get("JARIDA_WORK", REPO_DIR / "work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = REPO_DIR / "data" / "manifest.csv"
QUESTIONS = REPO_DIR / "data" / "questions.csv"
QDRANT_DIR = WORK_DIR / "qdrant"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(WORK_DIR / 'mlflow.db').as_posix()}")
COLLECTION = "gazette"

MIN_ARABIC_RATIO = 0.5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 30
TOP_K = 5
MAX_CHUNKS_PER_TEXT = 2

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DEVICE = "cuda:0"
EMBED_BATCH = 16

LLM_MODEL = "Qwen/Qwen3-8B"
LLM_DEVICE = "cuda:1"
MAX_NEW_TOKENS = 768
