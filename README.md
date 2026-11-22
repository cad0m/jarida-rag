# jarida-rag

Ask legal questions in Arabic or English about the Moroccan Official Gazette (الجريدة الرسمية) and get answers
with citations like `[BO 7470, p. 24]`.

**Status:** milestone 1, working on a handful of born-digital Arabic issues before scaling year by year.

## How it works

| Step | File | What it does |
|---|---|---|
| Extract | `src/jarida_rag/pdf.py` | PDF → clean pages with pypdfium2 (fixes broken Arabic ligatures, skips table-of-contents and unreadable pages) |
| Chunk | `src/jarida_rag/chunk.py` | Pages → legal texts (decrees, orders, dahirs) → chunks with metadata |
| Index | `src/jarida_rag/store.py` | BGE-M3 embeddings in a local Qdrant collection |
| Answer | `src/jarida_rag/rag.py` | Retrieve the top chunks, answer with Qwen3-8B (4-bit), traced in MLflow |
| Evaluate | `src/jarida_rag/evaluate.py` | Hit rate and MRR on `data/questions.csv`, logged to MLflow |

Settings (models, chunk size, top-k, GPUs) are plain variables in `src/jarida_rag/config.py`.

## Data

- PDFs are not in git. Put them in `data/` locally, or attach them as a Kaggle dataset.
- `data/manifest.csv`: one row per PDF (`filename, issue, date, lang`).
- `data/questions.csv`: evaluation questions. `expected` is the decree number that answers it, or a phrase from the
  title for studies and notices. Leave it empty for questions the gazette can't answer.
- Experiments and traces are logged to MLflow on DagsHub (local `work/mlflow.db` when no `MLFLOW_TRACKING_URI` is set).

## Run on Kaggle (GPU T4 x2)

Turn on internet in the notebook settings, then run `notebooks/kaggle.ipynb` top to bottom.
The embedder goes on GPU 0, the LLM on GPU 1.

## Run locally (CPU, no LLM)

```bash
pip install -e .
python -X utf8 -m jarida_rag.chunk
```

This prints every detected legal text with its pages, which is the way to check extraction and chunking.
