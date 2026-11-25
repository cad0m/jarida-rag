# jarida-rag

Ask legal questions in Arabic or English about the Moroccan Official Gazette (الجريدة الرسمية) and get answers
grounded in the gazette, with a citation for every fact: `[BO 7470, 2.24.830, المادة 5, p. 8]`.

**Status:** working end to end on three born-digital issues (7467, 7470, 7497), evaluated with RAGAS and tracked in MLflow.

## How it works

| Step | File | What it does |
|---|---|---|
| Extract | `src/jarida_rag/pdf.py` | PDF → clean page text with pypdfium2: crops the running header by position, repairs broken Arabic ligatures (`إعالن` → `إعلان`), mirrored brackets and reversed Latin words, and flags every page it doesn't keep |
| Chunk | `src/jarida_rag/chunk.py` | Pages → legal texts (decrees, orders, notices, studies) → chunks split by article, each labelled with its decree number, article and pages |
| Index | `src/jarida_rag/store.py` | BGE-M3 embeddings in Postgres + pgvector; search keeps at most 2 chunks per legal text, then an optional BGE cross-encoder reranks them |
| Answer | `src/jarida_rag/rag.py` | Qwen3-8B (4-bit) answers only from the retrieved context and refuses with a fixed sentence when the answer isn't there; each answer is traced in MLflow |
| Evaluate | `src/jarida_rag/evaluate.py`, `judge.py` | Retrieval metrics (hit rate, MRR, evidence recall) on 51 questions; RAGAS (faithfulness, answer relevancy, context precision and recall) with Gemma 4 as an independent judge |

Settings (models, chunking, top-k, GPUs) are plain variables in `src/jarida_rag/config.py`.

## Results (3 issues, final system with reranker)

| Metric | Value |
|---|---|
| Faithfulness | 0.94 |
| Context precision / recall | 1.00 / 0.97 |
| Refusals on out-of-scope questions | 3 / 3 |
| Retrieval MRR (51 questions) | 0.96 |

The reranker raised context precision from 0.95 to 1.00 for about 1.5 s more per question.
Scores are on a small, easy corpus; faithfulness does not measure correctness (two answers misread a number in a correctly retrieved text).

## Data

- PDFs are not in git: put them in `data/`, or attach them as a Kaggle dataset.
- `data/manifest.csv`: one row per issue (`filename, issue, date, lang`).
- `data/questions.csv`: the golden set. `expected` is the decree number (or title phrase) that answers the question,
  `evidence` an exact phrase from the gazette containing the answer, `reference` a reference answer for RAGAS.
  Questions with an empty `expected` have no answer in the corpus.

## Run on Kaggle (GPU T4 x2)

Add these Kaggle secrets: `NEON_DATABASE_URL` (Postgres with pgvector), `DAGSHUB_TOKEN` (MLflow) and `HF_TOKEN`
(the Gemma judge is gated). Then run `notebooks/kaggle.ipynb` top to bottom.
`notebooks/chunking_lab.ipynb` compares chunking variants on retrieval in a few minutes, without the LLM or the database.

## Run locally (CPU, extraction and chunking only)

```bash
pip install -e .
python -X utf8 -m jarida_rag.chunk
```

Prints every detected legal text with its pages and checks them against each issue's printed table of contents.

## Known limits

- Scanned issues (no text layer) need OCR, and older issues typeset in a non-Unicode Arabic font extract as garbage.
- Some recent issues (7167, 7168, 7401, 7439, 7441) extract body text correctly but scramble bold title lines.
- Study and opinion pages keep some line fragments out of order, with footnotes mixed into the text.
