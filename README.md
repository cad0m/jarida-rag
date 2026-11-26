# jarida-rag

Ask legal questions in Arabic or English about the Moroccan Official Gazette (الجريدة الرسمية) and get answers
grounded in the gazette, with a citation for every fact: `[BO 7470, 2.24.830, المادة 5, p. 8]`.

**Status:** working end to end on three born-digital issues (7467, 7470, 7497), evaluated with RAGAS and tracked in MLflow.

## How it works

| Step | File | What it does |
|---|---|---|
| Extract | `src/jarida_rag/pdf.py` | PDF → clean page text with pypdfium2: crops the running header by position, repairs broken Arabic ligatures (`إعالن` → `إعلان`), mirrored brackets and reversed Latin words, and flags every page it doesn't keep |
| Chunk | `src/jarida_rag/chunk.py` | Pages → legal texts (decrees, orders, notices, studies) → chunks split by article, each labelled with its decree number, text type, the texts it amends, article and pages |
| Index | `src/jarida_rag/store.py`, `lexical.py` | BGE-M3 embeddings in Postgres + pgvector, optionally fused with BM25; search keeps at most 2 chunks per legal text, then a BGE cross-encoder reranks them |
| Answer | `src/jarida_rag/rag.py` | Qwen3-8B (4-bit) answers only from the retrieved context and refuses with a fixed sentence when the answer isn't there; each answer is traced in MLflow |
| Evaluate retrieval | `src/jarida_rag/evaluation/retrieval.py` | Is the right legal text in the top 5, and the chunk holding the answer? 76 answerable questions, no LLM |
| Evaluate answers | `src/jarida_rag/evaluation/generation.py`, `judge.py` | Key-fact and refusal accuracy by string match, then RAGAS faithfulness and answer relevancy with Gemma 4 as an independent judge |

Settings (models, chunking, top-k, GPUs) are plain variables in `src/jarida_rag/config.py`.

## Results (3 issues, final system with reranker)

| Metric | Value |
|---|---|
| Right legal text in the top 5 | 73 / 76 |
| Chunk holding the answer in the top 5 | 62 / 76 |
| Correct key fact (15 single-fact questions) | 13 / 15 |
| Faithfulness / answer relevancy (RAGAS) | 0.87 / 0.94 |
| Refusals on out-of-scope questions (the fixed sentence alone) | 7 / 10 |

Measured improvements (evidence MRR, the rank of the chunk that holds the answer):
a one-line title on every chunk instead of its first 200 characters, 0.65 → 0.71 (`notebooks/retrieval.ipynb`),
and the reranker, 0.60 → 0.71 for about 1.6 s more per question.
Faithfulness does not measure correctness, so key facts are also checked by string match. Of the three missed refusals,
two declined correctly but not with the fixed sentence alone (one in its own words, one with an explanation before it),
and one quoted the replacement text of a repealed article as the repealed one.

## Data

- PDFs are not in git: put them in `data/`, or attach them as a Kaggle dataset.
- `data/manifest.csv`: one row per issue (`filename, issue, date, lang`).
- `data/questions.csv`: the golden set (86 questions, each checked against the extracted text).
  `expected` is the normalized answer (`OUT_OF_SCOPE` when the gazette can't answer), `evidence` a verbatim quote,
  `key_fact` the minimal fact a correct answer contains (`&` = and, `|` = or), `reference` the decree number(s).

## Run on Kaggle (GPU T4 x2)

Add these Kaggle secrets: `NEON_DATABASE_URL` (Postgres with pgvector), `DAGSHUB_TOKEN` (MLflow) and `HF_TOKEN`
(the Gemma judge is gated). Then run the two stages in order, each top to bottom:

1. `notebooks/retrieval.ipynb`: compares chunking and retrieval variants in memory, without the LLM or the database.
   Write the winner into `config.py`.
2. `notebooks/generation.ipynb`: indexes into Postgres, answers, scores key facts and refusals, then runs RAGAS.

Every MLflow run is tagged `stage` = `retrieval`, `generation` or `ragas`.

## Tests (CPU, no models)

```bash
pip install -e ".[dev]"
pytest
```

They check the answer scorer, the BM25 tokenizer, and chunking. Each issue's printed table of contents serves as the answer key for text detection.
The chunking tests are skipped when the PDFs aren't in `data/`.

## Known limits

- Scanned issues (no text layer) need OCR, and older issues typeset in a non-Unicode Arabic font extract as garbage.
- Some recent issues (7167, 7168, 7401, 7439, 7441) extract body text correctly but scramble bold title lines.
- Study and opinion pages keep some line fragments out of order, with footnotes mixed into the text.
- Each chunk stores its text type and the texts its title says it amends, but search doesn't filter on them yet.
  With 3 issues, no question needs a date or type filter, and most amended texts aren't indexed.
  This will be useful once more years are ingested.
