# jarida-rag

Ask questions in Arabic or English about the Moroccan Official Gazette (الجريدة الرسمية) and get an answer taken
only from the gazette, with the source of every fact: `[BO 7470, 2.24.830, المادة 5, p. 8]`.
When the gazette doesn't hold the answer, it says so instead of guessing.

## How it works

1. **Collect** (`src/jarida_rag/sources/`): lists every issue on sgg.gov.ma, downloads the new PDFs and uploads them
   to Google Drive with rclone. A `documents` table in Postgres tracks each one (listed, stored, failed, indexed).
2. **Extract** (`pdf.py`): turns each PDF page into clean Arabic text and repairs the usual extraction errors.
3. **Chunk** (`chunk.py`): splits an issue into its legal texts (decrees, orders, notices), then into chunks by article,
   each labelled with its text number, article and pages.
4. **Index** (`store.py`): embeds the chunks with BGE-M3 into Postgres + pgvector. `update_index` reads the PDFs from
   the shared Drive folder and only indexes documents that are new or were chunked with older settings.
5. **Search**: finds the closest chunks, keeps a few per legal text, and reranks them with a cross-encoder.
6. **Answer** (`rag.py`): Qwen3-8B answers from those chunks only and cites each one.
7. **Evaluate** (`evaluation/`): a golden set of questions checks retrieval, answers and refusals; RAGAS checks that
   answers stay faithful to the context. Every run is tracked in MLflow.

All settings (models, chunk size, top-k, GPUs) are plain variables in `src/jarida_rag/config.py`.

## Run

**Collect** (on your machine, with rclone set up for your Drive):

```bash
pip install -e .
python -m jarida_rag.sources.collect
```

Then share the Drive folder as "anyone with the link", so Kaggle can read the PDFs without logging in.

**Index, answer and evaluate** (Kaggle, GPU T4 x2), with the secrets `NEON_DATABASE_URL`, `DAGSHUB_TOKEN` and `HF_TOKEN`:

- `notebooks/retrieval.ipynb`: compares chunking and search settings, without the LLM.
- `notebooks/generation.ipynb`: indexes, answers the golden questions and runs RAGAS.

**Chat** (`frontend/`): a React chat page that calls the RAG API.

## Data

- `data/manifest.csv`: the issues used for evaluation (PDFs are not in git).
- `data/questions.csv`: the golden questions, each with its expected answer, a quote from the gazette and its key fact.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

They run on CPU without models and check the answer scorer, the BM25 tokenizer, chunking and the gazette site parser.

## Known limits

- Scanned issues need OCR, and some older issues use a font that extracts as garbage.
- Tables in studies and opinions lose part of their layout.
- Questions that need a full list across many texts get only the top results.
- Links between a text and the texts it amends or repeals come from titles only, and search doesn't filter by date
  or text type yet.
