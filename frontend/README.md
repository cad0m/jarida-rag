# Jarida frontend

The chat page for the gazette RAG: React, Vite and Motion. Chats and settings live in the browser (localStorage).

## Run

```bash
npm install
npm run dev
```

## Deploy on Vercel

New project → import the repo → Root Directory `frontend` (Vite is detected). Optionally set `VITE_API_URL`;
anyone can also paste the API URL in Settings → Connection.

## The API it calls

```
GET  /health  -> 200
POST /ask     {"question": "..."}
           -> {"answer": "...", "sources": [{"issue", "date", "title", "text_number", "article", "page_start", "text", "drive_id"}]}
```

`sources` are the hits `rag.answer()` returns. `text` (the chunk) is shown when a citation is clicked; `drive_id`
(from the documents table) links the source to its PDF.
The API must allow the site's origin (CORS), and must be served over HTTPS since Vercel is.
`notebooks/serve.ipynb` runs it on Kaggle's GPUs behind a free Cloudflare tunnel (`src/jarida_rag/api.py`).
