# DocuMind: RAG Document Q&A with Source Citations

Upload PDFs, ask questions in plain English, and get answers grounded in your documents, with citations pointing to the exact page and passage used.

**Stack:** Python, FastAPI, PostgreSQL + pgvector, SQLAlchemy, Pydantic, Google Gemini (embeddings + generation), pypdf, Docker

## How it works
```
PDF -> extract text per page -> chunk (size + overlap) -> embed -> store vectors
Question -> embed -> top-k cosine similarity search -> numbered context -> LLM -> answer with [1] [2] citations
```
- **Chunking** (`app/chunking.py`): per-page chunks with configurable size/overlap, cut on word boundaries, so every chunk maps to one page number.
- **Vector search** (`app/retrieval.py`): pgvector cosine distance in PostgreSQL; SQLite fallback ranks in Python for easy local runs.
- **Grounded answers**: the prompt allows only the retrieved passages and tells the model to say it cannot find the answer otherwise.
- **Citations**: every response lists the source file, page, similarity score and a snippet for each `[n]`.

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload a PDF (multipart `file`) |
| GET | `/documents` | List documents and chunk counts |
| DELETE | `/documents/{id}` | Delete a document and its vectors |
| POST | `/ask` | `{"question": "...", "top_k": 4, "document_id": optional}` |

## Run locally (SQLite)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add a free key from https://aistudio.google.com/apikey
uvicorn app.main:app --reload
```
Open http://localhost:8000/docs, upload a PDF with `POST /documents`, then call `POST /ask`. A sample PDF is included: `samples/company_handbook.pdf` (try "How many days of annual leave do employees get?").

## Run with PostgreSQL + pgvector
```bash
export GEMINI_API_KEY=your-key
docker compose up --build
```

## Tune retrieval
`CHUNK_SIZE`, `CHUNK_OVERLAP` and `TOP_K` are environment variables. Smaller chunks give sharper matches but less context per passage and more embedding calls; larger chunks do the opposite. Re-upload documents after changing chunk settings.

## Tests
```bash
pytest
```
Tests use a deterministic fake embedder and LLM, so no API key is needed.

## Limitations
Scanned (image-only) PDFs are not supported; there is no OCR.
