import io
from typing import Callable, List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PyPdfError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import gemini
from .chunking import chunk_pages
from .config import settings
from .db import get_db, init_db
from .models import Chunk, Document
from .retrieval import top_k_chunks
from .schemas import AskRequest, AskResponse, DocumentOut, Source

app = FastAPI(title="DocuMind: RAG Document Q&A")
init_db()

PROMPT = """You answer questions using ONLY the numbered context passages below.
- Cite the passages you used like [1] or [2] right after each claim.
- If the context does not contain the answer, reply exactly: "I could not find that in the uploaded documents."
- Do not use outside knowledge.

Context:
{context}

Question: {question}
Answer:"""


def get_embedder() -> Callable[..., List[List[float]]]:
    return gemini.embed


def get_generator() -> Callable[[str], str]:
    return gemini.generate


def _doc_out(db: Session, doc: Document) -> DocumentOut:
    count = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id))
    return DocumentOut(id=doc.id, filename=doc.filename, chunk_count=count or 0, created_at=doc.created_at)


@app.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    file: UploadFile = File(...), db: Session = Depends(get_db), embed=Depends(get_embedder)
):
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except (PyPdfError, ValueError, OSError):
        raise HTTPException(400, "Could not read file as a PDF")

    pieces = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not pieces:
        raise HTTPException(422, "No extractable text found (scanned PDFs are not supported)")

    try:
        vectors = embed([text for _, text in pieces], task_type="RETRIEVAL_DOCUMENT")
    except gemini.LLMError as exc:
        raise HTTPException(502, str(exc))

    doc = Document(filename=file.filename or "upload.pdf")
    doc.chunks = [
        Chunk(chunk_index=i, page=page, content=text, embedding=vec)
        for i, ((page, text), vec) in enumerate(zip(pieces, vectors))
    ]
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return _doc_out(db, doc)


@app.get("/documents", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    return [_doc_out(db, d) for d in db.scalars(select(Document).order_by(Document.id))]


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    db.delete(doc)
    db.commit()


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    db: Session = Depends(get_db),
    embed=Depends(get_embedder),
    generate=Depends(get_generator),
):
    if req.document_id is not None and db.get(Document, req.document_id) is None:
        raise HTTPException(404, "Document not found")
    try:
        q_vec = embed([req.question], task_type="RETRIEVAL_QUERY")[0]
        hits = top_k_chunks(db, q_vec, req.top_k or settings.top_k, req.document_id)
        if not hits:
            return AskResponse(answer="No documents have been uploaded yet.", sources=[])
        context = "\n\n".join(f"[{i}] {c.content}" for i, (c, _) in enumerate(hits, start=1))
        answer = generate(PROMPT.format(context=context, question=req.question))
    except gemini.LLMError as exc:
        raise HTTPException(502, str(exc))

    sources = [
        Source(
            ref=i,
            document_id=c.document_id,
            filename=c.document.filename,
            page=c.page,
            chunk_id=c.id,
            score=round(score, 4),
            snippet=c.content[:200],
        )
        for i, (c, score) in enumerate(hits, start=1)
    ]
    return AskResponse(answer=answer, sources=sources)
