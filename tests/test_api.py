import os

os.environ["DATABASE_URL"] = "sqlite:///./test.db"

import hashlib  # noqa: E402
import math  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, get_embedder, get_generator  # noqa: E402

DIM = 64


def fake_embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Deterministic bag-of-words hashing embedding: shared words => higher cosine similarity."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for w in t.lower().split():
            w = w.strip(".,?!")
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


prompts = []


def fake_generate(prompt):
    prompts.append(prompt)
    return "Answer based on context [1]."


app.dependency_overrides[get_embedder] = lambda: fake_embed
app.dependency_overrides[get_generator] = lambda: fake_generate
client = TestClient(app)


def make_pdf(pages):
    """Build a minimal valid PDF with one text line per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return out.encode("latin-1")


def upload(name, pages):
    return client.post("/documents", files={"file": (name, make_pdf(pages), "application/pdf")})


def test_upload_ask_with_citations():
    r = upload("cats.pdf", ["Cats sleep sixteen hours a day and purr when happy.", "Dogs bark loudly at strangers."])
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["chunk_count"] == 2

    r = client.post("/ask", json={"question": "How many hours do cats sleep?", "document_id": doc["id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["page"] == 1  # best match is the cats page
    assert body["sources"][0]["filename"] == "cats.pdf"
    assert body["sources"][0]["ref"] == 1
    assert "[1] Cats sleep" in prompts[-1]
    assert "ONLY" in prompts[-1]


def test_list_and_delete():
    doc = upload("temp.pdf", ["Temporary content for deletion."]).json()
    assert any(d["id"] == doc["id"] for d in client.get("/documents").json())
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_rejects_non_pdf_and_empty_pdf():
    r = client.post("/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
    assert upload("blank.pdf", [""]).status_code == 422


def test_ask_validation_and_unknown_document():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question?", "document_id": 99999}).status_code == 404
