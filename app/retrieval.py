import math
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import IS_POSTGRES
from .models import Chunk


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def top_k_chunks(
    db: Session, query_vec: List[float], k: int, document_id: Optional[int] = None
) -> List[Tuple[Chunk, float]]:
    """Return the k most similar chunks as (chunk, cosine_similarity), best first."""
    if IS_POSTGRES:
        dist = Chunk.embedding.cosine_distance(query_vec)
        stmt = select(Chunk, dist).order_by(dist).limit(k)
        if document_id is not None:
            stmt = stmt.where(Chunk.document_id == document_id)
        return [(c, 1.0 - float(d)) for c, d in db.execute(stmt).all()]

    stmt = select(Chunk)
    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)
    scored = [(c, _cosine(query_vec, c.embedding)) for c in db.scalars(stmt)]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]
