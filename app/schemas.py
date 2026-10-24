from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    chunk_count: int
    created_at: datetime


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    document_id: Optional[int] = None


class Source(BaseModel):
    ref: int
    document_id: int
    filename: str
    page: int
    chunk_id: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
