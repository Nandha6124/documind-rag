from typing import List

import httpx

from .config import settings

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class LLMError(Exception):
    pass


def _post(url: str, payload: dict) -> dict:
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")
    try:
        r = httpx.post(url, headers={"x-goog-api-key": settings.gemini_api_key}, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc


def embed(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Embed texts (batches of up to 100). task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    model = settings.gemini_embed_model
    vectors: List[List[float]] = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        payload = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": settings.embed_dim,
                }
                for t in batch
            ]
        }
        data = _post(f"{BASE}/{model}:batchEmbedContents", payload)
        try:
            vectors.extend(e["values"] for e in data["embeddings"])
        except KeyError as exc:
            raise LLMError("Unexpected embedding response") from exc
    return vectors


def generate(prompt: str) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    data = _post(f"{BASE}/{settings.gemini_chat_model}:generateContent", payload)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError("Unexpected generation response") from exc
