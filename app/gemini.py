import time
from typing import List

import httpx

from .config import settings

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class LLMError(Exception):
    pass


RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4


def _post(url: str, payload: dict, retries: int = MAX_RETRIES) -> dict:
    """POST with exponential backoff (1s, 2s, 4s, ...) on rate limits and temporary server errors."""
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set")
    last = "unknown error"
    for attempt in range(retries + 1):
        try:
            r = httpx.post(url, headers={"x-goog-api-key": settings.gemini_api_key}, json=payload, timeout=60)
            if r.status_code in RETRY_STATUS:
                last = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r.json()
        except httpx.TransportError as exc:
            last = str(exc)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise LLMError(f"Gemini is busy or unreachable after {retries + 1} attempts ({last}). Try again shortly.")


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
    """Try the main chat model, then the fallback model if the main one stays busy."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    last_error = None
    for model in dict.fromkeys([settings.gemini_chat_model, settings.gemini_fallback_model]):
        try:
            data = _post(f"{BASE}/{model}:generateContent", payload, retries=2)
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise LLMError("Unexpected generation response") from exc
        except LLMError as exc:
            if "busy" not in str(exc):
                raise
            last_error = exc
    raise LLMError("The AI service is busy right now. Please try again in a moment.") from last_error
