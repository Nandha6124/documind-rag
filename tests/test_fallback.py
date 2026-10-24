import pytest

from app import gemini
from app.config import settings


def test_falls_back_to_second_model_when_first_is_busy(monkeypatch):
    monkeypatch.setattr(settings, "gemini_chat_model", "main-model")
    monkeypatch.setattr(settings, "gemini_fallback_model", "backup-model")
    tried = []

    def fake_post(url, payload, retries=0):
        tried.append(url.split("/models/")[1].split(":")[0])
        if "main-model" in url:
            raise gemini.LLMError("Gemini is busy or unreachable after 3 attempts (HTTP 503)")
        return {"candidates": [{"content": {"parts": [{"text": " ok "}]}}]}

    monkeypatch.setattr(gemini, "_post", fake_post)
    assert gemini.generate("hi") == "ok"
    assert tried == ["main-model", "backup-model"]


def test_both_busy_gives_friendly_error(monkeypatch):
    monkeypatch.setattr(gemini, "_post", lambda *a, **k: (_ for _ in ()).throw(gemini.LLMError("busy")))
    with pytest.raises(gemini.LLMError, match="busy right now"):
        gemini.generate("hi")
