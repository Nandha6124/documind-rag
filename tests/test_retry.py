import httpx
import pytest

from app import gemini
from app.config import settings


class FakeResp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test")
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)


def test_retries_503_then_succeeds(monkeypatch):
    calls = iter([FakeResp(503), FakeResp(503), FakeResp(200, {"ok": 1})])
    monkeypatch.setattr(gemini.httpx, "post", lambda *a, **k: next(calls))
    assert gemini._post("http://x", {}) == {"ok": 1}


def test_gives_up_after_max_retries(monkeypatch):
    n = []
    monkeypatch.setattr(gemini.httpx, "post", lambda *a, **k: n.append(1) or FakeResp(503))
    with pytest.raises(gemini.LLMError, match="busy"):
        gemini._post("http://x", {})
    assert len(n) == gemini.MAX_RETRIES + 1


def test_client_errors_are_not_retried(monkeypatch):
    n = []
    monkeypatch.setattr(gemini.httpx, "post", lambda *a, **k: n.append(1) or FakeResp(404))
    with pytest.raises(gemini.LLMError):
        gemini._post("http://x", {})
    assert len(n) == 1
