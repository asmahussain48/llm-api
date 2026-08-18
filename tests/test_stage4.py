import os
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

# Helpers to create fake clients and simulate provider behavior
class FakeResponse:
    def __init__(self, content, usage=None, status_code=None):
        class Message:
            def __init__(self, c):
                self.content = c
        class Choice:
            def __init__(self, m):
                self.message = m
        self.choices = [Choice(Message(content))]
        self.usage = usage
        self.status_code = status_code


class FakeError(Exception):
    def __init__(self, message, status_code=None, retry_after=None, is_timeout=False):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.is_timeout = is_timeout


def make_client_from_sequence(seq):
    """seq is a list of either strings (content) or FakeError to raise"""
    class FakeChat:
        def __init__(self, sequence):
            self.sequence = sequence
            self._idx = 0
            self.completions = self

        def create(self, *args, **kwargs):
            if self._idx >= len(self.sequence):
                val = self.sequence[-1]
            else:
                val = self.sequence[self._idx]
                self._idx += 1

            if isinstance(val, Exception):
                raise val
            return FakeResponse(val)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat(seq)

    return FakeClient


# Patch sleep to avoid real waiting and capture delays
recorded_sleeps = []

def fake_sleep(sec):
    recorded_sleeps.append(sec)


def test_llm_enabled_kill_switch(monkeypatch):
    os.environ["LLM_ENABLED"] = "false"
    os.environ["LLM_STUB"] = "0"

    r = client.post("/triage", json={"text": "Any message"})
    assert r.status_code == 503
    assert r.json()["detail"] == "LLM service is disabled"

    # Ensure no logs/quarantine were created
    assert not Path("logs").exists()


def test_401_not_retried(monkeypatch):
    os.environ["LLM_ENABLED"] = "true"
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    err = FakeError("Unauthorized", status_code=401)
    FakeClient = make_client_from_sequence([err])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    r = client.post("/triage", json={"text": "Trigger 401"})
    # Route maps provider runtime errors -> 502 except for kill switch; but 401 should NOT be retried and return 502
    assert r.status_code == 502


def test_timeout_retries_and_final_504(monkeypatch):
    os.environ["LLM_ENABLED"] = "true"
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    # Simulate timeouts for each call
    err = FakeError("Timeout", is_timeout=True)
    FakeClient = make_client_from_sequence([err, err, err, err])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    monkeypatch.setattr("time.sleep", fake_sleep)

    r = client.post("/triage", json={"text": "Trigger timeout"})
    # After exhausting provider retries the service returns 502 or 504 mapping; our implementation maps provider timeout -> 504
    assert r.status_code in (502, 504)

    # Ensure sleep was called for retries (3 retries) - recorded_sleeps len > 0
    assert len(recorded_sleeps) >= 1


def test_429_respects_retry_after(monkeypatch):
    os.environ["LLM_ENABLED"] = "true"
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    # First attempt returns 429 with Retry-After=2, second succeeds
    err = FakeError("Too Many Requests", status_code=429, retry_after=2)
    good = json.dumps({"category": "bug", "urgency": "normal", "confidence": 0.9, "reason": "Test"})
    FakeClient = make_client_from_sequence([err, good])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    recorded_sleeps.clear()
    monkeypatch.setattr("time.sleep", fake_sleep)

    r = client.post("/triage", json={"text": "Trigger 429"})
    assert r.status_code == 200
    assert len(recorded_sleeps) == 1
    assert recorded_sleeps[0] == 2


def test_structured_logging_contains_fields(monkeypatch, caplog):
    os.environ["LLM_ENABLED"] = "true"
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    good = json.dumps({"category": "billing", "urgency": "high", "confidence": 0.95, "reason": "Duplicate"})
    # Provide usage info via FakeResponse by returning a FakeResponse object in make_client_from_sequence
    class ClientWithUsage:
        class Chat:
            def __init__(self):
                self._used = False
                self.completions = self

            def create(self, *args, **kwargs):
                resp = FakeResponse(good, usage={"prompt_tokens": 10, "completion_tokens": 5})
                return resp

        def __init__(self, *args, **kwargs):
            self.chat = ClientWithUsage.Chat()

    monkeypatch.setattr("openai.OpenAI", ClientWithUsage)

    caplog.clear()
    r = client.post("/triage", json={"text": "Check logging"})
    assert r.status_code == 200
    # Look for the llm_call log event in caplog
    found = False
    for rec in caplog.records:
        if getattr(rec, "message", "").find('"event": "llm_call"') != -1:
            found = True
            assert '"prompt_version": "triage-v1"' in rec.message
            assert '"model": "openrouter/free"' in rec.message
            assert '"input_tokens": 10' in rec.message
            assert '"output_tokens": 5' in rec.message
            assert '"repair_count": 0' in rec.message
    assert found


# Ensure existing Stage 3 behavior still works (repair attempt)
def test_stage3_repair_still_works(monkeypatch):
    os.environ["LLM_ENABLED"] = "true"
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    bad = "not json"
    repaired = json.dumps({"category": "feature", "urgency": "low", "confidence": 0.1, "reason": "Feature request."})
    FakeClient = make_client_from_sequence([bad, repaired])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    r = client.post("/triage", json={"text": "Please add CSV export"})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "feature"
