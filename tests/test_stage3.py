import os
import json
from pathlib import Path
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def make_fake_client_from_outputs(outputs):
    """Helper: create a FakeClient that returns successive outputs for each create() call."""

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(FakeMessage(content))]

    class FakeChat:
        def __init__(self, outputs):
            self.outputs = outputs
            self._idx = 0

        def create(self, *args, **kwargs):
            if self._idx >= len(self.outputs):
                return FakeResponse(self.outputs[-1])
            out = self.outputs[self._idx]
            self._idx += 1
            return FakeResponse(out)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat(outputs)

    return FakeClient


def test_valid_model_json_accepted(monkeypatch):
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    valid = json.dumps({
        "category": "billing",
        "urgency": "high",
        "confidence": 0.95,
        "reason": "Duplicate billing charge."
    })

    FakeClient = make_fake_client_from_outputs([valid])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    r = client.post("/triage", json={"text": "Charged twice"})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "billing"


def test_invalid_json_triggers_repair_success(monkeypatch):
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    bad = "This is not JSON."
    repaired = json.dumps({"category": "feature", "urgency": "low", "confidence": 0.1, "reason": "Feature request."})

    FakeClient = make_fake_client_from_outputs([bad, repaired])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    r = client.post("/triage", json={"text": "Please add CSV export"})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "feature"


def test_invalid_schema_triggers_repair_success(monkeypatch):
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    bad_schema = json.dumps({"category": "complaint", "urgency": "high", "confidence": 0.8, "reason": "Example."})
    repaired = json.dumps({"category": "other", "urgency": "normal", "confidence": 0.2, "reason": "Repaired."})

    FakeClient = make_fake_client_from_outputs([bad_schema, repaired])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    r = client.post("/triage", json={"text": "Bad category"})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "other"


def test_repair_failure_results_in_422_and_quarantine(monkeypatch, tmp_path):
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    bad1 = "not json"
    bad2 = "still not json"

    FakeClient = make_fake_client_from_outputs([bad1, bad2])
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    qpath = Path("logs") / "quarantine.jsonl"
    if qpath.exists():
        qpath.unlink()

    r = client.post("/triage", json={"text": "Trigger failure"})
    assert r.status_code == 422

    assert qpath.exists()
    lines = qpath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["endpoint"] == "/triage"
    assert "LLM_API_KEY" not in ''.join(lines)


def test_maximum_two_model_calls_on_failure(monkeypatch):
    # Ensure that on failure we call the model no more than twice (original + one repair)
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    calls = {"count": 0}

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(FakeMessage(content))]

    def fake_create_bad(*args, **kwargs):
        calls["count"] += 1
        return FakeResponse("not json")

    class FakeChat:
        def __init__(self):
            self.completions = self

        def create(self, *args, **kwargs):
            return fake_create_bad(*args, **kwargs)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    qpath = Path("logs") / "quarantine.jsonl"
    if qpath.exists():
        qpath.unlink()

    r = client.post("/triage", json={"text": "Trigger failure"})
    assert r.status_code == 422
    # Should be exactly 2 calls: original + repair
    assert calls["count"] == 2


def test_stub_mode_still_works_after_changes(monkeypatch):
    os.environ["LLM_STUB"] = "1"
    r = client.post("/triage", json={"text": "Is stub still on?"})
    assert r.status_code == 200
    data = r.json()
    assert data["reason"] == "Stub response for development."
