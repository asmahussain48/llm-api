import os
import pytest
import json
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_valid_triage_request_returns_200():
    os.environ["LLM_STUB"] = "1"
    r = client.post("/triage", json={"text": "My payment was charged twice."})
    assert r.status_code == 200
    data = r.json()
    assert "category" in data
    assert "urgency" in data
    assert "confidence" in data
    assert "reason" in data


def test_response_fields_match_allowed_values():
    os.environ["LLM_STUB"] = "1"
    r = client.post("/triage", json={"text": "Something happened."})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] in ["billing", "bug", "feature", "other"]
    assert data["urgency"] in ["low", "normal", "high"]
    assert isinstance(data["confidence"], float) or isinstance(data["confidence"], int)
    assert 0.0 <= float(data["confidence"]) <= 1.0


def test_missing_text_rejected():
    r = client.post("/triage", json={})
    assert r.status_code == 422


def test_empty_text_rejected():
    r = client.post("/triage", json={"text": ""})
    assert r.status_code == 422


def test_whitespace_only_rejected():
    r = client.post("/triage", json={"text": "   "})
    assert r.status_code == 422


def test_text_over_2000_rejected():
    long_text = "x" * 2001
    r = client.post("/triage", json={"text": long_text})
    assert r.status_code == 422


def test_stub_mode_does_not_call_real_llm(monkeypatch):
    # Ensure LLM_STUB=1 so service uses stub and does not attempt a real LLM call.
    os.environ["LLM_STUB"] = "1"

    # Monkeypatch a hypothetical OpenAI constructor to raise if called.
    def fail_on_init(*args, **kwargs):
        raise AssertionError("OpenAI client should not be instantiated in stub mode")

    monkeypatch.setattr("openai.OpenAI", fail_on_init, raising=False)

    r = client.post("/triage", json={"text": "Please help."})
    assert r.status_code == 200
    data = r.json()
    assert data["reason"] == "Stub response for development."


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
    os.environ["LLM_STUB"] = "1"  # temporarily set to stub for isolation
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

# Invalid category
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

# Both original and repair outputs invalid
bad1 = "not json"
bad2 = "still not json"

FakeClient = make_fake_client_from_outputs([bad1, bad2])
monkeypatch.setattr("openai.OpenAI", FakeClient)

# Ensure logs directory is in tmp and point to it by changing cwd? Instead, we will read logs/quarantine.jsonl
# Remove existing quarantine file if present
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


def test_real_llm_path_is_mockable(monkeypatch):
    # Ensure we test the real LLM path without making network calls.
    os.environ["LLM_STUB"] = "0"
    os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = "fake"
    os.environ["LLM_MODEL"] = "openrouter/free"

    # Prepare a fake response object structure similar to OpenAI client
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(FakeMessage(content))]

    # The model should return valid JSON matching the schema
    fake_output = {
        "category": "billing",
        "urgency": "high",
        "confidence": 0.9,
        "reason": "Customer reports duplicate payment."
    }

    def fake_create(*args, **kwargs):
        return FakeResponse(json.dumps(fake_output))

    class FakeChat:
        def __init__(self):
            self.completions = self
        def create(self, *args, **kwargs):
            return fake_create(*args, **kwargs)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    # Monkeypatch the OpenAI client used in the service
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    r = client.post("/triage", json={"text": "I was charged twice"})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "billing"
    assert data["urgency"] == "high"
    assert float(data["confidence"]) == 0.9
    assert "duplicate payment" in data["reason"]
