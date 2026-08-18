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
