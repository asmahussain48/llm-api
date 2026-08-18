import os
import pytest
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
