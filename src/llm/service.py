import os
import json
from typing import Dict
from pathlib import Path

from src.llm.schema import TriageOutput


def triage_support_message(text: str) -> TriageOutput:
    """Service that either returns a deterministic stub (LLM_STUB=1) or calls the real LLM.

    When LLM_STUB=1 (default), return a fixed deterministic response and do not call
    any external LLM provider. When LLM_STUB=0, call the configured OpenRouter model
    via the OpenAI-compatible client.
    """
    stub_mode = os.environ.get("LLM_STUB", "1") == "1"

    if stub_mode:
        # Deterministic stub output
        return TriageOutput(
            category="other",
            urgency="normal",
            confidence=0.5,
            reason="Stub response for development.",
        )

    # Real LLM path
    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")

    if not base_url or not api_key or not model:
        raise RuntimeError("LLM configuration missing (ensure LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL are set)")

    # Load prompt from file
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "triage-v1.md"
    if not prompt_path.exists():
        raise RuntimeError("Prompt file not found: triage-v1.md")

    prompt_text = prompt_path.read_text(encoding="utf-8")

    # Lazy import of OpenAI client so tests (stub mode) don't require it/mocking
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI client library not available")

    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
    except Exception as e:
        # Surface a generic provider error without leaking secrets
        raise RuntimeError("LLM provider request failed")

    # Extract content
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        raise RuntimeError("Unexpected LLM response shape")

    # Parse JSON
    try:
        parsed = json.loads(content)
    except Exception:
        raise RuntimeError("LLM did not return valid JSON")

    # Validate with Pydantic
    try:
        result = TriageOutput(**parsed)
    except Exception as e:
        raise RuntimeError("LLM returned JSON that does not conform to schema")

    return result
