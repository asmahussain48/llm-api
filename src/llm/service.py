import os
from typing import Dict

from src.llm.schema import TriageOutput


def triage_support_message(text: str) -> TriageOutput:
    """Deterministic stub service for Stage 1.

    When LLM_STUB=1 (default in development), return a fixed deterministic response
    and do not call any external LLM provider.
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

    # Placeholder for later: call real LLM client (not implemented in Stage 1)
    raise RuntimeError("Real LLM integration is disabled in Stage 1 (LLM_STUB not set).")
