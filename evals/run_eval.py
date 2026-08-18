import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Ensure project root is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.service import triage_support_message

EVAL_DIR = Path(__file__).resolve().parent
CASES_FILE = EVAL_DIR / "cases.json"
RESULTS_FILE = EVAL_DIR / "results.json"


def load_cases():
    with CASES_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_case(case):
    try:
        # Ensure LLM is enabled and not stubbed
        os.environ["LLM_ENABLED"] = "true"
        os.environ["LLM_STUB"] = "0"
        result = triage_support_message(case["text"])
        return {
            "id": case["id"],
            "text": case["text"],
            "expected_category": case["expected_category"],
            "expected_urgency": case["expected_urgency"],
            "category": result.category,
            "urgency": result.urgency,
            "confidence": float(result.confidence),
            "reason": result.reason,
            "error": None,
        }
    except Exception as e:
        return {
            "id": case.get("id"),
            "text": case.get("text"),
            "expected_category": case.get("expected_category"),
            "expected_urgency": case.get("expected_urgency"),
            "category": None,
            "urgency": None,
            "confidence": None,
            "reason": None,
            "error": str(e),
        }


def score_results(results):
    total = len(results)
    category_correct = 0
    urgency_correct = 0
    exact_match = 0
    for r in results:
        if r["error"] is not None:
            continue
        if r["category"] == r["expected_category"]:
            category_correct += 1
        if r["urgency"] == r["expected_urgency"]:
            urgency_correct += 1
        if r["category"] == r["expected_category"] and r["urgency"] == r["expected_urgency"]:
            exact_match += 1

    def pct(n):
        return round((n / total) * 100.0, 2) if total > 0 else 0.0

    return {
        "total": total,
        "category_correct": category_correct,
        "urgency_correct": urgency_correct,
        "exact_match": exact_match,
        "category_accuracy": pct(category_correct),
        "urgency_accuracy": pct(urgency_correct),
        "exact_match_accuracy": pct(exact_match),
    }


def main():
    if not CASES_FILE.exists():
        print("No cases.json found. Create evals/cases.json first.")
        return 1

    # Check that LLM_API_KEY is present before trying to call provider
    key = os.environ.get("LLM_API_KEY")
    if not key:
        print("LLM_API_KEY not set in environment. Aborting evaluation. Please put your key into .env and set LLM_ENABLED=true and LLM_STUB=0 before running.")
        return 2

    cases = load_cases()
    results = []
    for c in cases:
        print(f"Running case {c['id']}...")
        res = run_case(c)
        results.append(res)

    scores = score_results(results)

    out = {
        "evaluation_date": datetime.utcnow().isoformat() + "Z",
        "prompt_version": "triage-v1",
        "model": os.environ.get("LLM_MODEL"),
        "results": results,
        **scores,
    }

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Evaluation complete. Results saved to", RESULTS_FILE)
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
