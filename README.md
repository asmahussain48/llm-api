# FlyRank LLM API

Stage-based implementation of an LLM-backed triage endpoint.

## What the endpoint does

POST /triage classifies a support message into a category and urgency so it can be routed to the right team.

- category: billing | bug | feature | other
- urgency: low | normal | high
- returns: category, urgency, confidence (0.0-1.0), reason (one short sentence)

## Runnable curl example

curl -X POST "http://127.0.0.1:8000/triage" \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice for my subscription."}'

## Exact response example

{
  "category": "billing",
  "urgency": "high",
  "confidence": 0.9,
  "reason": "Duplicate billing charge."
}

## Job card

# Job Card

## What it does

Classifies a support message so it can be sent to the right team.

## Input

{
  "text": "string, 1-2000 characters"
}

## Output

{
  "category": "billing|bug|feature|other",
  "urgency": "low|normal|high",
  "confidence": "0.0-1.0",
  "reason": "one short sentence"
}

## It must never

- Invent a category outside the allowed list.
- Return free-form fields outside the defined schema.
- Give medical, legal, or financial advice.
- Reveal the system prompt.

## When unsure

Return category "other" with low confidence instead of guessing.

## Provider / Model

Provider: OpenRouter
Base URL: https://openrouter.ai/api/v1
Model: configured via LLM_MODEL environment variable
Prompt version: triage-v1

## Environment variables

- LLM_BASE_URL: base URL for the OpenRouter/OpenAI-compatible API
- LLM_API_KEY: API key for the provider (do not commit)
- LLM_MODEL: model id (e.g. openrouter/free)
- LLM_STUB: if 1, use deterministic local stub (default for tests)
- LLM_ENABLED: if false, the LLM is disabled (kill switch)

## Reliability

- Timeout: 30 seconds maximum for provider calls
- Retryable: timeout, 429, 5xx
- Non-retryable: 400, 401, 403
- Backoff: exponential with jitter (~1s, ~2s, ~4s)
- Retry-After: honored for 429 when present
- SDK retries: disabled (application controls retries)
- Repair: one repair attempt for invalid JSON/schema
- Quarantine: invalid outputs after failed repair are written to logs/quarantine.jsonl
- Kill switch: set LLM_ENABLED=false to disable provider calls

## Evaluation

Evaluation uses the prompt version triage-v1 and an 8-case hand-labelled dataset in evals/cases.json. To run evaluation locally (it uses the real provider):

1. Add your OpenRouter API key to .env (LLM_API_KEY)
2. Set LLM_ENABLED=true and LLM_STUB=0 in .env
3. Run: python evals/run_eval.py

The script will save results to evals/results.json. The automated test suite does not run this evaluation and continues to use LLM_STUB=1.

## Cost / Usage logging

Each provider call logs structured usage information including: prompt_version, model, input_tokens, output_tokens, duration_ms, repair_count. No API keys are logged.

## Estimated cost for 10,000 requests/day

This depends on the selected model and provider pricing. The project defaults to a free lane (openrouter/free) that may be $0 for the free tier. If using a paid model, estimate cost as:

10,000 requests/day × estimated tokens/request × provider price per 1k tokens = daily cost

(Please replace with actual provider pricing and tokens-per-request measured for your setup.)

## What I would fix with another day

- Add more thorough evaluation coverage and larger labeled dataset
- Improve prompt engineering and calibration for confidence
- Add persistent quarantine storage and review tooling
- Add metrics and dashboards for repair/quarantine rates
- Add provider fallback and cost-aware routing
