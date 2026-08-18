import os
import json
import openai
from fastapi.testclient import TestClient

# Prepare environment
os.environ["LLM_STUB"] = "0"
os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["LLM_API_KEY"] = "fake"
os.environ["LLM_MODEL"] = "openrouter/free"

# Fake classes
class FakeMessage:
    def __init__(self, content):
        self.content = content

class FakeChoice:
    def __init__(self, message):
        self.message = message

class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(FakeMessage(content))]

fake_output = {
    "category": "billing",
    "urgency": "high",
    "confidence": 0.9,
    "reason": "Customer reports duplicate payment."
}

class FakeChat:
    def __init__(self):
        self.completions = self
    def create(self, *args, **kwargs):
        return FakeResponse(json.dumps(fake_output))

class FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = FakeChat()

# Monkeypatch openai.OpenAI
openai.OpenAI = FakeClient

from src.main import app
client = TestClient(app)

r = client.post("/triage", json={"text":"I was charged twice"})
print('status', r.status_code)
print('body', r.text)
print('json', r.json() if r.status_code==200 else r.json())
