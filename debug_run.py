import os, json
from src.llm.service import triage_support_message

os.environ["LLM_STUB"] = "0"
os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["LLM_API_KEY"] = "fake"
os.environ["LLM_MODEL"] = "openrouter/free"

outputs = ["This is not JSON.", json.dumps({"category":"feature","urgency":"low","confidence":0.1,"reason":"Feature request."})]

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
        self.completions = self

    def create(self, *args, **kwargs):
        if self._idx >= len(self.outputs):
            return FakeResponse(self.outputs[-1])
        out = self.outputs[self._idx]
        self._idx += 1
        return FakeResponse(out)

class FakeClient:
    def __init__(self, *args, **kwargs):
        self.chat = FakeChat(outputs)

# monkeypatch in runtime
import openai
openai.OpenAI = FakeClient

print('Calling triage_support_message')
res = triage_support_message('Please add CSV export')
print('Result:', res)
