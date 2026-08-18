import os, json, openai
from src.llm import service

os.environ["LLM_STUB"] = "0"
os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["LLM_API_KEY"] = "fake"
os.environ["LLM_MODEL"] = "openrouter/free"

class FakeMessage:
    def __init__(self, content):
        self.content = content
class FakeChoice:
    def __init__(self, message):
        self.message = message
class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(FakeMessage(content))]

fake_output = {"category":"billing","urgency":"high","confidence":0.9,"reason":"duplicate"}

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

openai.OpenAI = FakeClient

try:
    res = service.triage_support_message('I was charged twice')
    print('Service returned:', res)
except Exception as e:
    print('Service raised:', type(e), e)
