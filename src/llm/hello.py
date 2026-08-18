import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

base_url = os.environ["LLM_BASE_URL"]
api_key = os.environ["LLM_API_KEY"]
model = os.environ["LLM_MODEL"]

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "Reply with exactly the word: ready"},
    ],
)

content = response.choices[0].message.content or ""
print(content.strip())
