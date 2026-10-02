import os

import requests


api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY environment variable is not set.")

response = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    timeout=30,
)
response.raise_for_status()

for model in response.json().get("data", []):
    print(model.get("id", ""))
