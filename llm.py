"""Local dev LLM wrapper.  The autograder REPLACES this file entirely."""

import json
import os
import time
import urllib.request

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("RAG_MODEL", "meta-llama/llama-3.1-8b-instruct")
TIMEOUT = 30
MAX_RETRIES = 3


def llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY env var to use the LLM locally."
        )

    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.0,
    }).encode()

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            raise
