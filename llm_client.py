"""
Thin wrapper around the local Ollama API for calling qwen2.5:1.5b.
Includes a timeout and retry so a single slow/stuck call doesn't kill the
whole batch run.
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"


def call_llm(prompt: str, temperature: float = 0.7, timeout: int = 60, max_retries: int = 2) -> str:
    """
    Sends a prompt to the local Ollama model and returns the raw text response.
    Retries on timeout/connection errors up to max_retries times.
    Raises the last exception if all retries fail.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["response"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            print(f"  [retry {attempt + 1}/{max_retries}] LLM call failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")
