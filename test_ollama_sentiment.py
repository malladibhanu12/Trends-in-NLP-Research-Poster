"""
Quick test script: sends ONE sentiment-classification prompt to your local
Qwen2.5:1.5b model via Ollama and prints the result.

Run this first to confirm everything works before running the full experiment.

Requirements:
    pip install requests
    (Ollama must already be installed and the qwen2.5:1.5b model pulled --
     you already have this based on `ollama list`)

How to run:
    1. Open a terminal (PowerShell / cmd)
    2. Make sure Ollama is running (it usually starts automatically, but if
       not, run: ollama serve   in a separate terminal window)
    3. Run: python test_ollama_sentiment.py
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

# A single test sentence to classify
test_text = "The movie was okay, nothing special but not bad either."

prompt = f"""Classify the sentiment of the following text as exactly one word: Positive, Negative, or Neutral.

Text: "{test_text}"

Sentiment:"""

print(f"Sending prompt to {MODEL}...")
print(f"Text: {test_text}\n")

start = time.time()

try:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,  # generous timeout since CPU inference can be slow
    )
    response.raise_for_status()
    result = response.json()

    elapsed = time.time() - start
    print(f"Response received in {elapsed:.1f} seconds")
    print(f"Model output: {result['response'].strip()}")

except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to Ollama.")
    print("Make sure Ollama is running. Try opening a new terminal and running: ollama serve")

except requests.exceptions.Timeout:
    print("ERROR: Request timed out after 120 seconds.")
    print("The model may be too slow on this machine, or something is stuck.")
    print("Try closing other applications to free up RAM, then run this again.")

except Exception as e:
    print(f"ERROR: {e}")