import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")  # Use smaller model in prod CPU

async def query_llm(question: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": question,
        "max_tokens": 512
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("completion", "")
