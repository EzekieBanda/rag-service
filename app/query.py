import os
import requests
from embeddings import embed
from indexer import load_index

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

def query_llm(question):
    index, docs = load_index()
    q_vec = embed([question])

    _, idxs = index.search(q_vec, 3)
    context = "\n".join([docs[i] for i in idxs[0]])

    payload = {
        "model": MODEL,
        "prompt": f"Context:\n{context}\n\nQuestion:\n{question}",
        "stream": False
    }

    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    return r.json()["response"]
