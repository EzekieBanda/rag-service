import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import httpx

DATA_DIR = os.getenv("DATA_PATH", "/data")
INDEX_DIR = os.getenv("INDEX_PATH", "/index")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
TOP_K = 5  # Number of document chunks to retrieve

# Load FAISS index
index_file = os.path.join(INDEX_DIR, "faiss.index")
chunk_mapping_file = os.path.join(INDEX_DIR, "chunk_mapping.pkl")

index = faiss.read_index(index_file) if os.path.exists(index_file) else None
with open(chunk_mapping_file, "rb") as f:
    chunk_mapping = pickle.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def query_llm(question: str) -> str:
    """
    1️⃣ Embed the question
    2️⃣ Retrieve top-k relevant document chunks from FAISS
    3️⃣ Prepend them to the prompt
    4️⃣ Call Ollama for answer
    """
    if index is None:
        return "Index not built yet."

    # Encode question
    q_embedding = model.encode([question])

    # Search FAISS
    distances, indices = index.search(q_embedding, TOP_K)

    # Retrieve the corresponding chunks
    context_chunks = [chunk_mapping[i]["text"] for i in indices[0]]
    context_text = "\n".join(context_chunks)

    # Prepare prompt for Ollama
    prompt = f"Answer the following question using ONLY the context below.\n\nContext:\n{context_text}\n\nQuestion:\n{question}\nAnswer:"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": 512
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    return data.get("completion", "")
