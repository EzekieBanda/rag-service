import os
import pickle
import faiss
import httpx
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.getenv("INDEX_PATH", "/index")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:14b")
TOP_K = 5

# Lazy-loaded globals
_index = None
_chunk_mapping = None
_embedder = None


def load_index():
    global _index, _chunk_mapping

    if _index is not None and _chunk_mapping is not None:
        return

    index_file = os.path.join(INDEX_DIR, "faiss.index")
    mapping_file = os.path.join(INDEX_DIR, "chunk_mapping.pkl")

    if not os.path.exists(index_file) or not os.path.exists(mapping_file):
        raise RuntimeError("FAISS index not found. Call /index first.")

    _index = faiss.read_index(index_file)

    with open(mapping_file, "rb") as f:
        _chunk_mapping = pickle.load(f)


def load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


async def query_llm(question: str) -> str:
    """
    1️⃣ Embed question
    2️⃣ Retrieve top-k chunks from FAISS
    3️⃣ Build RAG prompt
    4️⃣ Call Ollama
    """
    load_index()
    load_embedder()

    # Embed question
    q_embedding = _embedder.encode([question])

    # FAISS search
    distances, indices = _index.search(q_embedding, TOP_K)

    context_chunks = [
        _chunk_mapping[i]["text"]
        for i in indices[0]
        if i < len(_chunk_mapping)
    ]

    context_text = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{question}

Answer:
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=5000) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

    return data.get("response", "")
