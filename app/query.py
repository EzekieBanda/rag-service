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


def _find_policy_chunk() -> str:
    """Search indexed chunks for the greetings/identity policy marker.
    Return the chunk text if found, else empty string.
    """
    if _chunk_mapping is None:
        return ""
    markers = [
        "SYSTEM IDENTITY AND GREETINGS POLICY",
        "NBS Assistant",
        "GREETINGS AND BASIC INTERACTIONS",
        "IDENTITY",
        "CAPABILITIES",
    ]
    for c in _chunk_mapping:
        txt = c.get("text", "")
        up = txt.upper()
        if any(m in up for m in markers):
            return txt
    return ""


def _is_greeting_intent(question: str) -> bool:
    q = question.strip().lower()
    greetings = {"hi", "hello", "hey", "hi!", "hello!", "hey!"}
    if q in greetings:
        return True
    # short greeting phrases
    if any(q.startswith(g + " ") for g in greetings):
        return True
    return False


def _is_identity_intent(question: str) -> bool:
    q = question.strip().lower()
    patterns = ["who are you", "what is your name", "what are you called", "your name"]
    return any(p in q for p in patterns)


def _is_capabilities_intent(question: str) -> bool:
    q = question.strip().lower()
    patterns = ["what are you good at", "what can you do", "capabilities", "what can you do for me"]
    return any(p in q for p in patterns)


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

    # First: detect greeting/identity/capability intents and honor policy file if present
    policy_chunk = _find_policy_chunk()
    if policy_chunk:
        if _is_greeting_intent(question):
            return "Hello! How can I assist you today?"
        if _is_identity_intent(question):
            return "I am NBS Assistant, your helpful AI companion."
        if _is_capabilities_intent(question):
            return ("I can answer questions based on available documents, explain concepts clearly, "
                    "and assist with technical, professional, and general knowledge questions.")

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
