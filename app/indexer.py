import os
import faiss
import pickle
import hashlib
import threading
import time
from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer
from docx import Document as DocxDocument
import PyPDF2

DATA_DIR = os.getenv("DATA_PATH", "/data")
INDEX_DIR = os.getenv("INDEX_PATH", "/index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # tokens or approx characters
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "5"))

index = None
model = SentenceTransformer(EMBEDDING_MODEL)
chunk_mapping_file = os.path.join(INDEX_DIR, "chunk_mapping.pkl")
indexed_files_file = os.path.join(INDEX_DIR, "indexed_files.pkl")

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx_file(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_pdf_file(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(filter(None, text))

def chunk_text(text: str, size: int = CHUNK_SIZE):
    """Split text into fixed-size chunks."""
    return [text[i:i+size] for i in range(0, len(text), size)]

#Hash method to get file hash


def build_index():
    global index
    os.makedirs(INDEX_DIR, exist_ok=True)
    # Rebuild the entire index from all files
    index_file = os.path.join(INDEX_DIR, "faiss.index")

    texts = []
    chunk_mapping = []
    indexed_files = {}

    for fn in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(path):
            continue
        ext = fn.lower().split('.')[-1]
        try:
            if ext == "txt":
                content = read_text_file(path)
            elif ext == "docx":
                content = read_docx_file(path)
            elif ext == "pdf":
                content = read_pdf_file(path)
            else:
                continue
            chunks = chunk_text(content)
            texts.extend(chunks)
            chunk_mapping.extend([{"file": fn, "text": c} for c in chunks])
            indexed_files[fn] = get_file_hash(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")

    if texts:
        embeddings = model.encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, index_file)
        with open(chunk_mapping_file, "wb") as f:
            pickle.dump(chunk_mapping, f)
        save_indexed_files(indexed_files)

    print(f"Indexed {len(texts)} chunks from {len(set([c['file'] for c in chunk_mapping]))} files")


def index_new_files() -> Dict[str, int]:
    """Incrementally index new or modified files in DATA_DIR."""
    global index
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_file = os.path.join(INDEX_DIR, "faiss.index")

    indexed_files = load_indexed_files()
    chunk_mapping = []
    if os.path.exists(index_file) and os.path.exists(chunk_mapping_file):
        index = faiss.read_index(index_file)
        with open(chunk_mapping_file, "rb") as f:
            chunk_mapping = pickle.load(f)
    else:
        index = None

    current_files = set()
    new_files = []
    updated_files = []

    for fn in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(path):
            continue
        ext = fn.lower().split('.')[-1]
        if ext not in ("txt", "docx", "pdf"):
            continue
        current_files.add(fn)
        h = get_file_hash(path)
        if fn not in indexed_files:
            new_files.append((fn, path, ext))
        elif indexed_files.get(fn) != h:
            updated_files.append((fn, path, ext))

    # detect deleted files
    deleted_files = set(indexed_files.keys()) - current_files
    for dfn in deleted_files:
        chunk_mapping = [c for c in chunk_mapping if c["file"] != dfn]
        indexed_files.pop(dfn, None)

    added_chunks = []
    for fn, path, ext in new_files + updated_files:
        try:
            if ext == "txt":
                content = read_text_file(path)
            elif ext == "docx":
                content = read_docx_file(path)
            elif ext == "pdf":
                content = read_pdf_file(path)
            else:
                continue
            chunks = chunk_text(content)
            if (fn, path, ext) in updated_files:
                chunk_mapping = [c for c in chunk_mapping if c["file"] != fn]
            added_chunks.extend([(c, fn) for c in chunks])
            indexed_files[fn] = get_file_hash(path)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    if added_chunks:
        texts = [c for c, _ in added_chunks]
        emb = model.encode(texts)
        if index is None:
            index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        chunk_mapping.extend([{"file": fn, "text": txt} for txt, fn in added_chunks])

    if index is not None:
        faiss.write_index(index, index_file)
        with open(chunk_mapping_file, "wb") as f:
            pickle.dump(chunk_mapping, f)

    save_indexed_files(indexed_files)

    result = {
        "new_files": len(new_files),
        "updated_files": len(updated_files),
        "deleted_files": len(deleted_files),
        "new_chunks": len(added_chunks),
        "total_chunks": len(chunk_mapping),
    }
    print(f"Indexed {len(added_chunks)} new chunks ({len(new_files)} new files, {len(updated_files)} updated). Total chunks: {len(chunk_mapping)}")
    return result


def _watch_loop(poll_interval: float, stop_event: threading.Event):
    """Background loop that polls DATA_DIR and calls index_new_files when changes are detected."""
    # A simple approach: call index_new_files at startup, then poll
    try:
        index_new_files()
    except Exception:
        pass
    while not stop_event.is_set():
        try:
            index_new_files()
        except Exception as e:
            print(f"Watcher error: {e}")
        stop_event.wait(poll_interval)


def start_background_watcher(poll_interval: float = POLL_INTERVAL) -> Tuple[threading.Event, threading.Thread]:
    """Start the watcher in a daemon thread. Returns (stop_event, thread)."""
    stop_event = threading.Event()
    t = threading.Thread(target=_watch_loop, args=(poll_interval, stop_event), daemon=True)
    t.start()
    return stop_event, t
