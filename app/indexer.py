import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from docx import Document as DocxDocument
import PyPDF2

DATA_DIR = os.getenv("DATA_PATH", "/data")
INDEX_DIR = os.getenv("INDEX_PATH", "/index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # tokens or approx characters

index = None
model = SentenceTransformer(EMBEDDING_MODEL)
chunk_mapping_file = os.path.join(INDEX_DIR, "chunk_mapping.pkl")

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

def build_index():
    global index
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load existing index and mapping
    index_file = os.path.join(INDEX_DIR, "faiss.index")
    if os.path.exists(index_file) and os.path.exists(chunk_mapping_file):
        index = faiss.read_index(index_file)
        with open(chunk_mapping_file, "rb") as f:
            chunk_mapping = pickle.load(f)
        print(f"Loaded existing index with {len(chunk_mapping)} chunks")
        return

    # Collect all text chunks
    texts = []
    chunk_mapping = []
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            ext = f.lower().split('.')[-1]
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
                chunk_mapping.extend([{"file": f, "text": c} for c in chunks])
            except Exception as e:
                print(f"Failed to read {path}: {e}")

    # Generate embeddings
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and mapping
    faiss.write_index(index, index_file)
    with open(chunk_mapping_file, "wb") as f:
        pickle.dump(chunk_mapping, f)

    print(f"Indexed {len(texts)} chunks from {len(set([c['file'] for c in chunk_mapping]))} files")
