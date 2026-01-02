import os
import faiss
from sentence_transformers import SentenceTransformer
from docx import Document as DocxDocument
import PyPDF2

DATA_DIR = os.getenv("DATA_PATH", "/data")
INDEX_DIR = os.getenv("INDEX_PATH", "/index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

index = None
model = SentenceTransformer(EMBEDDING_MODEL)

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

def build_index():
    global index
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_file = os.path.join(INDEX_DIR, "faiss.index")

    # Load existing index if exists
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
        return

    # Collect all texts from data folder
    texts = []
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            ext = f.lower().split('.')[-1]
            try:
                if ext == "txt":
                    texts.append(read_text_file(path))
                elif ext == "docx":
                    texts.append(read_docx_file(path))
                elif ext == "pdf":
                    texts.append(read_pdf_file(path))
            except Exception as e:
                print(f"Failed to read {path}: {e}")

    # Embed texts and create FAISS index
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)


    faiss.write_index(index, index_file)
    print(f"Indexed {len(texts)} documents")
