import os
import faiss
import pickle
from embeddings import embed

DATA_PATH = os.getenv("DATA_PATH", "/data")
INDEX_PATH = os.getenv("INDEX_PATH", "/index")

index_file = f"{INDEX_PATH}/faiss.index"
meta_file = f"{INDEX_PATH}/meta.pkl"

os.makedirs(INDEX_PATH, exist_ok=True)

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)
        if os.path.isfile(path):
            with open(path, "r", errors="ignore") as f:
                docs.append(f.read())
    return docs

def build_index():
    docs = load_documents()
    vectors = embed(docs)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, index_file)
    pickle.dump(docs, open(meta_file, "wb"))

def load_index():
    index = faiss.read_index(index_file)
    docs = pickle.load(open(meta_file, "rb"))
    return index, docs
