from fastapi import FastAPI
from app.indexer import build_index
from app.query import query_llm

app = FastAPI(title="RAG FAISS Service")

@app.post("/index")
def index_documents():
    build_index()
    return {"status": "indexed"}

@app.post("/query")
def query(question: str):
    answer = query_llm(question)
    return {"answer": answer}
