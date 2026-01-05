from fastapi import FastAPI
from pydantic import BaseModel
from app.query import query_llm
from app.indexer import build_index

app = FastAPI(title="RAG FAISS Service")

class QueryRequest(BaseModel):
    question: str


@app.post("/index")
def index_documents():
    build_index()
    return {"status": "indexed"}


@app.post("/query")
async def query(req: QueryRequest):
    answer = await query_llm(req.question)
    return {"answer": answer}
