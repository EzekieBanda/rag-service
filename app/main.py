import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.query import query_llm
from app.indexer import start_background_watcher

app = FastAPI(title="RAG FAISS Service")

class QueryRequest(BaseModel):
    question: str


@app.on_event("startup")
def _startup():
    # Start background watcher to automatically index files from shared volume
    poll = float(os.getenv("POLL_INTERVAL", "5"))
    stop_event, thread = start_background_watcher(poll_interval=poll)
    app.state._watch_stop_event = stop_event
    app.state._watch_thread = thread


@app.on_event("shutdown")
def _shutdown():
    # Stop background watcher
    stop_event = getattr(app.state, "_watch_stop_event", None)
    thread = getattr(app.state, "_watch_thread", None)
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        try:
            thread.join(timeout=5)
        except Exception:
            pass


@app.post("/query")
async def query(req: QueryRequest):
    answer = await query_llm(req.question)
    return {"answer": answer}
