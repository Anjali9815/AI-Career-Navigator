# main.py
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THIS FILE DOES:
#   Wraps chain.py + retriever.py into a FastAPI web server
#   Exposes HTTP endpoints so any frontend can talk to your RAG pipeline
#
# ENDPOINTS:
#   GET  /                    → health check
#   POST /chat                → ask a question (single turn)
#   POST /chat/stream         → ask a question (streaming response)
#   POST /search              → search without LLM (raw chunks only)
#   POST /upload              → add new PDF at runtime
#   GET  /sources             → list all loaded profile files
#   DELETE /reset             → wipe and rebuild ChromaDB
#
# HOW FASTAPI WORKS:
#   You define a function → decorate with @app.get() or @app.post()
#   FastAPI auto-generates docs at http://localhost:8000/docs
#   Input/output types defined with Pydantic models (type safety)
# ─────────────────────────────────────────────────────────────────────────────

import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from retriever import load_vectorstore, retrieve_all, retrieve_from_resumes
from chain import run_chain, run_chat_chain, get_llm, SYSTEM_PROMPT, build_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ── CONFIG ────────────────────────────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR   = os.path.join(BASE_DIR, "data/docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
# These define the shape of request/response JSON
# FastAPI uses these for validation + auto docs
#
# Request model = what the client SENDS to us
# Response model = what we SEND BACK to the client

class ChatRequest(BaseModel):
    """Request body for /chat endpoint"""
    query: str                          # the question
    mode: Optional[str] = "resume"     # "resume", "job", "match", "all"
    session_id: Optional[str] = None   # for multi-turn (future use)

class ChatResponse(BaseModel):
    """Response from /chat endpoint"""
    answer: str
    sources: list[str]
    mode: str
    query: str

class SearchRequest(BaseModel):
    """Request body for /search endpoint"""
    query: str
    doc_type: Optional[str] = None     # filter by type
    k: Optional[int] = 4              # how many chunks

class SearchResponse(BaseModel):
    """Response from /search endpoint"""
    chunks: list[dict]
    count: int
    query: str

# ── APP LIFESPAN ──────────────────────────────────────────────────────────────
# Lifespan = code that runs when app STARTS and STOPS
#
# WHY: Loading ChromaDB + embedding model takes ~3 seconds
#      We load ONCE at startup, reuse for every request
#      Without this: every request reloads everything = 3s per request
#      With this: startup takes 3s, every request takes ~50ms
#
# app.state = global storage for things shared across requests
# Think of it like a global variable but cleaner

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ──
    print("Starting AI Career Navigator API...")
    print("Loading ChromaDB + embedding model...")

    try:
        app.state.vectorstore = load_vectorstore()
        print("Ready! Vectorstore loaded.")
    except FileNotFoundError:
        print("WARNING: ChromaDB not found. Run ingest.py first.")
        print("         /chat will return errors until you do.")
        app.state.vectorstore = None

    # App runs here (between yield)
    yield

    # ── SHUTDOWN ──
    print("Shutting down...")


# ── CREATE APP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Career Navigator",
    description="RAG-powered career analysis using LinkedIn profiles",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS MIDDLEWARE ───────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this: browser blocks requests from frontend to backend
# With this: frontend at localhost:3000 can call backend at localhost:8000
#
# allow_origins=["*"] = allow ALL origins (fine for development)
# In production: restrict to your actual frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── HEALTH CHECK ─────────────────────────────────────────────────────────────
# Simple endpoint to confirm the server is running
# Call: GET http://localhost:8000/
# Returns: {"status": "ok", "vectorstore": "loaded"}

@app.get("/")
def health_check():
    vs_status = "loaded" if app.state.vectorstore else "not loaded — run ingest.py"
    return {
        "status"     : "ok",
        "app"        : "AI Career Navigator",
        "vectorstore": vs_status,
        "docs"       : "http://localhost:8000/docs"
    }


# ── CHAT ENDPOINT ─────────────────────────────────────────────────────────────
# Main endpoint — takes a question, returns Claude's answer
#
# Call: POST http://localhost:8000/chat
# Body: {"query": "Who has Python experience?", "mode": "resume"}
#
# What happens inside:
#   1. Validate request (Pydantic)
#   2. Check vectorstore is loaded
#   3. Call run_chain() from chain.py
#   4. Return answer + sources as JSON

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Ask a question about the loaded profiles.

    mode options:
    - "resume"  → search only resume/profile documents
    - "job"     → search only job descriptions
    - "match"   → compare resumes against job descriptions
    - "all"     → search everything
    """
    # Guard: vectorstore must be loaded
    if not app.state.vectorstore:
        raise HTTPException(
            status_code=503,
            detail="Vectorstore not loaded. Run ingest.py first then restart the server."
        )

    # Guard: query can't be empty
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Run the full RAG chain
        result = run_chain(
            query=request.query,
            vectorstore=app.state.vectorstore,
            mode=request.mode
        )

        return ChatResponse(
            answer  = result["answer"],
            sources = result["sources"],
            mode    = request.mode,
            query   = request.query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── STREAMING CHAT ────────────────────────────────────────────────────────────
# Same as /chat but streams tokens as they arrive
# The frontend receives text word-by-word instead of waiting for full response
#
# This uses Python generators — yield sends each chunk to the client
# immediately instead of buffering the full response
#
# Try it: POST http://localhost:8000/chat/stream
# You'll see text appear progressively in the UI

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming version of /chat.
    Returns text/event-stream — tokens arrive as they're generated.
    """
    if not app.state.vectorstore:
        raise HTTPException(status_code=503, detail="Vectorstore not loaded")

    # Retrieve context (non-streaming)
    from retriever import retrieve_from_resumes, retrieve_all, format_context
    if request.mode == "resume":
        results, context = retrieve_from_resumes(request.query, app.state.vectorstore)
    else:
        results, context = retrieve_all(request.query, app.state.vectorstore)

    if context == "No relevant documents found.":
        async def no_results():
            yield "I couldn't find relevant information in the loaded profiles."
        return StreamingResponse(no_results(), media_type="text/plain")

    # Build prompt
    prompt = build_prompt(request.query, context)

    # Stream from Gemini
    # Note: LangChain's .stream() yields chunks as they arrive
    async def generate():
        llm = get_llm()
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain")


# ── SEARCH ENDPOINT ───────────────────────────────────────────────────────────
# Returns raw chunks WITHOUT calling the LLM
# Useful for debugging: see exactly what documents would be retrieved
# Also useful for a "browse profiles" feature in the UI
#
# Call: POST http://localhost:8000/search
# Body: {"query": "AWS experience", "doc_type": "resume", "k": 4}

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Search documents without LLM — returns raw chunks.
    Useful for debugging retrieval quality.
    """
    if not app.state.vectorstore:
        raise HTTPException(status_code=503, detail="Vectorstore not loaded")

    from retriever import retrieve
    results = retrieve(
        query=request.query,
        vectorstore=app.state.vectorstore,
        doc_type=request.doc_type,
        k=request.k
    )

    return SearchResponse(
        chunks=results,
        count=len(results),
        query=request.query
    )


# ── UPLOAD ENDPOINT ───────────────────────────────────────────────────────────
# Add new PDF documents at runtime WITHOUT restarting the server
#
# What happens:
#   1. Save PDF to data/docs/
#   2. Re-run the full ingestion pipeline
#   3. Reload vectorstore into app.state
#   4. New document is now searchable immediately
#
# Call: POST http://localhost:8000/upload
# Body: multipart/form-data with file field

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a new PDF to the knowledge base.
    Triggers re-ingestion — new doc is searchable immediately.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    # Save file
    save_path = Path(DOCS_DIR) / file.filename
    Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    print(f"Saved: {save_path}")

    # Re-run ingestion
    try:
        import subprocess
        result = subprocess.run(
            ["python", "backend/ingest.py"],
            capture_output=True, text=True
        )
        print(result.stdout)

        # Reload vectorstore
        app.state.vectorstore = load_vectorstore()

        return {
            "message"  : f"Uploaded and indexed: {file.filename}",
            "filename" : file.filename,
            "status"   : "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── SOURCES ENDPOINT ──────────────────────────────────────────────────────────
# List all documents currently in the knowledge base
# Useful for the UI to show "X profiles loaded"
#
# Call: GET http://localhost:8000/sources

@app.get("/sources")
def list_sources():
    """List all documents currently indexed in ChromaDB."""
    if not app.state.vectorstore:
        raise HTTPException(status_code=503, detail="Vectorstore not loaded")

    # Get all metadata from ChromaDB
    data = app.state.vectorstore._collection.get()
    metadatas = data.get("metadatas", [])

    # Extract unique filenames + doc types
    seen = {}
    for m in metadatas:
        fname = m.get("filename", "unknown")
        if fname not in seen:
            seen[fname] = {
                "filename" : fname,
                "doc_type" : m.get("doc_type", "general"),
                "source"   : m.get("source", "")
            }

    sources = list(seen.values())

    return {
        "total_documents" : len(sources),
        "total_chunks"    : len(metadatas),
        "documents"       : sources
    }


# ── RESET ENDPOINT ────────────────────────────────────────────────────────────
# Wipe ChromaDB and rebuild from scratch
# Useful when you add/remove documents and want a clean rebuild
#
# Call: DELETE http://localhost:8000/reset

@app.delete("/reset")
def reset_vectorstore():
    """Wipe ChromaDB and rebuild from scratch."""
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
            print("ChromaDB wiped")

        import subprocess
        subprocess.run(["python", "backend/ingest.py"], check=True)

        app.state.vectorstore = load_vectorstore()

        return {"message": "Vectorstore rebuilt successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── RUN SERVER ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # 0.0.0.0 = accessible from any device on network
        port=8000,
        reload=True        # auto-restart when you save changes to code
    )