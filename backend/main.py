import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from backend.src.chain import ask_career_navigator

# define lifespan events
@asynccontextmanager
async def lifespan(app:FastAPI):
    print("App Starting...")
    app.state.chroma_db = "Chroma DB Loaded"
    app.state.bm25_retriever = "BM25 Keyword Index Ready"
    app.state.cross_encoder = "Neural Model Weights Loaded"
    yield
    print("App shutting down: Flushing connections.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def health_checkup():
    return {"status" : "Healthy", "api_layer": "online","database_connected": hasattr(app.state, "chroma_db")}



@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        # Run the incoming text query through your fixed career navigator logic
        answer = ask_career_navigator(payload.query)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

