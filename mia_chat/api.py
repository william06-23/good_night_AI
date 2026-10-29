"""
FastAPI server for Mia — load models once, chat via HTTP.

Run from repo root:
  python -m uvicorn mia_chat.api:app --host 127.0.0.1 --port 8000

Or from mia_chat/ with PYTHONPATH=..
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chat_log import load_messages
from .engine import MiaEngine
from .memory import list_speakable_characters, load_memory, public_memory_store

engine = MiaEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Mia API up. Waiting for a model choice (7b or 14b) from the UI.")
    yield
    print("Mia API shutting down.")


app = FastAPI(title="Good Night AI — Mia", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    character: str | None = None


class ModelChoice(BaseModel):
    key: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
    recalled: list[dict[str, Any]]
    ended: bool
    facts_count: int
    session_id: str
    character: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return engine.status()


@app.post("/api/model")
def choose_model(body: ModelChoice) -> dict[str, Any]:
    try:
        engine.begin_load(body.key)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return engine.status()


@app.post("/api/model/retry")
def retry_model() -> dict[str, Any]:
    engine.clear_error()
    return engine.status()


@app.get("/api/memory")
def get_memory() -> dict[str, Any]:
    """Character-keyed string lists (Narrator / Player / NPCs)."""
    store = engine.get_memory()
    public = public_memory_store(store)
    if any(public.values()):
        return public
    return public_memory_store(load_memory())


@app.get("/api/characters")
def get_characters() -> dict[str, Any]:
    return {
        "characters": list_speakable_characters(),
        "active": engine.active_character,
    }


@app.get("/api/chat/history")
def chat_history() -> dict[str, Any]:
    """Past transcript for the UI only — not fed back into the model."""
    return {"messages": load_messages()}


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready — still loading?")
    try:
        result = engine.chat(body.message, character=body.character)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return ChatResponse(**result)


@app.post("/api/session/reset")
def reset_session() -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready")
    with engine._lock:
        engine.reset_session()
    return engine.status()
