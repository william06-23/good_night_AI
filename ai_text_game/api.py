"""
FastAPI server for the directed text game — load models once, story via HTTP.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .engine import TextGameEngine
from .memory import list_speakable_characters, load_memory, public_memory_store

engine = TextGameEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Text Game API up. Waiting for a model choice (7b or 14b) from the UI.")
    yield
    print("AI Text Game API shutting down.")


app = FastAPI(title="AI Text Game", lifespan=lifespan)
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
    to: list[str] = Field(default_factory=list)


class ModelChoice(BaseModel):
    key: str = Field(..., min_length=1)


class ThemeBody(BaseModel):
    theme: str | None = None
    language: str | None = "zh"


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


@app.get("/api/story")
def get_story() -> dict[str, Any]:
    return {
        "story": engine.story.public_dict() if engine.story else None,
        "status": engine.status(),
    }


@app.get("/api/story/live")
def story_live(since: int = Query(0, ge=0)) -> dict[str, Any]:
    """Poll while a background intro/advance/chat job is running (~0.25s)."""
    return engine.poll_live(since)


@app.post("/api/story/generate")
def story_generate(body: ThemeBody | None = None) -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready — still loading?")
    theme = body.theme if body else None
    language = (body.language if body else None) or "zh"
    try:
        return engine.generate_story(theme=theme, language=language)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.post("/api/story/intro")
def story_intro() -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready")
    try:
        return engine.begin_intro_job()
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/story/advance")
def story_advance() -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready")
    try:
        return engine.begin_advance_job()
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready — still loading?")
    try:
        return engine.begin_chat_job(body.message, to=body.to)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/session/reset")
def reset_session() -> dict[str, Any]:
    if not engine.ready:
        raise HTTPException(503, "Engine not ready")
    with engine._lock:
        engine.reset_session()
    return engine.status()
