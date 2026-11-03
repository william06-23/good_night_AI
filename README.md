# AI Text Game

Directed text adventure (branch **`AI_text_game`**). One local Qwen (7B/14B) plays story generator, persona designer, narrator/director, and NPC voices. **bge-m3** (CPU) retrieves multi-perspective memories (`Narrator` / `Player` / per-NPC).

Toggle **中文 / EN** in the web UI (top right) before generating — story setup, personas, narration, and dialogue all follow that language.

Classic girlfriend chat lives on `master`; dating-sim on `Rizz-the-AI`.

## Setup

Model weights go in `models/` (gitignored) or under `AI_TEXT_GAME_MODELS_DIR` (legacy alias: `MIA_MODELS_DIR`). Prefer a repo `.venv`, or copy `local_env.bat.example` → `local_env.bat` for machine-specific Python/models paths.

### What you need

- Windows + NVIDIA GPU (7B ≈ 6 GB VRAM; 14B GGUF uses GPU + RAM)
- Python **3.11 or 3.12**
- Node.js (web UI)
- Internet the first time if downloading weights

### Python packages

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes sentence-transformers huggingface_hub fastapi uvicorn llama-cpp-python
```

In `web/`, run `npm install` once.

### Model layout

| Role | Path |
|------|------|
| Qwen 2.5 7B 4-bit | `models/Qwen2.5-7B-Instruct-bnb-4bit` |
| Qwen 2.5 14B GGUF | `models/Qwen2.5-14B-Instruct-GGUF/Qwen2.5-14B-Instruct-Q4_K_M.gguf` |
| bge-m3 | `models/bge-m3` |
| llama-cpp CUDA 12 DLLs | `models/llama_cpp_cudart12` |

## How to run

### Web UI

Double-click **`run_web.bat`**.

1. API at http://127.0.0.1:8000 (pick 7B/14B in the UI)
2. Vite at http://127.0.0.1:5173 — set language → generate story → start scene → advance / speak

### CLI (no browser)

Double-click **`run_story.bat`**, or:

```bat
set PYTHONPATH=%CD%
.venv\Scripts\python.exe -m ai_text_game.story_cli
```

### API only

`run_api.bat` → `uvicorn ai_text_game.api:app`

## Package layout

Python package: **`ai_text_game/`** (engine, story, memory, API).

Runtime data (gitignored): `ai_text_game/data/`.

## API

- `GET /api/health`
- `POST /api/model`, `/api/model/retry`
- `GET /api/memory`, `/api/characters`, `/api/story`
- `POST /api/story/generate` — `{ theme?, language?: "zh"|"en" }`
- `POST /api/story/intro`, `/api/story/advance`
- `POST /api/chat` — `{ message, to: string[] }`
- `POST /api/session/reset`
