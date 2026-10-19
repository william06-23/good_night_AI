# Good Night AI (Mia) — Rizz branch

Local chat with Mia, a cold university stranger you can try to get closer to. Short-term chat stays in RAM. Important facts go to `mia_chat/data/long_term_memory.json`. Each turn, `BAAI/bge-m3` (CPU) retrieves the relevant facts and only those are added to the prompt.

Affection, dating, a possible boyfriend, and a fixed campus week are on this branch. `master` is the earlier girlfriend chat without that system.

This folder is the program. It does **not** include the model weights or the Python environment.

## What another machine needs

- Windows
- An NVIDIA GPU. 7B 4-bit wants about 6 GB VRAM. 14B GGUF keeps most layers on the GPU and the rest in RAM
- A recent NVIDIA driver
- Python **3.11 or 3.12** (3.14 is too new for these packages)
- Node.js, for the web UI
- Internet the first time, if weights are downloaded instead of copied

## Python packages

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes sentence-transformers huggingface_hub fastapi uvicorn llama-cpp-python
```

Use the CUDA wheel that matches the GPU driver. `cu124` is the index used on the original machine. The 14B path also needs the CUDA 12 runtime DLLs next to `llama.dll` (`llama_cpp_cudart12` on the original machine). `run_api.bat` puts that folder on `PATH`.

In `web/`, run `npm install` once.

## Model weights

Not bundled in this repo.

| Role | Runs on | Original path |
|------|---------|---------------|
| Chat, Qwen 2.5 7B 4-bit | GPU | `C:\Users\William\William's_AI_project\Qwen2.5-7B-Instruct-bnb-4bit` |
| Chat, Qwen 2.5 14B GGUF Q4_K_M | GPU + RAM | `C:\Users\William\William's_AI_project\Qwen2.5-14B-Instruct-GGUF\Qwen2.5-14B-Instruct-Q4_K_M.gguf` |
| Memory retrieval (`BAAI/bge-m3`) | CPU | `C:\Users\William\William's_AI_project\bge-m3` |

Downloads if you are not copying folders:

- 7B: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Retriever: `BAAI/bge-m3` (or `mia_chat/_download_bge_m3.py` after editing its `DEST` path)

Then set paths in:

- `CHAT_MODELS` in `mia_chat/engine.py` (7B folder, 14B `.gguf` file, and `LLAMA_CUDART_DIR`)
- `EMBED_MODEL_ID` in `mia_chat/memory.py` (keep `EMBED_DEVICE` as `"cpu"` so Qwen keeps the GPU)

## How to run

### Web UI

Double-click **`run_web.bat`**.

1. A **Mia API** window starts at http://127.0.0.1:8000. It does not load a chat model until you pick one.
2. Vite opens http://127.0.0.1:5173. Choose **7B** or **14B** there.

If the page says the API is offline, wait until the API window is up.

`run_api.bat` alone is enough for API debugging.

Leave a session with `good night`, `night night`, `晚安`, or `exit`. Mia replies first, then long-term memory is organized.

**New session** clears only the model’s short-term context. The transcript and long-term memory stay.

**Restart** (with a confirmation dialog) deletes the transcript, long-term memory, campus events, and relationship state. The affection speed slider is kept.

Settings has the UI language and an affection speed slider from **0.8** to **1.5** (default **1**).

### CLI

Edit the `python.exe` path in `run_mia.bat`, then double-click it, or:

```bat
set PYTHONPATH=%CD%
"path\to\your\venv\Scripts\python.exe" -m mia_chat.chat_cli
```

## API

- `GET /api/health`
- `POST /api/model` and `POST /api/model/retry`
- `GET /api/memory`
- `GET /api/chat/history`
- `POST /api/chat`
- `POST /api/session/reset`
- `POST /api/restart`
- `POST /api/speed`

## Private runtime files

These are created on first use and are gitignored:

- `mia_chat/data/chat_log.jsonl`
- `mia_chat/data/long_term_memory.json`
- `mia_chat/data/affection.json`
- `mia_chat/data/campus_life.json`

Behavior details for later edits live in `AGENT_CONTEXT.md`.
