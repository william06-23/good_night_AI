# Good Night AI (Mia)

Local chat with Mia on Qwen 2.5 7B (4-bit). Short-term chat stays in RAM for this session. Important facts are saved to `mia_chat/data/long_term_memory.json`. On each turn, `BAAI/bge-m3` (CPU) retrieves the most relevant facts and injects only those into the prompt.

This folder is the chat program. It does **not** include the model weights or the Python environment. Those stay on each machine.

## What the other machine needs

- Windows
- An NVIDIA GPU with enough VRAM for a 7B 4-bit model (about 6 GB or more)
- A recent NVIDIA driver
- Python **3.11 or 3.12** (3.14 is too new for these packages)
- Internet the first time, if the model is downloaded from Hugging Face instead of copied from a disk

## Python packages

Create a virtual environment, then install:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes sentence-transformers huggingface_hub
```

Use the CUDA wheel that matches the GPU driver. `cu124` is the index used on the original machine.

## Model weights

Two local folders (not bundled in this repo):

| Role | Runs on | Original path |
|------|---------|---------------|
| Chat (Qwen 2.5 7B 4-bit) | GPU | `C:\Users\William\William's_AI_project\Qwen2.5-7B-Instruct-bnb-4bit` |
| Memory retrieval (`BAAI/bge-m3`) | CPU | `C:\Users\William\William's_AI_project\bge-m3` |

On another device, copy those folders or download:

- Chat: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Retriever: `BAAI/bge-m3` (or run `mia_chat/_download_bge_m3.py` after editing its `DEST` path)

Then set:

- `MODEL_ID` in `mia_chat/chat_cli.py` → chat model folder
- `EMBED_MODEL_ID` in `mia_chat/memory.py` → bge-m3 folder
- `EMBED_DEVICE` in `mia_chat/memory.py` → keep `"cpu"` on 8 GB VRAM laptops so Qwen keeps the GPU

Do not zip the weights into this project just to share the Python files. Copy model folders separately if the other machine is offline.

## How to run

### CLI

1. Edit `run_mia.bat` / `run_api.bat` so the `python.exe` path is the venv on **that** machine.
2. Edit `MODEL_ID` in `mia_chat/engine.py` and `EMBED_MODEL_ID` in `mia_chat/memory.py`.
3. Double-click `run_mia.bat`, or:

```bat
set PYTHONPATH=%CD%
"path\to\your\venv\Scripts\python.exe" -m mia_chat.chat_cli
```

Leave with `good night`, `night night`, `晚安`, or `exit`.

### Web UI (simple React test front-end)

Double-click **`run_web.bat`**. It opens:
1. A **Mia API** window (loads GPU models on http://127.0.0.1:8000)
2. Vite + your browser at http://127.0.0.1:5173

If the page says “API offline”, wait until the API window finishes loading.  
`run_api.bat` alone is still available for API-only debugging.

API routes: `GET /api/health`, `GET /api/memory`, `POST /api/chat`, `POST /api/session/reset`.

Also: `pip install fastapi uvicorn` in the venv, and Node.js for the `web/` app (`cd web && npm install` once).
