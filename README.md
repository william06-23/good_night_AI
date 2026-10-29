# Good Night AI (Mia)

Local chat with Mia, your girlfriend. Short-term chat stays in RAM for this session. Important facts are saved to `mia_chat/data/long_term_memory.json`. On each turn, `BAAI/bge-m3` (CPU) retrieves the most relevant facts and injects only those into the prompt.

You can run Qwen 2.5 **7B** (4-bit, transformers) or **14B** (GGUF, llama.cpp). The web UI asks you to pick one after the API is up. The dating-sim version lives on the `Rizz-the-AI` branch.

This folder is the chat program. Model weights go in a local `models/` folder (gitignored). The Python env is `.venv` at the repo root.

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

Use the CUDA wheel that matches the GPU driver. `cu124` is a common CUDA 12 index. The 14B path also needs the CUDA 12 runtime DLLs in `models/llama_cpp_cudart12`. `run_api.bat` and `run_mia.bat` put that folder on `PATH`.

In `web/`, run `npm install` once.

## Model weights

Not bundled in this repo. Put them under `models/` next to the launchers:

| Role | Runs on | Default path |
|------|---------|--------------|
| Chat, Qwen 2.5 7B 4-bit | GPU | `models/Qwen2.5-7B-Instruct-bnb-4bit` |
| Chat, Qwen 2.5 14B GGUF Q4_K_M | GPU + RAM | `models/Qwen2.5-14B-Instruct-GGUF/Qwen2.5-14B-Instruct-Q4_K_M.gguf` |
| Memory retrieval (`BAAI/bge-m3`) | CPU | `models/bge-m3` |
| llama-cpp CUDA 12 DLLs | — | `models/llama_cpp_cudart12` |

Downloads if you are not copying folders:

- 7B: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- Retriever: `BAAI/bge-m3` (or run `python -m mia_chat._download_bge_m3`)

To use weights somewhere else, set `MIA_MODELS_DIR` to that folder (same layout as `models/` above).

## How to run

### Web UI

Double-click **`run_web.bat`**.

1. A **Mia API** window starts at http://127.0.0.1:8000. It does not load a chat model until you pick one.
2. Vite opens http://127.0.0.1:5173. Choose **7B** or **14B** there.

If the page says the API is offline, wait until the API window is up.

`run_api.bat` alone is enough for API debugging.

Leave a session with `good night`, `night night`, `晚安`, or `exit`. Mia replies first, then long-term memory is organized.

**New session** clears only the model’s short-term context. The transcript and long-term memory stay.

### CLI

Create `.venv` as above, then double-click **`run_mia.bat`**, or:

```bat
set PYTHONPATH=%CD%
.venv\Scripts\python.exe -m mia_chat.chat_cli
```

## API

- `GET /api/health`
- `POST /api/model` and `POST /api/model/retry`
- `GET /api/memory`
- `GET /api/chat/history`
- `POST /api/chat`
- `POST /api/session/reset`

## Private runtime files

These are created on first use and are gitignored:

- `mia_chat/data/chat_log.jsonl`
- `mia_chat/data/long_term_memory.json`
