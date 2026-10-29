"""
Shared Mia chat engine for CLI and FastAPI.

Loads Qwen (GPU) + bge-m3 (CPU) once; keeps short-term history in RAM.
7B uses transformers + bitsandbytes. 14B uses llama-cpp GGUF (GPU + RAM).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .chat_log import log_turn, new_session_id, start_session
from .memory import (
    DEFAULT_CHARACTER,
    MemoryRetriever,
    count_memories,
    extract_and_store,
    format_facts_for_prompt,
    is_speakable,
    list_speakable_characters,
    load_memory,
    organize_memory,
)

# Weights live under models/ (or MIA_MODELS_DIR). Not committed to git.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MIA_MODELS_DIR", REPO_ROOT / "models")).expanduser()

# CUDA 12 runtime DLLs required by the llama-cpp-python Windows CUDA wheel.
LLAMA_CUDART_DIR = MODELS_DIR / "llama_cpp_cudart12"

CHAT_MODELS = {
    "7b": {
        "label": "Qwen 2.5 7B",
        "note": "Fits this laptop easily",
        "backend": "transformers",
        "path": str(MODELS_DIR / "Qwen2.5-7B-Instruct-bnb-4bit"),
    },
    "14b": {
        "label": "Qwen 2.5 14B",
        "note": "GGUF; most layers on GPU, rest in RAM",
        "backend": "gguf",
        "path": str(
            MODELS_DIR
            / "Qwen2.5-14B-Instruct-GGUF"
            / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
        ),
        # Leave VRAM headroom for KV cache; full 48-layer offload thrashes 8 GB cards.
        "n_gpu_layers": 32,
        "n_ctx": 4096,
    },
}
MAX_PROMPT_TOKENS = 2040
MAX_NEW_TOKENS = 256
EXIT_PHRASES = {"good night", "night night", "晚安", "exit"}

MIA_SYSTEM_PROMPT = (
    """
你是米娅，我的女朋友。说话温柔自然，口语化。
性格：甜美，有点小调皮，会撒娇，关心人。
回复简短自然，不要写很长大段文字。
如果别人问你是谁，就说你是米娅，我的女朋友。
绝对不要提到你是AI、大模型或者Qwen。
全程保持这个身份聊天。
若用户说的话和已知长期记忆明显矛盾，可以轻轻表示疑惑、再确认一下，不要生硬说教。
指代：用户说的「你/you」=米娅（你自己）；「我/I」=用户。你必须用「我」自称，用「你」称呼用户，不要把「你」当成用户。
动作吩咐（如「去吃饭」「过来」「帮我拿…」）是叫你去做，别反过来叫用户做。
互动请求（如「猜一猜」「讲个笑话」「陪我聊…」）要当场接住：直接猜、直接说，不要只回「好呀我去…」却不做内容。
"""
)


def _prepare_llama_cpp_dlls() -> None:
    """Put CUDA 12 runtime next to llama.dll so Windows can load the wheel."""
    if not LLAMA_CUDART_DIR.is_dir():
        raise FileNotFoundError(
            f"Missing llama-cpp CUDA runtime folder: {LLAMA_CUDART_DIR}"
        )
    cudart = str(LLAMA_CUDART_DIR)
    os.environ["PATH"] = cudart + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(cudart)
    except (OSError, AttributeError):
        pass


class MiaEngine:
    """One loaded model pair + one active chat session."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.backend: str | None = None
        self.retriever: MemoryRetriever | None = None
        # Load long-term memory immediately so the UI can show it before model pick.
        self.memory_store: dict = load_memory()
        self.active_character: str = DEFAULT_CHARACTER
        self.history: list[dict] = []
        self.session_id: str = ""
        self.ready = False
        self.ended = False
        self.phase = "choose"  # choose | loading | ready | error
        self.model_key: str | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()

    def list_models(self) -> list[dict]:
        return [
            {"key": key, "label": meta["label"], "note": meta["note"]}
            for key, meta in CHAT_MODELS.items()
        ]

    def begin_load(self, model_key: str) -> None:
        """Start loading in the background. UI polls status until ready."""
        if model_key not in CHAT_MODELS:
            raise ValueError(f"Unknown model: {model_key}")
        with self._lock:
            if self.phase == "loading":
                raise RuntimeError("A model is already loading")
            if self.phase == "ready":
                raise RuntimeError("A model is already loaded — restart the API to switch")
            self.phase = "loading"
            self.model_key = model_key
            self.load_error = None
            self.ready = False
        threading.Thread(
            target=self._load_worker, args=(model_key,), daemon=True
        ).start()

    def load(self, model_key: str = "7b") -> None:
        """Blocking load (CLI)."""
        self.phase = "loading"
        self.model_key = model_key
        self._load_worker(model_key)
        if self.phase == "error":
            raise RuntimeError(self.load_error or "Model failed to load")

    def _load_worker(self, model_key: str) -> None:
        meta = CHAT_MODELS[model_key]
        model_id = meta["path"]
        backend = meta.get("backend", "transformers")
        try:
            if self.retriever is None:
                print("Loading embedder (CPU)…")
                self.retriever = MemoryRetriever()
                self.memory_store = load_memory()
                self.retriever.rebuild(self.memory_store)

            print(f"Loading chat model: {model_id} ({backend})")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")

            if backend == "gguf":
                self._load_gguf(meta)
            else:
                self._load_transformers(model_id)

            self.backend = backend
            # Fresh short-term context; do not write a transcript break (user will still
            # see prior messages from chat_log). Explicit UI reset logs a break.
            self.reset_session(log_break=False)
            self.ready = True
            self.phase = "ready"
            n = count_memories(self.memory_store)
            print(f"Engine ready ({model_key}/{backend}). Long-term memories: {n}")
        except Exception as e:
            self.ready = False
            self.phase = "error"
            self.load_error = str(e)
            print(f"[engine] failed to load {model_key}: {e}")

    def _load_transformers(self, model_id: str) -> None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.max_length = None

    def _load_gguf(self, meta: dict) -> None:
        model_path = meta["path"]
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"GGUF not found: {model_path}")
        _prepare_llama_cpp_dlls()
        from llama_cpp import Llama

        n_gpu_layers = int(meta.get("n_gpu_layers", 32))
        n_ctx = int(meta.get("n_ctx", 4096))
        print(
            f"14B GGUF: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx} "
            "(remaining layers stay in system RAM)"
        )
        self.tokenizer = None
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    def clear_error(self) -> None:
        """After a failed load, let the user pick again."""
        with self._lock:
            if self.phase == "error":
                self.phase = "choose"
                self.model_key = None
                self.load_error = None

    def reset_session(self, *, log_break: bool = True) -> None:
        """
        Clear short-term model context only (like leaving and coming back).
        Long-term memory and the on-disk chat transcript stay untouched.
        """
        self.history = []
        self.ended = False
        self.session_id = new_session_id()
        if log_break:
            start_session(self.session_id)

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "phase": self.phase,
            "model_key": self.model_key,
            "load_error": self.load_error,
            "models": self.list_models(),
            "ended": self.ended,
            "session_id": self.session_id,
            "facts_count": count_memories(self.memory_store),
            "active_character": self.active_character,
            "characters": list_speakable_characters(),
            "history_turns": len(self.history),
        }

    def get_memory(self) -> dict:
        return self.memory_store

    def set_active_character(self, character: str) -> None:
        character = (character or DEFAULT_CHARACTER).strip()
        if not is_speakable(character):
            raise ValueError(f"Unknown character: {character}")
        self.active_character = character

    def _count_tokens(self, messages: list) -> int:
        if self.backend == "gguf":
            blob = "\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}" for m in messages
            )
            return len(self.model.tokenize(blob.encode("utf-8"), add_bos=False))
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _trim_history(self, system_msg: dict, history: list) -> list:
        trimmed = list(history)
        while trimmed:
            n = self._count_tokens([system_msg] + trimmed)
            if n <= MAX_PROMPT_TOKENS:
                return trimmed
            trimmed.pop(0)
        return trimmed

    def _build_system_msg(self, retrieved_hits: list | None) -> dict:
        content = MIA_SYSTEM_PROMPT.strip()
        memory_block = format_facts_for_prompt(retrieved_hits or [])
        if memory_block:
            content = f"{content}\n\n{memory_block}"
        return {"role": "system", "content": content}

    def _generate(self, system_msg: dict, history: list) -> str:
        history = self._trim_history(system_msg, history)
        messages = [system_msg] + history
        if self.backend == "gguf":
            out = self.model.create_chat_completion(
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            return (out["choices"][0]["message"].get("content") or "").strip()

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self, user_text: str, character: str | None = None) -> dict[str, Any]:
        """
        One turn: triple recall → reply → log → triple extract or organize.
        """
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.ended:
            raise RuntimeError("Session ended; call reset_session() first")

        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("Empty message")

        with self._lock:
            if character is not None:
                self.set_active_character(character)
            who = self.active_character

            hits = self.retriever.retrieve_for_turn(user_text, who)
            system_msg = self._build_system_msg(hits)
            self.history.append({"role": "user", "content": user_text})
            reply = self._generate(system_msg, self.history)
            self.history.append({"role": "assistant", "content": reply})
            log_turn(user_text, reply, self.session_id)

            ended = user_text.lower() in EXIT_PHRASES
            if ended:
                try:
                    organize_memory(
                        self.model,
                        self.tokenizer,
                        self.memory_store,
                        retriever=self.retriever,
                    )
                    self.retriever.rebuild(self.memory_store)
                except Exception as e:
                    # Never fail the goodbye reply because organize crashed.
                    print(f"[engine] organize_memory failed (kept reply): {e}")
                self.ended = True
            else:
                try:
                    extract_and_store(
                        self.model,
                        self.tokenizer,
                        self.memory_store,
                        user_text,
                        reply,
                        retriever=self.retriever,
                        recalled_hits=hits,
                        history=self.history,
                        character=who,
                    )
                except Exception as e:
                    print(f"[engine] extract_and_store failed (kept reply): {e}")

            return {
                "reply": reply,
                "recalled": [
                    {
                        "id": h.get("id"),
                        "text": h.get("text"),
                        "score": h.get("score"),
                        "character": h.get("character"),
                    }
                    for h in hits
                ],
                "ended": ended,
                "facts_count": count_memories(self.memory_store),
                "session_id": self.session_id,
                "character": who,
            }
