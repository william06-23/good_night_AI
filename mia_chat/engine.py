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

from .affection import (
    BREAKUP_BELOW,
    CLINGY_ABOVE,
    CONFESS_THRESHOLD,
    absence_penalty,
    affection_prompt_block,
    accelerate_affection_delta,
    apply_affection_delta,
    format_affection_display,
    HURT_RAW_DELTA,
    judge_relationship_intent,
    catch_up_boyfriend,
    clamp_speed,
    _empty_store as empty_affection,
    load_affection,
    parabolic_speed_weight,
    save_affection,
    scale_by_user_speed,
    score_affection_delta,
    weighted_affection_delta,
)
from .chat_log import (
    clear_log,
    hours_since_last_user_message,
    log_turn,
    new_session_id,
    start_session,
)
from .memory import (
    MemoryRetriever,
    extract_and_store,
    format_facts_for_prompt,
    _empty_store as empty_memory,
    load_memory,
    organize_memory,
    save_memory,
    store_hurt_fact,
)
from .schedule import (
    _empty as empty_campus,
    ensure_mia_profile,
    maybe_grow_mia_life,
    maybe_roll_campus_event,
    persona_core_block,
    save_campus,
    schedule_prompt_block,
)

# CUDA 12 runtime DLLs required by the llama-cpp-python Windows CUDA wheel.
LLAMA_CUDART_DIR = Path(r"C:\Users\William\William's_AI_project\llama_cpp_cudart12")

CHAT_MODELS = {
    "7b": {
        "label": "Qwen 2.5 7B",
        "note": "Fits this laptop easily",
        "backend": "transformers",
        "path": r"C:\Users\William\William's_AI_project\Qwen2.5-7B-Instruct-bnb-4bit",
    },
    "14b": {
        "label": "Qwen 2.5 14B",
        "note": "GGUF; most layers on GPU, rest in RAM",
        "backend": "gguf",
        "path": (
            r"C:\Users\William\William's_AI_project"
            r"\Qwen2.5-14B-Instruct-GGUF\Qwen2.5-14B-Instruct-Q4_K_M.gguf"
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
你是米娅，大学里的陌生女生，和用户不熟。高冷，对搭讪冷淡；新鲜的事才有兴趣，平庸话题会敷衍。
口语短句，别写长段。别说自己是AI或Qwen。问你是谁时，按下面的关系状态回答。
长期记忆里若有用户伤害过你，可以更冷，但别每次翻旧账。和记忆矛盾时轻轻确认，别说教。
被问到你自己、而固定事实和记忆里没有的事，可以现编一个合理答案，之后不要改口。
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
        # Load long-term facts immediately so the UI can show them before model pick.
        self.memory_store: dict = load_memory()
        self.affection: dict = load_affection()
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
            n = len(self.memory_store.get("facts") or [])
            print(f"Engine ready ({model_key}/{backend}). Long-term facts: {n}")
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

    def restart_all(self) -> None:
        """Wipe chat, long-term memory, campus events, and relationship state."""
        self.history = []
        self.ended = False
        self.session_id = new_session_id()
        clear_log()
        self.memory_store = empty_memory()
        save_memory(self.memory_store)
        if self.retriever is not None:
            self.retriever.rebuild(self.memory_store)
        speed = clamp_speed((self.affection or {}).get("speed", 1))
        self.affection = empty_affection()
        self.affection["speed"] = speed
        save_affection(self.affection)
        save_campus(empty_campus())
        if self.model is not None:
            try:
                ensure_mia_profile(
                    self.model,
                    self.tokenizer,
                    self.memory_store,
                    retriever=self.retriever,
                )
            except Exception as e:
                print(f"[engine] mia profile failed: {e}")

    def set_speed(self, speed: float) -> None:
        with self._lock:
            aff = self.affection or load_affection()
            self.affection = aff
            aff["speed"] = clamp_speed(speed)
            save_affection(aff)

    def status(self) -> dict[str, Any]:
        with self._lock:
            aff = self.affection or load_affection()
            self.affection = aff
            catch_up_boyfriend(aff)
            return {
                "ready": self.ready,
                "phase": self.phase,
                "model_key": self.model_key,
                "load_error": self.load_error,
                "models": self.list_models(),
                "ended": self.ended,
                "session_id": self.session_id,
                "facts_count": len(self.memory_store.get("facts") or []),
                "history_turns": len(self.history),
                "affection": float(aff.get("value", 20.0)),
                "blocked": bool(aff.get("blocked")),
                "accepted": bool(aff.get("accepted")),
                "has_boyfriend": bool(aff.get("has_boyfriend")),
                "speed": clamp_speed(aff.get("speed", 1)),
                "confess_threshold": float(CONFESS_THRESHOLD),
            }

    def get_memory(self) -> dict:
        return self.memory_store

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

    def _build_system_msg(
        self,
        retrieved_facts: list | None,
        *,
        confess_mode: str | None = None,
        silence_hours: float | None = None,
    ) -> dict:
        content = MIA_SYSTEM_PROMPT.strip()
        core = persona_core_block()
        if core:
            content = f"{content}\n\n{core}"
        content = (
            f"{content}\n\n"
            f"{affection_prompt_block(self.affection, silence_hours=silence_hours)}"
        )
        if confess_mode == "accept":
            content += "\n用户正在认真告白。答应他，简短说清你现在是他的女朋友，别突然变成无脑甜妹。"
        elif confess_mode == "taken":
            content += "\n用户正在告白。你有男朋友，不是他。拒绝，并说你有对象了。"
        elif confess_mode == "reject":
            content += "\n用户正在告白，但好感不够。按当前阶段拒绝，别给恋爱希望。"
        elif confess_mode == "breakup":
            content += "\n用户在提分手。接受，说清关系结束，之后不再以女朋友自居。"
        content += (
            "\n\n"
            + schedule_prompt_block(float(self.affection.get("value", 20.0)))
        )
        memory_block = format_facts_for_prompt(retrieved_facts or [])
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

    def chat(self, user_text: str) -> dict[str, Any]:
        """
        One turn: affection check → recall → reply → log → extract or organize.
        """
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.ended:
            raise RuntimeError("Session ended; call reset_session() first")

        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("Empty message")

        with self._lock:
            self.affection = load_affection()
            catch_up_boyfriend(self.affection)
            hits: list = []
            reply = ""
            blocked_turn = False

            # Already blocked: log the bounce, no reply.
            if self.affection.get("blocked"):
                self.history.append({"role": "user", "content": user_text})
                log_turn(user_text, "", self.session_id, blocked=True)
                blocked_turn = True
            else:
                try:
                    ensure_mia_profile(
                        self.model,
                        self.tokenizer,
                        self.memory_store,
                        retriever=self.retriever,
                    )
                    maybe_grow_mia_life(
                        self.model,
                        self.tokenizer,
                        self.memory_store,
                        retriever=self.retriever,
                    )
                    maybe_roll_campus_event(
                        self.model,
                        self.tokenizer,
                        self.memory_store,
                        retriever=self.retriever,
                    )
                except Exception as e:
                    print(f"[engine] campus event failed: {e}")
                hits = self.retriever.retrieve(user_text)
                self.history.append({"role": "user", "content": user_text})

                silence_hours = None
                if (
                    self.affection.get("accepted")
                    and float(self.affection.get("value", 0.0)) > CLINGY_ABOVE
                ):
                    silence_hours = hours_since_last_user_message()
                    gap = absence_penalty(silence_hours)
                    gap, pace = scale_by_user_speed(self.affection, gap)
                    if gap < 0:
                        before_gap = float(self.affection.get("value", 0.0))
                        apply_affection_delta(self.affection, gap)
                        print(
                            f"[affection] absence {silence_hours:.1f}h "
                            f"{format_affection_display(before_gap)} "
                            f"{gap:+.2f} → "
                            f"{format_affection_display(self.affection.get('value', 0))}"
                        )

                # Score affection from this message + recent context, then maybe block.
                before = float(self.affection.get("value", 20.0))
                try:
                    raw_delta = score_affection_delta(
                        self.model,
                        self.tokenizer,
                        self.history,
                        user_text,
                        recalled_hits=hits,
                    )
                except Exception as e:
                    print(f"[engine] affection delta failed: {e}")
                    raw_delta = 0.0
                curve = parabolic_speed_weight(before)
                delta = weighted_affection_delta(raw_delta, before)
                delta, streak_mult = accelerate_affection_delta(self.affection, delta)
                delta, pace = scale_by_user_speed(self.affection, delta)
                apply_affection_delta(self.affection, delta)
                after = float(self.affection.get("value", 0.0))
                print(
                    f"[affection] {format_affection_display(before)} "
                    f"raw={raw_delta:+.2f}×{curve:.2f}×streak{streak_mult:.2f}"
                    f"×pace{pace:.2f} "
                    f"→ {delta:+.2f} "
                    f"→ {format_affection_display(after)} "
                    f"(streak={self.affection.get('streak')})"
                )
                if raw_delta <= HURT_RAW_DELTA:
                    try:
                        store_hurt_fact(
                            self.model,
                            self.tokenizer,
                            self.memory_store,
                            user_text,
                            history=self.history,
                            retriever=self.retriever,
                        )
                    except Exception as e:
                        print(f"[engine] hurt fact failed: {e}")

                if self.affection.get("blocked"):
                    log_turn(user_text, "", self.session_id, blocked=True)
                    blocked_turn = True
                else:
                    intent = judge_relationship_intent(
                        self.model, self.tokenizer, self.history, user_text
                    )
                    confess_mode = None
                    user_breakup = bool(
                        intent.get("breakup") and self.affection.get("accepted")
                    )
                    if user_breakup:
                        confess_mode = "breakup"
                    elif intent.get("confession") and not self.affection.get("accepted"):
                        if self.affection.get("has_boyfriend"):
                            confess_mode = "taken"
                        elif after >= CONFESS_THRESHOLD:
                            confess_mode = "accept"
                        else:
                            confess_mode = "reject"

                    breakup = bool(
                        user_breakup
                        or (
                            self.affection.get("accepted") and after < BREAKUP_BELOW
                        )
                    )

                    system_msg = self._build_system_msg(
                        hits,
                        confess_mode=confess_mode,
                        silence_hours=silence_hours,
                    )
                    reply = self._generate(system_msg, self.history)
                    self.history.append({"role": "assistant", "content": reply})
                    log_turn(user_text, reply, self.session_id)

                    if confess_mode == "accept":
                        self.affection["accepted"] = True
                        save_affection(self.affection)
                    elif breakup:
                        self.affection["accepted"] = False
                        save_affection(self.affection)

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
                            print(
                                f"[engine] organize_memory failed (kept reply): {e}"
                            )
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
                            )
                        except Exception as e:
                            print(
                                f"[engine] extract_and_store failed (kept reply): {e}"
                            )

            ended = self.ended
            return {
                "reply": reply,
                "recalled": [
                    {
                        "id": h.get("id"),
                        "text": h.get("text"),
                        "score": h.get("score"),
                        "certainty": next(
                            (
                                f.get("certainty")
                                for f in (self.memory_store.get("facts") or [])
                                if f.get("id") == h.get("id")
                            ),
                            None,
                        ),
                    }
                    for h in hits
                ],
                "ended": ended,
                "facts_count": len(self.memory_store.get("facts") or []),
                "session_id": self.session_id,
                "affection": float(self.affection.get("value", 0.0)),
                "blocked": bool(
                    self.affection.get("blocked") or blocked_turn
                ),
                "accepted": bool(self.affection.get("accepted")),
                "has_boyfriend": bool(self.affection.get("has_boyfriend")),
                "blocked_message": blocked_turn,
            }
