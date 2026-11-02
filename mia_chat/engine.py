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
    NARRATOR_KEY,
    PLAYER_KEY,
    MemoryRetriever,
    add_memory,
    count_memories,
    ensure_character,
    extract_and_store,
    format_facts_for_prompt,
    is_speakable,
    list_speakable_characters,
    load_memory,
    organize_memory,
    save_memory,
    set_speakable_characters,
)
from .story import (
    StoryState,
    append_log,
    clamp_addressees,
    generate_character_line,
    generate_story_setup,
    lookup_character,
    narrator_bridge,
    narrator_cue,
    narrator_intro,
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
        self.story: StoryState | None = None
        self.mode: str = "chat"  # chat | story

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
        out: dict[str, Any] = {
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
            "mode": self.mode,
            "story": self.story.public_dict() if self.story else None,
        }
        return out

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

    def chat(
        self,
        user_text: str,
        character: str | None = None,
        to: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Chat turn. In story mode this is a player interrupt with required to[].
        Legacy girlfriend chat still works when mode != story.
        """
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.ended and self.mode != "story":
            raise RuntimeError("Session ended; call reset_session() first")

        user_text = (user_text or "").strip()
        if not user_text:
            raise ValueError("Empty message")

        with self._lock:
            if self.mode == "story" and self.story is not None:
                return self._story_player_turn(user_text, to or [])

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
                "events": [],
                "story": None,
            }

    # --- Story mode ---------------------------------------------------------

    def _seed_story_memories(self, state: StoryState) -> None:
        names = state.cast_names()
        set_speakable_characters(names)
        for name in names:
            ensure_character(self.memory_store, name)
        ensure_character(self.memory_store, NARRATOR_KEY)
        ensure_character(self.memory_store, PLAYER_KEY)

        seeds: list[tuple[str, str]] = [
            (NARRATOR_KEY, f"故事背景：{state.background[:400]}"),
            (
                PLAYER_KEY,
                f"我是{state.player.get('name')}，{state.player.get('identity')}。"
                f"目标：{state.player.get('goal')}",
            ),
        ]
        for card in state.character_dict.values():
            seeds.append(
                (
                    card.name,
                    f"{card.name}（{card.slot}）：{card.description} 性格：{card.persona}",
                )
            )
        for who, text in seeds:
            add_memory(self.memory_store, who, text)
        save_memory(self.memory_store)
        if self.retriever is not None:
            self.retriever.rebuild(self.memory_store)
        if names:
            self.active_character = names[0]

    def generate_story(self, theme: str | None = None) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        with self._lock:
            state = generate_story_setup(self.model, self.tokenizer, theme=theme)
            self._seed_story_memories(state)
            self.story = state
            self.mode = "story"
            self.ended = False
            self.waiting_cleared()
            return {"story": state.public_dict(), "status": self.status()}

    def waiting_cleared(self) -> None:
        if self.story:
            self.story.waiting_for_player = False

    def start_intro(self) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.story is None:
            raise RuntimeError("Generate a story first")
        with self._lock:
            state = self.story
            if state.story_complete:
                raise RuntimeError("Story already complete")
            text = narrator_intro(self.model, self.tokenizer, state)
            state.current_narration = text
            append_log(
                state, speaker=NARRATOR_KEY, text=text, to=[], kind="narrator"
            )
            # First cue after intro
            events = self._advance_once_unlocked(force_cue=True)
            return {
                "events": events,
                "story": state.public_dict(),
                "status": self.status(),
            }

    def advance_scene(self) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.story is None:
            raise RuntimeError("Generate a story first")
        with self._lock:
            if self.story.story_complete:
                return {
                    "events": [],
                    "story": self.story.public_dict(),
                    "status": self.status(),
                }
            if self.story.waiting_for_player:
                return {
                    "events": [],
                    "story": self.story.public_dict(),
                    "status": self.status(),
                    "waiting_for_player": True,
                }
            events = self._advance_once_unlocked(force_cue=True)
            return {
                "events": events,
                "story": self.story.public_dict(),
                "status": self.status(),
            }

    def _maybe_end_scene_unlocked(self, cue_end: bool) -> list[dict[str, Any]]:
        state = self.story
        assert state is not None
        events: list[dict[str, Any]] = []
        hit_cap = state.message_count_in_scene >= state.scene_message_cap
        if not (cue_end or hit_cap):
            return events
        if state.scene_index >= state.max_scenes:
            state.story_complete = True
            events.append(
                {
                    "type": "story_complete",
                    "text": "故事结束（已达 5 幕上限）。",
                }
            )
            return events
        bridge = narrator_bridge(self.model, self.tokenizer, state)
        state.current_narration = bridge
        entry = append_log(
            state, speaker=NARRATOR_KEY, text=bridge, to=[], kind="narrator"
        )
        events.append({"type": "narrator", **entry})
        state.scene_index += 1
        state.message_count_in_scene = 0
        events.append(
            {
                "type": "scene_start",
                "scene_index": state.scene_index,
                "max_scenes": state.max_scenes,
            }
        )
        return events

    def _advance_once_unlocked(self, *, force_cue: bool = False) -> list[dict[str, Any]]:
        _ = force_cue
        state = self.story
        assert state is not None
        events: list[dict[str, Any]] = []

        cue = narrator_cue(self.model, self.tokenizer, state)
        events.append({"type": "cue", **cue})

        # Scene end before speaking if narrator says so and there is already content
        if cue.get("end_scene") and state.message_count_in_scene > 0:
            events.extend(self._maybe_end_scene_unlocked(True))
            if state.story_complete:
                return events
            # After bridge, cue again for the new scene
            cue = narrator_cue(self.model, self.tokenizer, state)
            events.append({"type": "cue", **cue})

        speaker = str(cue.get("speaker") or PLAYER_KEY)
        to = list(cue.get("to") or [])

        if speaker == PLAYER_KEY:
            state.waiting_for_player = True
            state.suggested_to = [
                n for n in to if n in state.cast_names()
            ] or list(state.cast_names()[:1])
            events.append(
                {
                    "type": "wait_player",
                    "suggested_to": list(state.suggested_to),
                    "reason": cue.get("reason"),
                }
            )
            return events

        if speaker == NARRATOR_KEY:
            # Short narrator beat using intro-style context
            text = narrator_intro(self.model, self.tokenizer, state)
            state.current_narration = text
            entry = append_log(
                state, speaker=NARRATOR_KEY, text=text, to=[], kind="narrator"
            )
            events.append({"type": "narrator", **entry})
            events.extend(
                self._maybe_end_scene_unlocked(
                    state.message_count_in_scene >= state.scene_message_cap
                )
            )
            return events

        card = lookup_character(state.character_dict, speaker)
        if card is None:
            # Fallback: wait for player
            state.waiting_for_player = True
            state.suggested_to = list(state.cast_names()[:1])
            events.append(
                {
                    "type": "wait_player",
                    "suggested_to": list(state.suggested_to),
                    "reason": f"未知角色「{speaker}」，改由玩家行动",
                }
            )
            return events

        to = clamp_addressees(to, state)
        query = state.current_narration or state.background[:200]
        hits = []
        if self.retriever is not None:
            hits = self.retriever.retrieve_for_turn(query, card.name)
        line = generate_character_line(
            self.model,
            self.tokenizer,
            state,
            card,
            to,
            recalled_hits=hits,
        )
        entry = append_log(
            state, speaker=card.name, text=line, to=to, kind="dialogue"
        )
        self.active_character = card.name
        events.append(
            {
                "type": "dialogue",
                **entry,
                "recalled": hits,
            }
        )
        try:
            extract_and_store(
                self.model,
                self.tokenizer,
                self.memory_store,
                f"（对 {', '.join(to)}）{line}",
                "",
                retriever=self.retriever,
                recalled_hits=hits,
                history=[],
                character=card.name,
            )
        except Exception as e:
            print(f"[engine] NPC memory extract failed: {e}")

        # End scene after the line if soft-capped, or if cue requested end
        # (and we did not already bridge at the top of this advance).
        events.extend(
            self._maybe_end_scene_unlocked(
                state.message_count_in_scene >= state.scene_message_cap
            )
        )
        if cue.get("end_scene") and not state.story_complete:
            events.extend(self._maybe_end_scene_unlocked(True))
        return events

    def _story_player_turn(self, user_text: str, to: list[str]) -> dict[str, Any]:
        state = self.story
        assert state is not None
        if state.story_complete:
            raise RuntimeError("Story already complete")

        addressees = clamp_addressees(to, state)
        # Must address at least one NPC
        npc_targets = [n for n in addressees if n in state.character_dict]
        if not npc_targets:
            raise ValueError("Select at least one character to speak to")

        state.waiting_for_player = False
        entry = append_log(
            state,
            speaker=PLAYER_KEY,
            text=user_text,
            to=npc_targets,
            kind="dialogue",
        )
        log_turn(user_text, "", self.session_id)
        self.history.append({"role": "user", "content": user_text})

        for who in npc_targets:
            try:
                hits = (
                    self.retriever.retrieve_for_turn(user_text, who)
                    if self.retriever
                    else []
                )
                extract_and_store(
                    self.model,
                    self.tokenizer,
                    self.memory_store,
                    user_text,
                    "",
                    retriever=self.retriever,
                    recalled_hits=hits,
                    history=self.history,
                    character=who,
                )
            except Exception as e:
                print(f"[engine] player memory extract failed for {who}: {e}")

        self.active_character = npc_targets[0]
        events: list[dict[str, Any]] = [{"type": "player", **entry}]
        events.extend(
            self._maybe_end_scene_unlocked(
                state.message_count_in_scene >= state.scene_message_cap
            )
        )
        return {
            "reply": "",
            "recalled": [],
            "ended": state.story_complete,
            "facts_count": count_memories(self.memory_store),
            "session_id": self.session_id,
            "character": self.active_character,
            "to": npc_targets,
            "events": events,
            "story": state.public_dict(),
        }
