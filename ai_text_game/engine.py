"""
Shared engine for the directed text game (CLI + FastAPI).

Loads Qwen (GPU) + bge-m3 (CPU) once. Story generation, narrator cues,
and NPC lines live in story.py; this module owns model load + session state.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .chat_log import log_turn, new_session_id, start_session
from .memory import (
    NARRATOR_KEY,
    PLAYER_KEY,
    RETRIEVE_TOP_K,
    MemoryRetriever,
    add_memory,
    count_memories,
    ensure_character,
    extract_and_store,
    extract_attitude_updates,
    is_speakable,
    list_speakable_characters,
    load_memory,
    organize_memory,
    record_dialogue_beat,
    record_present_scene_views,
    save_memory,
    seed_attitude_memories,
    set_memory_language,
    set_speakable_characters,
    upsert_attitude_memory,
    wipe_memory,
)
from .prompts import normalize_lang, story_prompts
from .story import (
    StoryState,
    append_log,
    choose_present_cast,
    clamp_addressees,
    format_scene_log,
    generate_character_beat,
    generate_story_setup,
    lookup_character,
    narrator_cue,
    prepare_character_line,
    prepare_narrator_bridge,
    prepare_narrator_continue,
    prepare_narrator_intro,
    scene_entries,
    validate_player_action,
)

# Weights live under models/ (or AI_TEXT_GAME_MODELS_DIR). Not committed to git.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Pause after a spoken/narrated line before the next cue or speaker.
# Memory extract / light prep run during this window.
LINE_BEAT_SECONDS = 1.5


def _resolve_models_dir() -> Path:
    for key in ("AI_TEXT_GAME_MODELS_DIR", "MIA_MODELS_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser()
    return REPO_ROOT / "models"


MODELS_DIR = _resolve_models_dir()

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


class TextGameEngine:
    """One loaded model pair + one active text-game session."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.backend: str | None = None
        self.retriever: MemoryRetriever | None = None
        # Load long-term memory immediately so status/API can show it before model pick.
        self.memory_store: dict = load_memory()
        self.active_character: str = ""
        self.history: list[dict] = []
        self.session_id: str = ""
        self.ready = False
        self.phase = "choose"  # choose | loading | ready | error
        self.model_key: str | None = None
        self.load_error: str | None = None
        self._lock = threading.Lock()
        # Memory/prep for the line that just finished; runs at start of next turn
        # so the client interrupt window does not hold the engine lock.
        self._pending_followup: Callable[[], None] | None = None
        # Testing: prompts used for each generated line (also embedded on events).
        self.prompt_log: list[dict[str, Any]] = []
        self.story: StoryState | None = None
        # Experimental: small stage actions for NPCs + player (model-checked).
        self.actions_enabled: bool = False
        # Live UI feed: background jobs publish here; client polls every ~0.25s.
        self._live_lock = threading.Lock()
        self._job_busy = False
        self._live_seq = 0
        self._live_messages: list[dict[str, Any]] = []

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
        """Clear short-term context; long-term memory and transcript stay."""
        self.history = []
        self.story = None
        self.active_character = ""
        self.prompt_log = []
        self._pending_followup = None
        self.session_id = new_session_id()
        if log_break:
            start_session(self.session_id)

    def _record_prompt(self, debug: dict[str, Any]) -> dict[str, Any]:
        """Append to prompt_log (testing) and return the stored entry."""
        entry = {"id": len(self.prompt_log), **debug}
        self.prompt_log.append(entry)
        return entry

    def status(self) -> dict[str, Any]:
        story_done = bool(self.story and self.story.story_complete)
        return {
            "ready": self.ready,
            "phase": self.phase,
            "model_key": self.model_key,
            "load_error": self.load_error,
            "models": self.list_models(),
            "ended": story_done,
            "session_id": self.session_id,
            "facts_count": count_memories(self.memory_store),
            "active_character": self.active_character,
            "characters": list_speakable_characters(),
            "history_turns": len(self.history),
            "mode": "story" if self.story is not None else "idle",
            "story": self.story.public_dict() if self.story else None,
            "actions_enabled": bool(self.actions_enabled),
            # UI can detect a stale API process that predates live polling.
            "live_feed": True,
            "job_busy": bool(self._job_busy),
        }

    def set_actions_enabled(self, enabled: bool) -> None:
        self.actions_enabled = bool(enabled)
        if self.story is not None:
            self.story.actions_enabled = self.actions_enabled

    def get_memory(self) -> dict:
        return self.memory_store

    def set_active_character(self, character: str) -> None:
        character = (character or "").strip()
        if not character or not is_speakable(character):
            raise ValueError(f"Unknown character: {character}")
        self.active_character = character

    def _reset_live(self) -> None:
        with self._live_lock:
            self._live_seq = 0
            self._live_messages = []

    def _publish_live(self, msg: dict[str, Any]) -> None:
        with self._live_lock:
            self._live_seq += 1
            self._live_messages.append({**msg, "seq": self._live_seq})

    def poll_live(self, since: int = 0) -> dict[str, Any]:
        """Messages with seq > since for the UI poll loop."""
        with self._live_lock:
            msgs = [m for m in self._live_messages if int(m.get("seq") or 0) > since]
            next_since = self._live_seq
            job_busy = self._job_busy
        return {
            "messages": msgs,
            "next_since": next_since,
            "job_busy": job_busy,
            "story": self.story.public_dict() if self.story else None,
            "status": self.status(),
        }

    def _start_live_job(self, worker: Callable[[], None], name: str) -> dict[str, Any]:
        with self._live_lock:
            if self._job_busy:
                raise RuntimeError("A story turn is already running")
            self._job_busy = True
            self._live_seq = 0
            self._live_messages = []

        def _run() -> None:
            try:
                worker()
            except Exception as e:
                self._publish_live({"type": "error", "message": str(e)})
            finally:
                with self._live_lock:
                    self._job_busy = False

        threading.Thread(target=_run, daemon=True, name=name).start()
        return {"started": True, "job_busy": True}

    def begin_intro_job(self) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.story is None:
            raise RuntimeError("Generate a story first")

        def worker() -> None:
            for msg in self.start_intro_stream():
                self._publish_live(msg)

        return self._start_live_job(worker, "story-intro")

    def begin_advance_job(self) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.story is None:
            raise RuntimeError("Generate a story first")

        def worker() -> None:
            for msg in self.advance_scene_stream():
                self._publish_live(msg)

        return self._start_live_job(worker, "story-advance")

    def begin_chat_job(
        self,
        user_text: str,
        to: list[str] | None = None,
        action: str = "",
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        user_text = (user_text or "").strip()
        action = (action or "").strip()
        if not user_text and not action:
            raise ValueError("Empty message")
        if action and not self.actions_enabled:
            raise ValueError("Actions are disabled — turn on Actions in settings")
        if self.story is None:
            raise RuntimeError("Generate a story first")
        targets = list(to or [])

        def worker() -> None:
            for msg in self.chat_stream(user_text, to=targets, action=action):
                self._publish_live(msg)

        return self._start_live_job(worker, "story-chat")

    def chat(
        self,
        user_text: str,
        character: str | None = None,
        to: list[str] | None = None,
        action: str = "",
    ) -> dict[str, Any]:
        """Player line in the active story (cued or interrupt). Requires to[]."""
        return self._collect_stream(
            self.chat_stream(user_text, character=character, to=to, action=action)
        )

    def chat_stream(
        self,
        user_text: str,
        character: str | None = None,
        to: list[str] | None = None,
        action: str = "",
    ) -> Iterator[dict[str, Any]]:
        """SSE-friendly player turn: stream NPC reaction lines as they generate."""
        _ = character  # unused; roster comes from the story cast
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        user_text = (user_text or "").strip()
        action = (action or "").strip()
        if not user_text and not action:
            raise ValueError("Empty message")
        with self._lock:
            if self.story is None:
                raise RuntimeError("Generate a story first")
            self._flush_followup()
            yield from self._story_player_turn_stream(
                user_text, to or [], action=action
            )
        self._spawn_followup()

    @staticmethod
    def _collect_stream(gen: Iterator[dict[str, Any]]) -> dict[str, Any]:
        """Fold a stream into the old batch {events, story, status, ...} shape."""
        events: list[dict[str, Any]] = []
        final: dict[str, Any] = {}
        for msg in gen:
            kind = msg.get("type")
            if kind == "event":
                events.append(msg["event"])
            elif kind == "stream_end":
                events.append(msg["event"])
            elif kind == "done":
                final = {k: v for k, v in msg.items() if k != "type"}
            elif kind == "error":
                raise RuntimeError(str(msg.get("message") or "stream error"))
        out = dict(final)
        out["events"] = events
        return out

    # --- Story mode ---------------------------------------------------------

    def _seed_story_memories(self, state: StoryState) -> None:
        names = state.cast_names()
        set_speakable_characters(names)
        for name in names:
            ensure_character(self.memory_store, name)
        ensure_character(self.memory_store, NARRATOR_KEY)
        ensure_character(self.memory_store, PLAYER_KEY)

        lang = normalize_lang(state.language)
        # Seed short identity/stance lines only — full persona lives in the system prompt,
        # not in long-term memory (avoids persona/memory overlap in the debug panel).
        if lang == "en":
            seeds: list[tuple[str, str]] = [
                (NARRATOR_KEY, f"Story background: {state.background[:400]}"),
                (
                    PLAYER_KEY,
                    f"I am {state.player.get('name')}, {state.player.get('identity')}. "
                    f"Goal: {state.player.get('goal')}",
                ),
            ]
            for card in state.character_dict.values():
                seeds.append(
                    (
                        card.name,
                        f"I am {card.name} ({card.slot}). In this story: {card.description}",
                    )
                )
        else:
            seeds = [
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
                        f"我是{card.name}（{card.slot}）。在本故事中：{card.description}",
                    )
                )
        for who, text in seeds:
            add_memory(self.memory_store, who, text)
        for card in state.character_dict.values():
            attitudes = (card.persona_profile or {}).get("attitudes") or {}
            if isinstance(attitudes, dict) and attitudes:
                seed_attitude_memories(
                    self.memory_store,
                    holder=card.name,
                    attitudes={str(k): str(v) for k, v in attitudes.items()},
                    retriever=None,  # rebuild once below
                )
        save_memory(self.memory_store)
        if self.retriever is not None:
            self.retriever.rebuild(self.memory_store)
        if names:
            self.active_character = names[0]

    def _recall_query(self, state: StoryState, fallback: str = "") -> str:
        """Prefer current-scene dialogue so retrieval can surface older-scene memories."""
        heard = format_scene_log(
            scene_entries(state), 16, language=state.language
        ).strip()
        if heard and heard != story_prompts(state.language)["label_no_dialogue"]:
            return heard
        return (
            (fallback or "").strip()
            or (state.current_narration or "").strip()
            or state.background[:200]
        )

    def _remember_narration(self, state: StoryState, text: str) -> None:
        """Narrator LTM beat + queued biased scene impressions for present cast."""
        record_dialogue_beat(
            self.memory_store,
            speaker=NARRATOR_KEY,
            text=text,
            retriever=self.retriever,
        )
        present = list(state.present_names())
        narration = text

        def _impressions() -> None:
            briefs = {
                name: (
                    state.character_dict[name].acting_brief(state.language)
                    if name in state.character_dict
                    else ""
                )
                for name in present
            }
            record_present_scene_views(
                self.model,
                self.tokenizer,
                self.memory_store,
                narration=narration,
                present_names=present,
                retriever=self.retriever,
                persona_briefs=briefs,
            )

        self._queue_followup(_impressions)

    def _character_recall(
        self, state: StoryState, card_name: str, fallback: str = ""
    ) -> list[dict]:
        """Embed-rank that character's personal memories; keep top ~6–7."""
        if self.retriever is None:
            return []
        query = self._recall_query(state, fallback)
        return self.retriever.retrieve(
            query,
            top_k=RETRIEVE_TOP_K,
            characters=[card_name],
        )

    def _attitude_roster_lines(self, state: StoryState, holder: str) -> list[str]:
        lines: list[str] = []
        pname = str(state.player.get("name") or "").strip() or "Player"
        lines.append(
            f"- Player (in-story: {pname}): {state.player.get('identity') or ''} / "
            f"{state.player.get('goal') or ''}"
        )
        for card in state.character_dict.values():
            if card.name == holder:
                continue
            lines.append(f"- {card.name} ({card.slot}): {card.description}")
        return lines

    def _resolve_attitude_toward(self, state: StoryState, raw: str) -> str | None:
        raw = (raw or "").strip()
        if not raw:
            return None
        if raw in {PLAYER_KEY, "玩家", "Player", "player"}:
            return PLAYER_KEY
        card = lookup_character(state.character_dict, raw)
        if card:
            return card.name
        # Exact cast / player-name match
        if raw in state.character_dict:
            return raw
        pname = str(state.player.get("name") or "").strip()
        if pname and raw == pname:
            return PLAYER_KEY
        return None

    def _patch_card_attitude(
        self, state: StoryState, holder: str, toward: str, attitude: str
    ) -> None:
        card = state.character_dict.get(holder)
        if card is None:
            return
        prof = dict(card.persona_profile or {})
        attitudes = dict(prof.get("attitudes") or {})
        attitudes[toward] = attitude
        prof["attitudes"] = attitudes
        card.persona_profile = prof

    def _refresh_attitudes_for(
        self, state: StoryState, holder: str, recent_text: str
    ) -> None:
        """Extract attitude deltas; upsert LTM via same/contradict; patch persona map."""
        holder = (holder or "").strip()
        if not holder or not is_speakable(holder):
            return
        card = state.character_dict.get(holder)
        current = dict((card.persona_profile or {}).get("attitudes") or {}) if card else {}
        persona_brief = card.acting_brief(state.language) if card else ""
        updates = extract_attitude_updates(
            self.model,
            self.tokenizer,
            holder=holder,
            current_attitudes=current,
            roster_lines=self._attitude_roster_lines(state, holder),
            recent_text=recent_text,
        )
        for item in updates:
            toward = self._resolve_attitude_toward(state, item.get("toward") or "")
            attitude = str(item.get("attitude") or "").strip()
            if not toward or not attitude or toward == holder:
                continue
            result = upsert_attitude_memory(
                self.model,
                self.tokenizer,
                self.memory_store,
                holder=holder,
                toward=toward,
                attitude=attitude,
                retriever=self.retriever,
                stimulus_text=recent_text,
                persona_brief=persona_brief,
            )
            if result in {"added", "replaced"}:
                self._patch_card_attitude(state, holder, toward, attitude)

    def _remember_line(
        self,
        *,
        speaker: str,
        text: str,
        to: list[str] | None = None,
        extract_for: str | None = None,
        extract_user: str = "",
        extract_assistant: str = "",
        history: list[dict] | None = None,
    ) -> None:
        """
        Scene messages stay in the scene log (prompt block), not LTM.
        Store biased impressions + selective attitude updates for the character.
        """
        _ = (speaker, to)  # transcript is already in scene_log / scene-messages block
        who = (extract_for or "").strip()
        if not who or not is_speakable(who):
            return
        state = self.story
        stimulus = (extract_user or text or "").strip()
        assistant = (extract_assistant or "").strip()

        def _extract() -> None:
            persona_brief = ""
            if state is not None and who in state.character_dict:
                persona_brief = state.character_dict[who].acting_brief(state.language)
            extract_and_store(
                self.model,
                self.tokenizer,
                self.memory_store,
                stimulus,
                assistant,
                retriever=self.retriever,
                history=history or [],
                character=who,
                skip_gate=True,
                character_only=True,
                persona_brief=persona_brief,
            )
            if state is not None:
                blob = stimulus if not assistant else f"{stimulus}\n{assistant}"
                self._refresh_attitudes_for(state, who, blob)

        self._queue_followup(_extract)

    def generate_story(
        self, theme: str | None = None, *, language: str = "zh"
    ) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        with self._lock:
            lang = normalize_lang(language)
            set_memory_language(lang)
            # Fresh story → wipe prior cast/scene memories (no cross-story bleed).
            self._pending_followup = None
            self.memory_store = wipe_memory()
            if self.retriever is not None:
                self.retriever.rebuild(self.memory_store)
            state = generate_story_setup(
                self.model, self.tokenizer, theme=theme, language=lang
            )
            state.actions_enabled = self.actions_enabled
            self._seed_story_memories(state)
            self.story = state
            self.history = []
            self.prompt_log = []
            self.waiting_cleared()
            return {"story": state.public_dict(), "status": self.status()}

    def waiting_cleared(self) -> None:
        if self.story:
            self.story.waiting_for_player = False

    def start_intro(self) -> dict[str, Any]:
        return self._collect_stream(self.start_intro_stream())

    def start_intro_stream(self) -> Iterator[dict[str, Any]]:
        """Scene open + narrator only; client interrupt window then auto-advance."""
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.story is None:
            raise RuntimeError("Generate a story first")
        with self._lock:
            self._flush_followup()
            state = self.story
            if state.story_complete:
                raise RuntimeError("Story already complete")
            present = choose_present_cast(self.model, self.tokenizer, state)
            yield {
                "type": "event",
                "event": {
                    "type": "scene_start",
                    "scene_index": state.scene_index,
                    "max_scenes": state.max_scenes,
                    "present": list(present),
                },
            }
            text = ""
            dbg, chunks = prepare_narrator_intro(self.model, self.tokenizer, state)
            for msg in self._stream_prose_unlocked(
                event_type="narrator",
                speaker=NARRATOR_KEY,
                to=[],
                kind="narrator",
                chunks=chunks,
                extra={"debug_prompt": self._record_prompt(dbg)},
            ):
                if msg["type"] == "stream_end":
                    text = msg["event"].get("text") or ""
                yield msg
            state.current_narration = text
            self._remember_narration(state, text)
            yield from self._emit_done(auto_continue=True)
        self._spawn_followup()

    def advance_scene(self) -> dict[str, Any]:
        return self._collect_stream(self.advance_scene_stream())

    def advance_scene_stream(self) -> Iterator[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Engine not loaded")
        if self.story is None:
            raise RuntimeError("Generate a story first")
        with self._lock:
            self._flush_followup()
            state = self.story
            if state.story_complete or state.waiting_for_player:
                yield from self._emit_done(auto_continue=False)
            elif state.pending_scene_bridge:
                yield from self._run_pending_scene_bridge_unlocked()
                yield from self._emit_done(auto_continue=True)
            elif state.pending_reactions:
                yield from self._run_one_pending_reaction_unlocked()
                yield from self._emit_done(auto_continue=True)
            else:
                yield from self._advance_once_stream_unlocked()
                yield from self._emit_done(
                    auto_continue=not state.waiting_for_player
                    and not state.story_complete
                )
        self._spawn_followup()

    def _emit_done(self, *, auto_continue: bool) -> Iterator[dict[str, Any]]:
        state = self.story
        assert state is not None
        if state.waiting_for_player or state.story_complete:
            auto_continue = False
        yield {
            "type": "done",
            "story": state.public_dict(),
            "status": self.status(),
            "auto_continue": bool(auto_continue),
            "beat_seconds": LINE_BEAT_SECONDS,
            "waiting_for_player": bool(state.waiting_for_player),
            "ended": bool(state.story_complete),
            "facts_count": count_memories(self.memory_store),
            "session_id": self.session_id,
            "character": self.active_character,
        }

    def _queue_followup(self, work: Callable[[], None] | None) -> None:
        """Defer memory/prep into the interrupt window (lock released after done)."""
        if work is None:
            return
        prev = self._pending_followup
        if prev is None:
            self._pending_followup = work
            return

        def _chained() -> None:
            try:
                prev()
            except Exception as e:
                print(f"[engine] followup work failed: {e}")
            work()

        self._pending_followup = _chained

    def _flush_followup(self) -> None:
        work = self._pending_followup
        self._pending_followup = None
        if work is None:
            return
        try:
            work()
        except Exception as e:
            print(f"[engine] followup work failed: {e}")

    def _spawn_followup(self) -> None:
        """Run queued memory in the background during the client's interrupt window."""
        if self._pending_followup is None:
            return

        def _run() -> None:
            with self._lock:
                self._flush_followup()

        threading.Thread(target=_run, daemon=True, name="story-followup").start()

    def _stream_prose_unlocked(
        self,
        *,
        event_type: str,
        speaker: str,
        to: list[str],
        kind: str,
        chunks: Iterator[str],
        extra: dict[str, Any] | None = None,
        action: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Stream one spoken/narrated line; caller must append_log via stream_end text."""
        state = self.story
        assert state is not None
        extra = dict(extra or {})
        action = (action or "").strip()
        draft = {
            "type": event_type,
            "speaker": speaker,
            "to": list(to),
            "text": "",
            "action": action,
            "kind": kind,
            **extra,
        }
        yield {"type": "stream_start", "event": draft}
        parts: list[str] = []
        for delta in chunks:
            if not delta:
                continue
            parts.append(delta)
            yield {"type": "stream_delta", "delta": delta}
        text = "".join(parts).strip()
        entry = append_log(
            state, speaker=speaker, text=text, to=to, kind=kind, action=action
        )
        full = {"type": event_type, **entry, **extra}
        yield {"type": "stream_end", "event": full}

    def _emit_character_beat_unlocked(
        self,
        *,
        card,
        to: list[str],
        hits: list[dict] | None = None,
        player_turn_mode: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """NPC beat: stream dialogue, or JSON line+action when actions_enabled."""
        state = self.story
        assert state is not None
        extra = dict(extra or {})
        if state.actions_enabled:
            line, action, dbg = generate_character_beat(
                self.model,
                self.tokenizer,
                state,
                card,
                to,
                recalled_hits=hits,
                player_turn_mode=player_turn_mode,
            )
            recorded = self._record_prompt(dbg)

            def _chunks() -> Iterator[str]:
                if line:
                    yield line

            yield from self._stream_prose_unlocked(
                event_type="dialogue",
                speaker=card.name,
                to=to,
                kind="dialogue",
                chunks=_chunks(),
                action=action,
                extra={
                    **extra,
                    "recalled": hits or [],
                    "debug_prompt": recorded,
                },
            )
            return

        dbg, chunks = prepare_character_line(
            self.model,
            self.tokenizer,
            state,
            card,
            to,
            recalled_hits=hits,
            player_turn_mode=player_turn_mode,
        )
        yield from self._stream_prose_unlocked(
            event_type="dialogue",
            speaker=card.name,
            to=to,
            kind="dialogue",
            chunks=chunks,
            extra={
                **extra,
                "recalled": hits or [],
                "debug_prompt": self._record_prompt(dbg),
            },
        )

    def _queue_scene_end_if_needed(self, cue_end: bool) -> Iterator[dict[str, Any]]:
        """Mark bridge for next advance, or finish the story immediately."""
        state = self.story
        assert state is not None
        dcount = state.dialogue_count_in_scene
        hit_max = dcount >= state.scene_dialogue_max
        if hit_max:
            allow = True
        elif cue_end and dcount >= state.scene_dialogue_min:
            allow = True
        else:
            allow = False
        if not allow:
            return
        # Scene-end attitude pass for everyone who was present (same/contradict upsert).
        present = list(state.present_names())
        recent = format_scene_log(
            scene_entries(state), 40, language=state.language
        )

        def _scene_attitudes_and_organize() -> None:
            for who in present:
                if is_speakable(who):
                    self._refresh_attitudes_for(state, who, recent)
            # Rizz-style: after the scene, resolve same/contradict by higher certainty.
            organize_memory(
                self.model,
                self.tokenizer,
                self.memory_store,
                retriever=self.retriever,
            )

        self._queue_followup(_scene_attitudes_and_organize)
        if state.scene_index >= state.max_scenes:
            state.story_complete = True
            state.pending_scene_bridge = False
            yield {
                "type": "event",
                "event": {
                    "type": "story_complete",
                    "text": story_prompts(state.language)["story_complete"],
                },
            }
            return
        state.pending_scene_bridge = True

    def _run_pending_scene_bridge_unlocked(self) -> Iterator[dict[str, Any]]:
        state = self.story
        assert state is not None
        state.pending_scene_bridge = False
        state.scene_index += 1
        state.message_count_in_scene = 0
        state.dialogue_count_in_scene = 0
        state.last_player_mode = ""
        present = choose_present_cast(self.model, self.tokenizer, state)
        yield {
            "type": "event",
            "event": {
                "type": "scene_start",
                "scene_index": state.scene_index,
                "max_scenes": state.max_scenes,
                "present": list(present),
            },
        }
        text = ""
        dbg, chunks = prepare_narrator_bridge(self.model, self.tokenizer, state)
        for msg in self._stream_prose_unlocked(
            event_type="narrator",
            speaker=NARRATOR_KEY,
            to=[],
            kind="narrator",
            chunks=chunks,
            extra={"debug_prompt": self._record_prompt(dbg)},
        ):
            if msg["type"] == "stream_end":
                text = msg["event"].get("text") or ""
            yield msg
        state.current_narration = text
        self._remember_narration(state, text)

    def _run_one_pending_reaction_unlocked(self) -> Iterator[dict[str, Any]]:
        state = self.story
        assert state is not None
        if not state.pending_reactions:
            return
        item = state.pending_reactions.pop(0)
        who = str(item.get("name") or "")
        mode = str(item.get("mode") or "cued")
        user_text = str(item.get("user_text") or "")
        card = lookup_character(state.character_dict, who)
        if card is None:
            return
        hits = self._character_recall(state, card.name, user_text)
        line = ""
        action = ""
        for msg in self._emit_character_beat_unlocked(
            card=card,
            to=[PLAYER_KEY],
            hits=hits,
            player_turn_mode=mode,
            extra={"player_mode": mode},
        ):
            if msg["type"] == "stream_end":
                line = msg["event"].get("text") or ""
                action = msg["event"].get("action") or ""
            yield msg
        remember_text = line if line else f"*{(action or '').strip()}*"
        self.active_character = card.name
        self._remember_line(
            speaker=card.name,
            text=remember_text,
            to=[PLAYER_KEY],
            extract_for=card.name,
            extract_user=user_text,
            extract_assistant=remember_text,
            history=self.history,
        )
        yield from self._queue_scene_end_if_needed(
            state.dialogue_count_in_scene >= state.scene_dialogue_max
        )

    def _advance_once_stream_unlocked(
        self, *, force_cue: bool = False
    ) -> Iterator[dict[str, Any]]:
        _ = force_cue
        state = self.story
        assert state is not None

        # Director asked to end before speaking — queue bridge; no mid-stream chain.
        cue = narrator_cue(self.model, self.tokenizer, state)
        yield {"type": "event", "event": {"type": "cue", **cue}}

        if cue.get("end_scene") and state.dialogue_count_in_scene > 0:
            yield from self._queue_scene_end_if_needed(True)
            if state.story_complete or state.pending_scene_bridge:
                return

        speaker = str(cue.get("speaker") or PLAYER_KEY)
        to = list(cue.get("to") or [])

        if speaker == PLAYER_KEY:
            state.waiting_for_player = True
            state.suggested_to = [
                n for n in to if n in state.cast_names()
            ] or list(state.cast_names()[:1])
            yield {
                "type": "event",
                "event": {
                    "type": "wait_player",
                    "suggested_to": list(state.suggested_to),
                    "reason": cue.get("reason"),
                },
            }
            return

        if speaker == NARRATOR_KEY:
            text = ""
            dbg, chunks = prepare_narrator_continue(
                self.model, self.tokenizer, state
            )
            for msg in self._stream_prose_unlocked(
                event_type="narrator",
                speaker=NARRATOR_KEY,
                to=[],
                kind="narrator",
                chunks=chunks,
                extra={"debug_prompt": self._record_prompt(dbg)},
            ):
                if msg["type"] == "stream_end":
                    text = msg["event"].get("text") or ""
                yield msg
            state.current_narration = text
            self._remember_narration(state, text)
            return

        card = lookup_character(state.character_dict, speaker)
        if card is None:
            state.waiting_for_player = True
            state.suggested_to = list(state.cast_names()[:1])
            yield {
                "type": "event",
                "event": {
                    "type": "wait_player",
                    "suggested_to": list(state.suggested_to),
                    "reason": (
                        f"调用失败：名单中找不到「{cue.get('raw_speaker') or speaker}」"
                        f"（已解析为 {speaker}），改由玩家行动"
                    ),
                    "stage": "lookup",
                },
            }
            return

        to = clamp_addressees(to, state)
        hits = self._character_recall(state, card.name)
        line = ""
        action = ""
        for msg in self._emit_character_beat_unlocked(
            card=card,
            to=to,
            hits=hits,
        ):
            if msg["type"] == "stream_end":
                line = msg["event"].get("text") or ""
                action = msg["event"].get("action") or ""
            yield msg
        remember_text = line if line else f"*{(action or '').strip()}*"
        self.active_character = card.name
        self._remember_line(
            speaker=card.name,
            text=remember_text,
            to=to,
            extract_for=card.name,
            extract_user=self._recall_query(state),
            extract_assistant=remember_text,
        )
        yield from self._queue_scene_end_if_needed(
            state.dialogue_count_in_scene >= state.scene_dialogue_max
            or bool(cue.get("end_scene"))
        )

    def _story_player_turn_stream(
        self, user_text: str, to: list[str], *, action: str = ""
    ) -> Iterator[dict[str, Any]]:
        state = self.story
        assert state is not None
        if state.story_complete:
            raise RuntimeError("Story already complete")

        user_text = (user_text or "").strip()
        action = (action or "").strip()
        if action and not state.actions_enabled:
            raise ValueError("Actions are disabled")
        if not user_text and not action:
            raise ValueError("Empty message")

        addressees = clamp_addressees(to, state)
        npc_targets = [n for n in addressees if n in state.character_dict]
        if not npc_targets:
            raise ValueError("Select at least one character to speak to")

        was_waiting = bool(state.waiting_for_player)
        mode = "cued" if was_waiting else "interrupt"

        if action:
            check = validate_player_action(
                self.model, self.tokenizer, state, action
            )
            if not check.get("ok"):
                reason = str(check.get("reason") or "").strip()
                hint = str(check.get("hint") or "").strip()
                lang = normalize_lang(state.language)
                if lang == "en":
                    msg = f"Action rejected: {reason}"
                    if hint:
                        msg += f" Hint: {hint}"
                else:
                    msg = f"动作未通过：{reason}"
                    if hint:
                        msg += f" 提示：{hint}"
                yield {
                    "type": "event",
                    "event": {
                        "type": "action_rejected",
                        "text": msg,
                        "reason": reason,
                        "hint": hint,
                        "action": action,
                    },
                }
                # Keep cued wait / interrupt window state unchanged.
                state.waiting_for_player = was_waiting
                yield from self._emit_done(auto_continue=False)
                return

        # Interrupting cancels queued NPC reactions / bridge from the prior beat.
        state.pending_reactions = []
        state.pending_scene_bridge = False

        state.last_player_mode = mode
        state.waiting_for_player = False
        entry = append_log(
            state,
            speaker=PLAYER_KEY,
            text=user_text,
            to=npc_targets,
            kind="dialogue",
            action=action if state.actions_enabled else "",
        )
        entry["player_mode"] = mode
        stimulus = user_text
        if action:
            stimulus = (
                f"[Player action: {action}] {user_text}".strip()
                if user_text
                else f"[Player action: {action}]"
            )
        log_turn(stimulus, "", self.session_id)
        self.history.append({"role": "user", "content": stimulus})

        self.active_character = npc_targets[0]
        yield {
            "type": "event",
            "event": {"type": "player", **entry, "player_mode": mode},
        }

        # Player line stays in scene_log (direct scene knowledge). NPCs only store
        # biased personal impressions via extract — not a transcript copy.
        reactors = npc_targets[:2]

        def _player_memory() -> None:
            for who in npc_targets:
                persona_brief = ""
                if who in state.character_dict:
                    persona_brief = state.character_dict[who].acting_brief(
                        state.language
                    )
                extract_and_store(
                    self.model,
                    self.tokenizer,
                    self.memory_store,
                    stimulus,
                    "",
                    retriever=self.retriever,
                    history=self.history,
                    character=who,
                    skip_gate=True,
                    character_only=True,
                    persona_brief=persona_brief,
                )
                self._refresh_attitudes_for(state, who, stimulus)

        self._queue_followup(_player_memory)
        state.pending_reactions = [
            {"name": who, "mode": mode, "user_text": stimulus}
            for who in reactors
            if state.dialogue_count_in_scene < state.scene_dialogue_max
        ]
        # Cap queue if scene already at max after the player line itself.
        if state.dialogue_count_in_scene >= state.scene_dialogue_max:
            state.pending_reactions = []
            yield from self._queue_scene_end_if_needed(True)

        yield from self._emit_done(
            auto_continue=bool(state.pending_reactions or state.pending_scene_bridge)
        )


# Backward-compatible alias
MiaEngine = TextGameEngine

