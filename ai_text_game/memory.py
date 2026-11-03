"""
Multi-perspective long-term memory: character-keyed fact lists with certainty,
importance gate, POV rewriters, bge-m3 retrieval (CPU), scene-end organize.
Certainty (from Rizz-the-AI): evidence strength + context match + persona receptivity;
on same/contradict, keep the higher-certainty entry.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent / "data"
MEMORY_PATH = DATA_DIR / "long_term_memory.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]
def _resolve_models_dir() -> Path:
    for key in ("AI_TEXT_GAME_MODELS_DIR", "MIA_MODELS_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser()
    return _REPO_ROOT / "models"


_MODELS_DIR = _resolve_models_dir()
EMBED_MODEL_ID = str(_MODELS_DIR / "bge-m3")
EMBED_DEVICE = "cpu"
# How many personal memory lines to inject into a character turn (embedding rank).
RETRIEVE_TOP_K = 7
RETRIEVE_MIN_SCORE = 0.25
RETRIEVE_CANDIDATE_MULT = 3

ORGANIZE_CANDIDATE_MIN = 0.45
NEAR_DUP_MIN = 0.88

GATE_MAX_NEW_TOKENS = 64
REWRITE_MAX_NEW_TOKENS = 192
CERTAINTY_MAX_NEW_TOKENS = 64
ORGANIZE_MAX_NEW_TOKENS = 96
MAX_FACTS_PER_TURN = 5
MEMORY_CONTEXT_MESSAGES = 5

# Fixed perspective keys + speakable NPC roster (filled after story generation).
NARRATOR_KEY = "Narrator"
PLAYER_KEY = "Player"
# Legacy girlfriend-chat default; kept only so old memory JSON still loads.
LEGACY_DEFAULT_CHARACTER = "Mia"
DEFAULT_CHARACTER = LEGACY_DEFAULT_CHARACTER  # alias used by older call sites
SPEAKABLE_CHARACTERS: list[str] = []
SESSION_NEW_KEY = "_session_new"  # in-RAM only; never written to JSON
_ACTIVE_LANGUAGE = "zh"


def set_memory_language(lang: str) -> None:
    """Story language drives memory gate/rewrite prompts."""
    global _ACTIVE_LANGUAGE
    from .prompts import normalize_lang

    _ACTIVE_LANGUAGE = normalize_lang(lang)


def _memory_prompt_pack() -> dict[str, str]:
    from .prompts import memory_prompts

    return memory_prompts(_ACTIVE_LANGUAGE)

_CALL_PREF_RE = re.compile(
    r"称呼|叫作|叫我|叫他|叫她|昵称|宝宝|老公|老婆|亲爱的|"
    r"call(?:ed|ing)?\s+me|nickname|prefer(?:s|red)?\s+to\s+be\s+called",
    re.I,
)
_LIKE_PREF_RE = re.compile(
    r"喜欢|不喜欢|爱吃|讨厌|偏好|喜好|\blikes?\b|\bdislikes?\b|\bprefer(?:s|ence)?\b",
    re.I,
)


def list_speakable_characters() -> list[str]:
    return list(SPEAKABLE_CHARACTERS)


def set_speakable_characters(names: list[str]) -> list[str]:
    """Replace the NPC roster (used after story generation)."""
    global SPEAKABLE_CHARACTERS
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = (raw or "").strip()
        if not name or name in (NARRATOR_KEY, PLAYER_KEY):
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    SPEAKABLE_CHARACTERS = cleaned
    return list(SPEAKABLE_CHARACTERS)


def is_speakable(name: str) -> bool:
    return (name or "").strip() in SPEAKABLE_CHARACTERS


def turn_memory_keys(character: str) -> list[str]:
    """Buckets recalled/written on a user-initiated turn with this interlocutor."""
    return [NARRATOR_KEY, PLAYER_KEY, character]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


def _entry_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("text") or "").strip()
    return ""


def _entry_certainty(item: object) -> float:
    if isinstance(item, dict):
        return _clamp(item.get("certainty", 0.5))
    return 0.5


def _make_entry(
    text: str, *, certainty: float = 0.5, session_new: bool = True
) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "text": (text or "").strip(),
        "certainty": _clamp(certainty),
        "session_new": bool(session_new),
    }


def _normalize_entry(item: object) -> dict | None:
    if isinstance(item, dict):
        text = _entry_text(item)
        if not text:
            return None
        out = dict(item)
        out["text"] = text
        out["id"] = str(out.get("id") or uuid.uuid4())
        out["certainty"] = _clamp(out.get("certainty", 0.5))
        out["session_new"] = bool(out.get("session_new", False))
        return out
    if isinstance(item, str) and item.strip():
        return _make_entry(item.strip(), certainty=0.5, session_new=False)
    return None


def _empty_store() -> dict:
    store: dict = {
        NARRATOR_KEY: [],
        PLAYER_KEY: [],
    }
    for name in SPEAKABLE_CHARACTERS:
        store.setdefault(name, [])
    return store


def ensure_character(store: dict, name: str) -> list:
    """Return the character's fact-entry list (dicts with text + certainty)."""
    name = (name or "").strip()
    if not name or name.startswith("_"):
        raise ValueError("Invalid memory key")
    if not isinstance(store.get(name), list):
        store[name] = []
    cleaned: list[dict] = []
    for item in store[name]:
        entry = _normalize_entry(item)
        if entry is not None:
            cleaned.append(entry)
    store[name] = cleaned
    return store[name]


def character_memories(store: dict, name: str) -> list[str]:
    return [_entry_text(e) for e in ensure_character(store, name)]


def character_entries(store: dict, name: str) -> list[dict]:
    return list(ensure_character(store, name))


def iter_memories(store: dict):
    """Yield (character, index, text) for public keys."""
    for key, values in store.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if not isinstance(values, list):
            continue
        for i, item in enumerate(values):
            text = _entry_text(item)
            if text:
                yield key, i, text


def count_memories(store: dict) -> int:
    return sum(1 for _ in iter_memories(store))


def public_memory_store(store: dict) -> dict:
    """JSON/API copy: fact dicts with text + certainty (legacy strings upgraded)."""
    out: dict[str, list] = {}
    for key, values in store.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if not isinstance(values, list):
            continue
        bucket: list[dict] = []
        for item in values:
            entry = _normalize_entry(item)
            if entry is not None:
                bucket.append(
                    {
                        "id": entry["id"],
                        "text": entry["text"],
                        "certainty": entry["certainty"],
                        "session_new": bool(entry.get("session_new", False)),
                    }
                )
        out[key] = bucket
    for required in (NARRATOR_KEY, PLAYER_KEY):
        out.setdefault(required, [])
    for name in SPEAKABLE_CHARACTERS:
        out.setdefault(name, [])
    return out


# Backward-compatible private alias used by save_memory.
_public_store = public_memory_store


def _migrate_legacy(data: dict) -> dict:
    """Old {'facts': [{text, ...}]} → Narrator texts + empty Player/Mia."""
    texts: list[str] = []
    for f in data.get("facts") or []:
        if isinstance(f, dict):
            t = (f.get("text") or "").strip()
            if t:
                texts.append(t)
        elif isinstance(f, str) and f.strip():
            texts.append(f.strip())
    store = _empty_store()
    # Preserve any already-valid character lists beside 'facts'.
    for key, values in data.items():
        if key == "facts" or not isinstance(key, str) or key.startswith("_"):
            continue
        if isinstance(values, list):
            ensure_character(store, key)
            for item in values:
                if isinstance(item, str) and item.strip():
                    store[key].append(item.strip())
                elif isinstance(item, dict) and (item.get("text") or "").strip():
                    store[key].append(item["text"].strip())
    seen = {_normalize(_entry_text(t)) for t in store[NARRATOR_KEY]}
    for t in texts:
        n = _normalize(t)
        if n and n not in seen:
            store[NARRATOR_KEY].append(
                _make_entry(t, certainty=0.5, session_new=False)
            )
            seen.add(n)
    # Coerce any leftover string appends from the loop above.
    for key in list(store.keys()):
        if isinstance(store.get(key), list):
            ensure_character(store, key)
    return store


def _normalize_store(data: dict) -> dict:
    if not isinstance(data, dict):
        return _empty_store()
    if "facts" in data and isinstance(data["facts"], list):
        # Legacy if facts hold objects, or empty facts with no other string lists.
        facts = data["facts"]
        looks_legacy = (
            not facts
            or all(isinstance(f, dict) for f in facts)
            or any(isinstance(f, dict) for f in facts)
        )
        if looks_legacy:
            return _migrate_legacy(data)
    store = _empty_store()
    for key, values in data.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if isinstance(values, list):
            ensure_character(store, key)
            store[key] = []
            for item in values:
                entry = _normalize_entry(item)
                if entry is not None:
                    store[key].append(entry)
    return store


def load_memory(path: Path = MEMORY_PATH) -> dict:
    """Load long-term memory JSON; create empty multi-perspective store if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        store = _empty_store()
        save_memory(store, path)
        return store
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        store = _normalize_store(data)
        # Persist migration if shape changed from legacy.
        if isinstance(data, dict) and "facts" in data:
            save_memory(store, path)
        return store
    except (json.JSONDecodeError, OSError):
        return _empty_store()


def wipe_memory(path: Path = MEMORY_PATH) -> dict:
    """
    Clear all persisted memories (new story). Roster keys are empty until seed.
    """
    global SPEAKABLE_CHARACTERS
    SPEAKABLE_CHARACTERS = []
    store = {
        NARRATOR_KEY: [],
        PLAYER_KEY: [],
        SESSION_NEW_KEY: {},
    }
    save_memory(store, path)
    return store


def save_memory(store: dict, path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_public_store(store), f, ensure_ascii=False, indent=2)


def _session_new_map(store: dict) -> dict[str, list[str]]:
    raw = store.get(SESSION_NEW_KEY)
    if not isinstance(raw, dict):
        store[SESSION_NEW_KEY] = {}
        return store[SESSION_NEW_KEY]
    return raw


def mark_session_new(store: dict, character: str, text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    m = _session_new_map(store)
    bucket = m.setdefault(character, [])
    if text not in bucket:
        bucket.append(text)
    # Also flag the matching fact entry when present.
    for entry in ensure_character(store, character):
        if _normalize(_entry_text(entry)) == _normalize(text):
            entry["session_new"] = True


def format_facts_for_prompt(hits: list[dict] | list[str], max_facts: int = 7) -> str:
    """
    Personal biased-memory block for a character turn.
    Not the same as scene messages — those are short-term direct knowledge.
    """
    if not hits:
        return ""
    lines_out: list[str] = []
    for h in hits[:max_facts]:
        if isinstance(h, dict):
            text = (h.get("text") or "").strip()
        else:
            text = str(h).strip()
        if text:
            lines_out.append(f"- {text}")
    if not lines_out:
        return ""
    if _ACTIVE_LANGUAGE == "en":
        header = (
            "Personal memories (biased takes you formed earlier — not a transcript; "
            "use naturally, do not recite):"
        )
    else:
        header = "个人记忆（你先前形成的主观看法/印象，不是逐字记录；可自然用到，不要照念）："
    return header + "\n" + "\n".join(lines_out)


def memory_id(character: str, index: int) -> str:
    return f"{character}::{index}"


def parse_memory_id(mid: str) -> tuple[str, int] | None:
    if "::" not in (mid or ""):
        return None
    char, _, idx = mid.partition("::")
    try:
        return char, int(idx)
    except ValueError:
        return None


class MemoryRetriever:
    """Dense retrieval over multi-perspective string memories (CPU)."""

    def __init__(
        self,
        model_id: str = EMBED_MODEL_ID,
        device: str = EMBED_DEVICE,
    ):
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedder: {model_id} (device={device})")
        self.device = device
        self.model = SentenceTransformer(model_id, device=device)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._characters: list[str] = []
        self._matrix: torch.Tensor | None = None

    def rebuild(self, store: dict) -> None:
        self._ids = []
        self._texts = []
        self._characters = []
        texts: list[str] = []
        for character, index, text in iter_memories(store):
            self._ids.append(memory_id(character, index))
            self._texts.append(text)
            self._characters.append(character)
            texts.append(text)
        if not texts:
            self._matrix = None
            print("[memory] retriever index empty")
            return
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._matrix = torch.from_numpy(emb)
        print(f"[memory] retriever indexed {len(texts)} memor(ies) on {self.device}")

    def add_memory_embedding(self, character: str, text: str, index: int) -> None:
        text = (text or "").strip()
        if not text:
            return
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        row = torch.from_numpy(emb)
        self._ids.append(memory_id(character, index))
        self._texts.append(text)
        self._characters.append(character)
        if self._matrix is None:
            self._matrix = row
        else:
            self._matrix = torch.cat([self._matrix, row], dim=0)

    def encode_one(self, text: str) -> torch.Tensor | None:
        text = (text or "").strip()
        if not text:
            return None
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return torch.from_numpy(emb)

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVE_TOP_K,
        min_score: float = RETRIEVE_MIN_SCORE,
        characters: list[str] | None = None,
    ) -> list[dict]:
        if self._matrix is None or not self._texts or not query.strip():
            return []
        q_t = self.encode_one(query)
        if q_t is None:
            return []
        sims = (q_t @ self._matrix.T).squeeze(0)
        pool = min(sims.numel(), max(top_k * RETRIEVE_CANDIDATE_MULT, top_k))
        vals, idxs = torch.topk(sims, k=pool)
        allow = set(characters) if characters else None
        ranked: list[dict] = []
        for sim, idx in zip(vals.tolist(), idxs.tolist()):
            if sim < min_score:
                continue
            who = self._characters[idx]
            if allow is not None and who not in allow:
                continue
            ranked.append(
                {
                    "id": self._ids[idx],
                    "text": self._texts[idx],
                    "character": who,
                    "score": float(sim),
                }
            )
        ranked.sort(key=lambda h: h["score"], reverse=True)
        return ranked[:top_k]

    def retrieve_for_turn(
        self,
        query: str,
        character: str,
        top_k_per_bucket: int = RETRIEVE_TOP_K,
        min_score: float = RETRIEVE_MIN_SCORE,
    ) -> list[dict]:
        """Recall from Narrator + Player + interlocutor; merge and dedupe."""
        merged: list[dict] = []
        seen_text: set[str] = set()
        for key in turn_memory_keys(character):
            for h in self.retrieve(
                query, top_k=top_k_per_bucket, min_score=min_score, characters=[key]
            ):
                n = _normalize(h.get("text") or "")
                if not n or n in seen_text:
                    continue
                seen_text.add(n)
                merged.append(h)
        merged.sort(key=lambda h: h["score"], reverse=True)
        return merged

    def best_match(
        self,
        query: str,
        character: str,
        min_score: float = NEAR_DUP_MIN,
        exclude_ids: set[str] | None = None,
    ) -> dict | None:
        hits = self.retrieve(
            query, top_k=5, min_score=min_score, characters=[character]
        )
        exclude_ids = exclude_ids or set()
        for h in hits:
            if h["id"] not in exclude_ids:
                return h
        return None


def find_string_duplicate(store: dict, character: str, fact_text: str) -> int | None:
    """Return index of exact normalized match within character list, else None."""
    new_n = _normalize(fact_text)
    if not new_n:
        return None
    for i, text in enumerate(character_memories(store, character)):
        if _normalize(text) == new_n:
            return i
    return None


def score_evidence(
    model, tokenizer, stimulus_text: str, fact_text: str
) -> float:
    """How strongly the stimulus asserts this memory (0–1)."""
    mp = _memory_prompt_pack()
    if _ACTIVE_LANGUAGE == "en":
        blob = f"Stimulus:\n{stimulus_text}\n\nCandidate memory:\n{fact_text}"
    else:
        blob = f"刺激文本：\n{stimulus_text}\n\n待记内容：\n{fact_text}"
    obj = _generate_json(
        model, tokenizer, mp["evidence_score"], blob, CERTAINTY_MAX_NEW_TOKENS
    )
    if not obj:
        return 0.35
    try:
        return _clamp(float(obj.get("evidence")))
    except (TypeError, ValueError):
        return 0.35


def score_context_match(
    fact_text: str,
    recalled_hits: list[dict] | None,
    retriever: MemoryRetriever | None,
) -> float:
    """How well the new fact aligns with recalled long-term context (0–1)."""
    if not recalled_hits:
        return 0.3
    if retriever is None:
        scores = [float(h.get("score", 0)) for h in recalled_hits]
        return _clamp(sum(scores) / len(scores)) if scores else 0.3
    q = retriever.encode_one(fact_text)
    if q is None:
        return 0.3
    sims: list[float] = []
    for h in recalled_hits:
        t = (h.get("text") or "").strip()
        if not t:
            continue
        e = retriever.encode_one(t)
        if e is None:
            continue
        sims.append(float((q @ e.T).item()))
    if not sims:
        return 0.3
    return _clamp(sum(sims) / len(sims))


def score_persona_receptivity(
    model, tokenizer, persona_brief: str, fact_text: str
) -> float:
    """
    How readily this persona would internalize the impression (0–1).
    Gullible / desire-aligned → higher; skeptical / value-clash → lower.
    """
    persona_brief = (persona_brief or "").strip()
    if not persona_brief:
        return 0.45
    mp = _memory_prompt_pack()
    if _ACTIVE_LANGUAGE == "en":
        blob = f"Persona:\n{persona_brief}\n\nCandidate memory:\n{fact_text}"
    else:
        blob = f"人设：\n{persona_brief}\n\n待记内容：\n{fact_text}"
    obj = _generate_json(
        model, tokenizer, mp["persona_receptivity"], blob, CERTAINTY_MAX_NEW_TOKENS
    )
    if not obj:
        return 0.45
    try:
        return _clamp(float(obj.get("receptivity")))
    except (TypeError, ValueError):
        return 0.45


def compute_certainty(
    model,
    tokenizer,
    stimulus_text: str,
    fact_text: str,
    *,
    recalled_hits: list[dict] | None = None,
    retriever: MemoryRetriever | None = None,
    persona_brief: str = "",
) -> float:
    """
    New memories: evidence + context match + persona receptivity.
    certainty ≈ 0.05 + 0.25*evidence + 0.20*context + 0.25*persona  (≈0.05–0.75)
    """
    evidence = score_evidence(model, tokenizer, stimulus_text, fact_text)
    context = score_context_match(fact_text, recalled_hits, retriever)
    persona = score_persona_receptivity(model, tokenizer, persona_brief, fact_text)
    certainty = _clamp(
        0.05 + 0.25 * evidence + 0.20 * context + 0.25 * persona,
        0.05,
        0.75,
    )
    print(
        f"[memory] certainty: evidence={evidence:.2f} context={context:.2f} "
        f"persona={persona:.2f} -> {certainty:.2f}"
    )
    return certainty


def bump_certainty(entry: dict, *, amount: float = 0.08) -> float:
    old = _entry_certainty(entry)
    new = _clamp(old + amount * (1.0 - old), 0.05, 0.99)
    entry["certainty"] = new
    entry["session_new"] = True
    return new


def add_memory(
    store: dict,
    character: str,
    text: str,
    *,
    certainty: float = 0.5,
    session_new: bool = True,
) -> tuple[str, int] | None:
    """Append a fact memory. Returns (text, index) or None."""
    text = (text or "").strip()
    if not text:
        return None
    bucket = ensure_character(store, character)
    entry = _make_entry(text, certainty=certainty, session_new=session_new)
    bucket.append(entry)
    mark_session_new(store, character, text)
    return text, len(bucket) - 1


# Tagged attitude lines: "[attitude:Wang Li] warm ally" — one slot per target.
_ATTITUDE_RE = re.compile(
    r"^\[attitude:(?P<toward>[^\]]+)\]\s*(?P<take>.+)$",
    re.I | re.DOTALL,
)


def format_attitude_memory(toward: str, attitude: str) -> str:
    toward = (toward or "").strip()
    attitude = (attitude or "").strip()
    if toward in {"玩家"}:
        toward = PLAYER_KEY
    return f"[attitude:{toward}] {attitude}"


def parse_attitude_memory(text: str) -> tuple[str, str] | None:
    m = _ATTITUDE_RE.match((text or "").strip())
    if not m:
        return None
    toward = m.group("toward").strip()
    take = m.group("take").strip()
    if toward in {"玩家"}:
        toward = PLAYER_KEY
    if not toward or not take:
        return None
    return toward, take


def find_attitude_index(store: dict, character: str, toward: str) -> int | None:
    toward_n = _normalize(toward)
    if toward_n in {_normalize("玩家"), _normalize(PLAYER_KEY)}:
        toward_alts = {_normalize(PLAYER_KEY), _normalize("玩家")}
    else:
        toward_alts = {toward_n}
    for i, text in enumerate(character_memories(store, character)):
        parsed = parse_attitude_memory(text)
        if parsed and _normalize(parsed[0]) in toward_alts:
            return i
    return None


def replace_memory_at_index(
    store: dict,
    character: str,
    index: int,
    new_text: str,
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
    *,
    certainty: float | None = None,
) -> str | None:
    """Replace one fact in-place; rebuild retriever embeddings for that store."""
    new_text = (new_text or "").strip()
    bucket = ensure_character(store, character)
    if index < 0 or index >= len(bucket) or not new_text:
        return None
    old = bucket[index]
    c = (
        _clamp(certainty)
        if certainty is not None
        else _entry_certainty(old)
    )
    bucket[index] = _make_entry(new_text, certainty=c, session_new=True)
    mark_session_new(store, character, new_text)
    save_memory(store, path)
    if retriever is not None:
        retriever.rebuild(store)
    return new_text


def upsert_attitude_memory(
    model,
    tokenizer,
    store: dict,
    *,
    holder: str,
    toward: str,
    attitude: str,
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
    stimulus_text: str = "",
    persona_brief: str = "",
    recalled_hits: list[dict] | None = None,
) -> str:
    """
    Keep one attitude line per (holder → toward).
    Conflicts resolve by higher certainty (Rizz-style), not “always prefer new”.
    """
    holder = (holder or "").strip()
    toward = (toward or "").strip()
    attitude = (attitude or "").strip()
    if toward in {"玩家"}:
        toward = PLAYER_KEY
    if not holder or not toward or not attitude or not is_speakable(holder):
        return "skipped"
    line = format_attitude_memory(toward, attitude)
    stimulus = (stimulus_text or line).strip()
    new_c = compute_certainty(
        model,
        tokenizer,
        stimulus,
        line,
        recalled_hits=recalled_hits,
        retriever=retriever,
        persona_brief=persona_brief,
    )
    idx = find_attitude_index(store, holder, toward)
    if idx is None:
        saved = _store_one_string(
            store,
            holder,
            line,
            path,
            retriever,
            certainty=new_c,
            skip_scoring=True,
        )
        return "added" if saved else "skipped"

    bucket = ensure_character(store, holder)
    old_entry = bucket[idx]
    old_text = _entry_text(old_entry)
    old_parsed = parse_attitude_memory(old_text)
    old_take = old_parsed[1] if old_parsed else old_text
    old_c = _entry_certainty(old_entry)

    if _normalize(old_take) == _normalize(attitude):
        bump_certainty(old_entry, amount=0.06)
        mark_session_new(store, holder, old_text)
        save_memory(store, path)
        return "kept"

    rel = classify_relation(model, tokenizer, old_take, attitude)
    if rel == "same":
        bump_certainty(old_entry, amount=0.06)
        mark_session_new(store, holder, old_text)
        save_memory(store, path)
        return "kept"

    # contradict / unrelated about same target → keep higher certainty
    if new_c > old_c or (abs(new_c - old_c) < 1e-6 and rel == "contradict"):
        replaced = replace_memory_at_index(
            store,
            holder,
            idx,
            line,
            path=path,
            retriever=retriever,
            certainty=new_c,
        )
        print(
            f"[attitude] {holder} → {toward} ({rel}): "
            f"keep new c={new_c:.2f} over old c={old_c:.2f}"
        )
        return "replaced" if replaced else "skipped"

    bump_certainty(old_entry, amount=0.05)
    mark_session_new(store, holder, old_text)
    save_memory(store, path)
    print(
        f"[attitude] {holder} → {toward} ({rel}): "
        f"keep old c={old_c:.2f} over new c={new_c:.2f}"
    )
    return "kept"


def seed_attitude_memories(
    store: dict,
    *,
    holder: str,
    attitudes: dict[str, str],
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
) -> int:
    """Write initial persona attitudes into LTM at high baseline certainty."""
    n = 0
    for toward, take in (attitudes or {}).items():
        toward = str(toward or "").strip()
        take = str(take or "").strip()
        if toward in {"玩家"}:
            toward = PLAYER_KEY
        if not toward or not take:
            continue
        if find_attitude_index(store, holder, toward) is not None:
            continue
        line = format_attitude_memory(toward, take)
        if _store_one_string(
            store,
            holder,
            line,
            path,
            retriever,
            certainty=0.72,
            skip_scoring=True,
        ):
            n += 1
    return n


def extract_attitude_updates(
    model,
    tokenizer,
    *,
    holder: str,
    current_attitudes: dict[str, str],
    roster_lines: list[str],
    recent_text: str,
) -> list[dict[str, str]]:
    """Ask the model which attitudes changed; returns [{toward, attitude}, ...]."""
    holder = (holder or "").strip()
    recent_text = (recent_text or "").strip()
    if not holder or not recent_text:
        return []
    mp = _memory_prompt_pack()
    if _ACTIVE_LANGUAGE == "en":
        cur = "\n".join(
            f"- {k}: {v}" for k, v in (current_attitudes or {}).items()
        ) or "(none yet)"
        roster = "\n".join(roster_lines) or "(none)"
        user = (
            f"Character whose attitudes we update: {holder}\n"
            f"Current attitudes:\n{cur}\n\n"
            f"People they may refer to:\n{roster}\n\n"
            f"Recent content:\n{recent_text}\n\n"
            "List only meaningful attitude updates."
        )
    else:
        cur = "\n".join(
            f"- {k}：{v}" for k, v in (current_attitudes or {}).items()
        ) or "（尚无）"
        roster = "\n".join(roster_lines) or "（无）"
        user = (
            f"要更新态度的角色：{holder}\n"
            f"当前态度：\n{cur}\n\n"
            f"可能涉及的人物：\n{roster}\n\n"
            f"近期内容：\n{recent_text}\n\n"
            "只列出有实质变化的态度更新。"
        )
    obj = _generate_json(
        model, tokenizer, mp["attitude_update"], user, REWRITE_MAX_NEW_TOKENS
    ) or {}
    raw = obj.get("updates")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        toward = str(item.get("toward") or "").strip()
        attitude = str(item.get("attitude") or "").strip()
        if toward in {"玩家"}:
            toward = PLAYER_KEY
        if toward and attitude:
            out.append({"toward": toward, "attitude": attitude})
    return out


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _generate_json(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
) -> dict | None:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(model, "create_chat_completion"):
        out = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        reply = (out["choices"][0]["message"].get("content") or "").strip()
        return _extract_json_object(reply)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _extract_json_object(reply)


def _format_recent_dialogue(
    history: list[dict] | None,
    user_text: str,
    assistant_text: str = "",
    character: str = "",
    max_messages: int = MEMORY_CONTEXT_MESSAGES,
) -> str:
    lines: list[str] = []
    prior = list(history or [])
    if prior and prior[-1].get("role") == "assistant" and assistant_text:
        if (prior[-1].get("content") or "") == assistant_text:
            prior = prior[:-1]
    if prior and prior[-1].get("role") == "user" and (prior[-1].get("content") or "") == user_text:
        prior = prior[:-1]

    recent = prior[-max_messages:] if max_messages > 0 else prior
    if recent:
        lines.append("【最近对话】")
        for msg in recent:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            who = "用户" if role == "user" else character if role == "assistant" else role
            lines.append(f"{who}：{content}")
    lines.append("【最新一轮】")
    lines.append(f"用户：{user_text}")
    if assistant_text:
        lines.append(f"{character}：{assistant_text}")
    return "\n".join(lines)


def is_important(
    model,
    tokenizer,
    user_text: str,
    assistant_text: str = "",
    history: list[dict] | None = None,
    character: str = "",
) -> bool:
    user_blob = _format_recent_dialogue(
        history, user_text, assistant_text, character=character
    )
    obj = _generate_json(
        model,
        tokenizer,
        _memory_prompt_pack()["gate"],
        user_blob,
        GATE_MAX_NEW_TOKENS,
    )
    if not obj:
        return False
    val = obj.get("important")
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1"}
    return False


def _parse_facts_list(obj: dict | None) -> list[str]:
    if not obj:
        return []
    raw = obj.get("facts")
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
    elif isinstance(obj.get("fact"), str) and obj["fact"].strip():
        out.append(obj["fact"].strip())
    seen: set[str] = set()
    unique: list[str] = []
    for t in out[:MAX_FACTS_PER_TURN]:
        key = _normalize(t)
        if key and key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def summarize_for_perspective(
    model,
    tokenizer,
    perspective: str,
    user_text: str,
    assistant_text: str = "",
    history: list[dict] | None = None,
    character: str = "",
) -> list[str]:
    """Rewrite important content for Narrator, Player, or interlocutor POV."""
    user_blob = _format_recent_dialogue(
        history, user_text, assistant_text, character=character
    )
    mp = _memory_prompt_pack()
    if perspective == NARRATOR_KEY:
        system = mp["rewrite_narrator"]
    elif perspective == PLAYER_KEY:
        system = mp["rewrite_player"]
    else:
        system = mp["rewrite_other"]
        label = "Character name" if _ACTIVE_LANGUAGE == "en" else "角色名"
        user_blob = f"{label}：{character}\n{user_blob}"
    obj = _generate_json(
        model, tokenizer, system, user_blob, REWRITE_MAX_NEW_TOKENS
    )
    return _parse_facts_list(obj)


def _store_one_string(
    store: dict,
    character: str,
    fact_text: str,
    path: Path,
    retriever: MemoryRetriever | None,
    skip_norms: set[str] | None = None,
    *,
    certainty: float | None = None,
    skip_scoring: bool = False,
    stimulus_text: str = "",
    persona_brief: str = "",
    recalled_hits: list[dict] | None = None,
    model=None,
    tokenizer=None,
) -> str | None:
    """Insert one fact into a character bucket, or skip near/exact dup."""
    fact_text = (fact_text or "").strip()
    if not fact_text:
        return None
    skip_norms = skip_norms or set()
    norm = _normalize(fact_text)
    if norm in skip_norms:
        return None

    dup_i = find_string_duplicate(store, character, fact_text)
    if dup_i is not None:
        entry = ensure_character(store, character)[dup_i]
        bump_certainty(entry, amount=0.05)
        mark_session_new(store, character, fact_text)
        save_memory(store, path)
        print(f"[memory] already known, bumped ({character}): {fact_text}")
        return None

    if retriever is not None:
        hit = retriever.best_match(fact_text, character=character, min_score=NEAR_DUP_MIN)
        if hit:
            parsed = parse_memory_id(hit.get("id") or "")
            if parsed:
                _, idx = parsed
                bucket = ensure_character(store, character)
                if 0 <= idx < len(bucket):
                    bump_certainty(bucket[idx], amount=0.06)
                    mark_session_new(store, character, _entry_text(bucket[idx]))
                    save_memory(store, path)
            print(f"[memory] near-dup bump ({character}): {fact_text}")
            return None

    if certainty is None and not skip_scoring and model is not None and tokenizer is not None:
        certainty = compute_certainty(
            model,
            tokenizer,
            stimulus_text or fact_text,
            fact_text,
            recalled_hits=recalled_hits,
            retriever=retriever,
            persona_brief=persona_brief,
        )
    if certainty is None:
        certainty = 0.45

    result = add_memory(store, character, fact_text, certainty=certainty)
    if result is None:
        return None
    text, index = result
    save_memory(store, path)
    if retriever is not None:
        retriever.add_memory_embedding(character, text, index)
    print(f"[memory] saved ({character}, c={certainty:.2f}): {text}")
    return text


def record_dialogue_beat(
    store: dict,
    *,
    speaker: str,
    text: str,
    to: list[str] | None = None,
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
) -> str | None:
    """
    Always store a short beat for the speaker so the next cue sees fresh context.
    No importance gate / rewrite — used by the directed story loop.
    """
    speaker = (speaker or "").strip()
    text = (text or "").strip()
    if not speaker or not text:
        return None
    if speaker in (NARRATOR_KEY, PLAYER_KEY):
        who = speaker
    elif is_speakable(speaker):
        who = speaker
    else:
        return None
    en = _ACTIVE_LANGUAGE == "en"
    join = ", " if en else "、"
    to_s = join.join(t for t in (to or []) if t) or ("—" if en else "—")
    if speaker == NARRATOR_KEY:
        beat = f"Narrator: {text}" if en else f"旁白：{text}"
    elif speaker == PLAYER_KEY:
        beat = (
            f"Player said to {to_s}: {text}"
            if en
            else f"玩家对{to_s}说：{text}"
        )
    else:
        beat = (
            f"{speaker} said to {to_s}: {text}"
            if en
            else f"{speaker}对{to_s}说：{text}"
        )
    return _store_one_string(store, who, beat, path, retriever)


def store_character_impressions(
    model,
    tokenizer,
    store: dict,
    *,
    character: str,
    stimulus_text: str,
    assistant_text: str = "",
    history: list[dict] | None = None,
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
    persona_brief: str = "",
    recalled_hits: list[dict] | None = None,
) -> list[str]:
    """
    Write first-person *biased* memory lines for one character.
    Scene transcripts stay out of LTM — only subjective takeaways are stored.
    """
    character = (character or "").strip()
    stimulus_text = (stimulus_text or "").strip()
    if not character or not is_speakable(character) or not stimulus_text:
        return []
    facts = summarize_for_perspective(
        model,
        tokenizer,
        character,
        stimulus_text,
        assistant_text,
        history=history,
        character=character,
    )
    saved: list[str] = []
    norms: set[str] = set()
    blob = stimulus_text if not assistant_text else f"{stimulus_text}\n{assistant_text}"
    for fact in facts:
        out = _store_one_string(
            store,
            character,
            fact,
            path,
            retriever,
            skip_norms=norms,
            stimulus_text=blob,
            persona_brief=persona_brief,
            recalled_hits=recalled_hits,
            model=model,
            tokenizer=tokenizer,
        )
        if out is not None:
            saved.append(out)
            norms.add(_normalize(out))
    return saved


def record_present_scene_views(
    model,
    tokenizer,
    store: dict,
    *,
    narration: str,
    present_names: list[str],
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
    persona_briefs: dict[str, str] | None = None,
) -> None:
    """Biased first-person scene impressions for each present NPC (not verbatim dump)."""
    text = (narration or "").strip()
    if not text:
        return
    persona_briefs = persona_briefs or {}
    for raw in present_names or []:
        name = (raw or "").strip()
        if not name or not is_speakable(name):
            continue
        saved = store_character_impressions(
            model,
            tokenizer,
            store,
            character=name,
            stimulus_text=text,
            path=path,
            retriever=retriever,
            persona_brief=persona_briefs.get(name, ""),
        )
        if saved:
            continue
        # Fallback if the rewrite returns nothing.
        en = _ACTIVE_LANGUAGE == "en"
        impression = (
            f"I took the scene as: {text[:220]}"
            if en
            else f"我对这场戏的感受是：{text[:220]}"
        )
        _store_one_string(
            store,
            name,
            impression,
            path,
            retriever,
            stimulus_text=text,
            persona_brief=persona_briefs.get(name, ""),
            model=model,
            tokenizer=tokenizer,
        )


def extract_and_store(
    model,
    tokenizer,
    store: dict,
    user_text: str,
    assistant_text: str = "",
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
    recalled_hits: list[dict] | None = None,
    history: list[dict] | None = None,
    character: str = "",
    *,
    skip_gate: bool = False,
    character_only: bool = False,
    persona_brief: str = "",
) -> dict[str, list[str]]:
    """
    Gate once → rewrite perspectives → store with certainty scores.
    Story NPC turns use character_only=True so only that character's biased
    impressions are written (scene messages stay out of LTM).
    """
    character = (character or "").strip()
    if not character:
        raise ValueError("character required for memory extract")
    if not is_speakable(character):
        raise ValueError(f"Unknown speakable character: {character}")
    if character in (NARRATOR_KEY, PLAYER_KEY):
        raise ValueError(f"Cannot speak to reserved key: {character}")

    if not user_text or not user_text.strip():
        return {NARRATOR_KEY: [], PLAYER_KEY: [], character: []}

    if not skip_gate and not is_important(
        model,
        tokenizer,
        user_text,
        assistant_text,
        history=history,
        character=character,
    ):
        print("[memory] not important — skip")
        return {NARRATOR_KEY: [], PLAYER_KEY: [], character: []}

    results: dict[str, list[str]] = {
        NARRATOR_KEY: [],
        PLAYER_KEY: [],
        character: [],
    }
    perspectives = [character] if character_only else turn_memory_keys(character)
    blob = user_text if not assistant_text else f"{user_text}\n{assistant_text}"
    for perspective in perspectives:
        fact_texts = summarize_for_perspective(
            model,
            tokenizer,
            perspective,
            user_text,
            assistant_text,
            history=history,
            character=character,
        )
        if not fact_texts:
            print(f"[memory] {perspective}: no facts")
            continue
        print(f"[memory] {perspective}: {len(fact_texts)} fact(s)")
        this_turn_norms: set[str] = set()
        for fact_text in fact_texts:
            saved = _store_one_string(
                store,
                perspective,
                fact_text,
                path,
                retriever,
                skip_norms=this_turn_norms,
                stimulus_text=blob,
                persona_brief=persona_brief if perspective == character else "",
                recalled_hits=recalled_hits,
                model=model,
                tokenizer=tokenizer,
            )
            if saved is not None:
                results[perspective].append(saved)
                this_turn_norms.add(_normalize(saved))
    return results


def _compatible_preference_pair(text_a: str, text_b: str) -> bool:
    a = text_a or ""
    b = text_b or ""
    if _CALL_PREF_RE.search(a) and _CALL_PREF_RE.search(b):
        return True
    if _LIKE_PREF_RE.search(a) and _LIKE_PREF_RE.search(b):
        return True
    return False


def classify_relation(model, tokenizer, text_a: str, text_b: str) -> str:
    if _normalize(text_a) and _normalize(text_a) == _normalize(text_b):
        return "same"
    if _compatible_preference_pair(text_a, text_b):
        return "unrelated"

    blob = f"事实A：{text_a}\n事实B：{text_b}"
    obj = _generate_json(
        model,
        tokenizer,
        _memory_prompt_pack()["organize"],
        blob,
        ORGANIZE_MAX_NEW_TOKENS,
    )
    if not obj:
        return "unrelated"
    rel = str(obj.get("relation", "unrelated")).strip().lower()
    if rel not in {"same", "contradict", "unrelated"}:
        return "unrelated"
    if rel in {"same", "contradict"} and _compatible_preference_pair(text_a, text_b):
        return "unrelated"
    return rel


def organize_memory(
    model,
    tokenizer,
    store: dict,
    retriever: MemoryRetriever | None = None,
    path: Path = MEMORY_PATH,
) -> dict:
    """
    After each scene (and good night): compare session_new facts to peers.
    same / contradict → keep the higher-certainty entry (Rizz rule).
    """
    # Prefer fact-level session_new flags; fall back to SESSION_NEW_KEY text map.
    keys: list[str] = []
    for key, values in store.items():
        if not isinstance(key, str) or key.startswith("_") or not isinstance(values, list):
            continue
        ensure_character(store, key)
        if any(bool(e.get("session_new")) for e in store[key]):
            keys.append(key)
    if not keys:
        session_map = _session_new_map(store)
        keys = [k for k, texts in session_map.items() if texts]
    if not keys:
        print("[organize] no new memories this session")
        return store

    if retriever is not None:
        retriever.rebuild(store)

    for key in keys:
        bucket = ensure_character(store, key)
        new_idxs = [i for i, e in enumerate(bucket) if e.get("session_new")]
        if not new_idxs:
            # Legacy text map path
            session_map = _session_new_map(store)
            wanted = {_normalize(t) for t in (session_map.get(key) or [])}
            new_idxs = [
                i
                for i, e in enumerate(bucket)
                if _normalize(_entry_text(e)) in wanted
            ]
        if not new_idxs:
            continue
        print(f"[organize] {key}: reviewing {len(new_idxs)} new memor(ies)…")
        drop_indices: set[int] = set()

        for new_i in list(new_idxs):
            if new_i in drop_indices or new_i >= len(bucket):
                continue
            new_entry = bucket[new_i]
            new_text = _entry_text(new_entry)
            if not new_text:
                drop_indices.add(new_i)
                continue

            candidates: list[int] = []
            if retriever is not None:
                hits = retriever.retrieve(
                    new_text,
                    top_k=8,
                    min_score=ORGANIZE_CANDIDATE_MIN,
                    characters=[key],
                )
                for h in hits:
                    parsed = parse_memory_id(h.get("id") or "")
                    if not parsed:
                        continue
                    _, idx = parsed
                    if idx == new_i or idx in drop_indices:
                        continue
                    if 0 <= idx < len(bucket):
                        candidates.append(idx)
            else:
                candidates = [
                    i for i in range(len(bucket)) if i != new_i and i not in drop_indices
                ]

            for other_i in candidates:
                if new_i in drop_indices or other_i in drop_indices:
                    continue
                other_entry = bucket[other_i]
                other_text = _entry_text(other_entry)
                rel = classify_relation(model, tokenizer, new_text, other_text)
                if rel == "unrelated":
                    continue
                c_new = _entry_certainty(new_entry)
                c_old = _entry_certainty(other_entry)
                if c_new > c_old or (
                    abs(c_new - c_old) < 1e-9 and new_entry.get("session_new")
                ):
                    keep_i, discard_i = new_i, other_i
                else:
                    keep_i, discard_i = other_i, new_i
                print(
                    f"[organize] {key} {rel}: keep «{_entry_text(bucket[keep_i])}» "
                    f"(c={_entry_certainty(bucket[keep_i]):.2f}) "
                    f"drop «{_entry_text(bucket[discard_i])}» "
                    f"(c={_entry_certainty(bucket[discard_i]):.2f})"
                )
                drop_indices.add(discard_i)
                if keep_i == other_i and rel in {"same", "contradict"}:
                    bump_certainty(bucket[keep_i], amount=0.05)

        if drop_indices:
            store[key] = [e for i, e in enumerate(bucket) if i not in drop_indices]
            print(f"[organize] {key}: removed {len(drop_indices)} memor(ies)")

    # Clear session flags on survivors.
    for key, values in store.items():
        if not isinstance(key, str) or key.startswith("_") or not isinstance(values, list):
            continue
        for entry in ensure_character(store, key):
            entry["session_new"] = False
    store[SESSION_NEW_KEY] = {}
    save_memory(store, path)
    if retriever is not None:
        retriever.rebuild(store)
    print(f"[organize] done. {count_memories(store)} memor(ies) remain.")
    return store
