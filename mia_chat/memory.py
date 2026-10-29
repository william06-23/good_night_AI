"""
Multi-perspective long-term memory: character-keyed string lists,
importance gate, three POV rewriters, bge-m3 retrieval (CPU), organize.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parent / "data"
MEMORY_PATH = DATA_DIR / "long_term_memory.json"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = Path(os.environ.get("MIA_MODELS_DIR", _REPO_ROOT / "models")).expanduser()
EMBED_MODEL_ID = str(_MODELS_DIR / "bge-m3")
EMBED_DEVICE = "cpu"
RETRIEVE_TOP_K = 5
RETRIEVE_MIN_SCORE = 0.25
RETRIEVE_CANDIDATE_MULT = 3

ORGANIZE_CANDIDATE_MIN = 0.45
NEAR_DUP_MIN = 0.88

GATE_MAX_NEW_TOKENS = 64
REWRITE_MAX_NEW_TOKENS = 192
ORGANIZE_MAX_NEW_TOKENS = 96
MAX_FACTS_PER_TURN = 5
MEMORY_CONTEXT_MESSAGES = 5

# Fixed perspective keys + speakable NPC roster.
NARRATOR_KEY = "Narrator"
PLAYER_KEY = "Player"
DEFAULT_CHARACTER = "Mia"
SPEAKABLE_CHARACTERS = [DEFAULT_CHARACTER]
SESSION_NEW_KEY = "_session_new"  # in-RAM only; never written to JSON

GATE_SYSTEM = (
    "你是一个记忆筛选器。结合最近对话上下文，判断【最新一轮】是否包含值得长期记住的信息。"
    "值得记住：个人事实、身份、地点、计划、关系、承诺、重要情绪；"
    "以及各类偏好与喜好（喜欢/讨厌的食物、颜色、活动、习惯、口味等），即使说得随意也算重要。"
    "不值得：纯寒暄、无信息的玩笑、一次性语气词、没有新信息的附和。"
    "若最新一句本身含糊，但结合上文能确定是在陈述偏好或事实，也判 important。"
    '只输出 JSON，格式严格为：{"important": true} 或 {"important": false}'
)

REWRITE_NARRATOR_SYSTEM = (
    "你是客观旁白记忆整理器。根据最近对话，把【最新一轮】里值得长期记住的信息"
    "改写成简短、第三人称、客观的陈述句（旁白视角，不偏任何一方）。"
    "必须结合上文理解指代与省略。默认 1 条，最多 5 条互不相关的独立信息。"
    "没有可记内容则输出空数组。不要撒娇口吻。"
    '只输出 JSON：{"facts": ["事实1", ...]}'
)

REWRITE_PLAYER_SYSTEM = (
    "你是玩家视角记忆整理器。根据最近对话，把【最新一轮】里玩家会记住的信息"
    "改写成简短陈述，用玩家第一人称「我」或「用户」指自己，对方用名字。"
    "只写玩家会关心/会记住的内容。默认 1 条，最多 5 条。没有可记内容则空数组。"
    '只输出 JSON：{"facts": ["事实1", ...]}'
)

REWRITE_OTHER_SYSTEM = (
    "你是角色视角记忆整理器。角色名会在用户消息里给出。"
    "把【最新一轮】里该角色会记住的信息改写成简短陈述，用该角色第一人称「我」，"
    "把玩家称为「用户」或对方的称呼。只写该角色视角下稳定、可回忆的内容。"
    "默认 1 条，最多 5 条。没有可记内容则空数组。"
    '只输出 JSON：{"facts": ["事实1", ...]}'
)

ORGANIZE_SYSTEM = (
    "你是记忆冲突仲裁器。比较同一视角下的两条记忆，判断关系。"
    'relation 只能是 "same"、"contradict"、"unrelated" 之一。'
    "same：几乎同一条事实的同义改写。"
    "contradict：两条不能同时为真。"
    "unrelated：可以同时为真。"
    "多种称呼/喜好可以并存 → unrelated。"
    '只输出 JSON：{"relation": "same"|"contradict"|"unrelated"}'
)

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


def is_speakable(name: str) -> bool:
    return (name or "").strip() in SPEAKABLE_CHARACTERS


def turn_memory_keys(character: str) -> list[str]:
    """Buckets recalled/written on a user-initiated turn with this interlocutor."""
    return [NARRATOR_KEY, PLAYER_KEY, character]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _empty_store() -> dict:
    store = {
        NARRATOR_KEY: [],
        PLAYER_KEY: [],
        DEFAULT_CHARACTER: [],
    }
    for name in SPEAKABLE_CHARACTERS:
        store.setdefault(name, [])
    return store


def ensure_character(store: dict, name: str) -> list[str]:
    name = (name or "").strip()
    if not name or name.startswith("_"):
        raise ValueError("Invalid memory key")
    if not isinstance(store.get(name), list):
        store[name] = []
    # Coerce entries to strings.
    cleaned: list[str] = []
    for item in store[name]:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        elif isinstance(item, dict):
            t = (item.get("text") or "").strip()
            if t:
                cleaned.append(t)
    store[name] = cleaned
    return store[name]


def character_memories(store: dict, name: str) -> list[str]:
    return list(ensure_character(store, name))


def iter_memories(store: dict):
    """Yield (character, index, text) for public keys."""
    for key, values in store.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if not isinstance(values, list):
            continue
        for i, item in enumerate(values):
            if isinstance(item, str) and item.strip():
                yield key, i, item.strip()


def count_memories(store: dict) -> int:
    return sum(1 for _ in iter_memories(store))


def public_memory_store(store: dict) -> dict:
    """Copy suitable for JSON/API: no underscore keys; string lists only."""
    out: dict[str, list[str]] = {}
    for key, values in store.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if not isinstance(values, list):
            continue
        out[key] = [
            (x.strip() if isinstance(x, str) else str(x.get("text", "")).strip())
            for x in values
            if (isinstance(x, str) and x.strip())
            or (isinstance(x, dict) and (x.get("text") or "").strip())
        ]
    for required in (NARRATOR_KEY, PLAYER_KEY, DEFAULT_CHARACTER):
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
    seen = {_normalize(t) for t in store[NARRATOR_KEY]}
    for t in texts:
        n = _normalize(t)
        if n and n not in seen:
            store[NARRATOR_KEY].append(t)
            seen.add(n)
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
                if isinstance(item, str) and item.strip():
                    store[key].append(item.strip())
                elif isinstance(item, dict) and (item.get("text") or "").strip():
                    store[key].append(item["text"].strip())
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


def format_facts_for_prompt(hits: list[dict] | list[str], max_facts: int = 40) -> str:
    """Labeled memory block for the system prompt."""
    if not hits:
        return ""
    by_who: dict[str, list[str]] = {}
    for h in hits[-max_facts:]:
        if isinstance(h, dict):
            text = (h.get("text") or "").strip()
            who = (h.get("character") or "Memory").strip() or "Memory"
        else:
            text = str(h).strip()
            who = "Memory"
        if text:
            by_who.setdefault(who, []).append(text)
    if not by_who:
        return ""
    lines = ["已知长期记忆（按视角分组；聊天时可自然用到，不要逐条念出来）："]
    for who, texts in by_who.items():
        lines.append(f"【{who}】")
        for t in texts:
            lines.append(f"- {t}")
    return "\n".join(lines)


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


def add_memory(store: dict, character: str, text: str) -> tuple[str, int] | None:
    """Append a string memory. Returns (text, index) or None."""
    text = (text or "").strip()
    if not text:
        return None
    bucket = ensure_character(store, character)
    bucket.append(text)
    mark_session_new(store, character, text)
    return text, len(bucket) - 1


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
    character: str = DEFAULT_CHARACTER,
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
    character: str = DEFAULT_CHARACTER,
) -> bool:
    user_blob = _format_recent_dialogue(
        history, user_text, assistant_text, character=character
    )
    obj = _generate_json(
        model, tokenizer, GATE_SYSTEM, user_blob, GATE_MAX_NEW_TOKENS
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
    character: str = DEFAULT_CHARACTER,
) -> list[str]:
    """Rewrite important content for Narrator, Player, or interlocutor POV."""
    user_blob = _format_recent_dialogue(
        history, user_text, assistant_text, character=character
    )
    if perspective == NARRATOR_KEY:
        system = REWRITE_NARRATOR_SYSTEM
    elif perspective == PLAYER_KEY:
        system = REWRITE_PLAYER_SYSTEM
    else:
        system = REWRITE_OTHER_SYSTEM
        user_blob = f"角色名：{character}\n{user_blob}"
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
) -> str | None:
    """Insert one string into a character bucket, or skip near/exact dup."""
    fact_text = (fact_text or "").strip()
    if not fact_text:
        return None
    skip_norms = skip_norms or set()
    norm = _normalize(fact_text)
    if norm in skip_norms:
        return None

    if find_string_duplicate(store, character, fact_text) is not None:
        print(f"[memory] already known ({character}): {fact_text}")
        return None

    if retriever is not None:
        hit = retriever.best_match(fact_text, character=character, min_score=NEAR_DUP_MIN)
        if hit:
            print(f"[memory] near-dup skip ({character}): {fact_text}")
            return None

    result = add_memory(store, character, fact_text)
    if result is None:
        return None
    text, index = result
    save_memory(store, path)
    if retriever is not None:
        retriever.add_memory_embedding(character, text, index)
    print(f"[memory] saved ({character}): {text}")
    return text


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
    character: str = DEFAULT_CHARACTER,
) -> dict[str, list[str]]:
    """
    Gate once → rewrite Narrator / Player / interlocutor → store each list.
    """
    _ = recalled_hits  # available for future scoring; unused without certainty
    character = (character or DEFAULT_CHARACTER).strip()
    if not is_speakable(character):
        raise ValueError(f"Unknown speakable character: {character}")
    if character in (NARRATOR_KEY, PLAYER_KEY):
        raise ValueError(f"Cannot speak to reserved key: {character}")

    if not user_text or not user_text.strip():
        return {NARRATOR_KEY: [], PLAYER_KEY: [], character: []}

    if not is_important(
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
    for perspective in turn_memory_keys(character):
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
    obj = _generate_json(model, tokenizer, ORGANIZE_SYSTEM, blob, ORGANIZE_MAX_NEW_TOKENS)
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
    After good night: within each key that has session-new strings,
    drop same/contradict peers (keep the earlier list entry).
    """
    session_map = _session_new_map(store)
    keys = [k for k, texts in session_map.items() if texts]
    if not keys:
        print("[organize] no new memories this session")
        return store

    if retriever is not None:
        retriever.rebuild(store)

    for key in keys:
        new_texts = list(session_map.get(key) or [])
        if not new_texts:
            continue
        print(f"[organize] {key}: reviewing {len(new_texts)} new memor(ies)…")
        bucket = ensure_character(store, key)
        drop_indices: set[int] = set()

        for new_text in new_texts:
            new_n = _normalize(new_text)
            new_idxs = [
                i
                for i, t in enumerate(bucket)
                if _normalize(t) == new_n and i not in drop_indices
            ]
            if not new_idxs:
                continue
            new_i = new_idxs[0]

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
                other_text = bucket[other_i]
                rel = classify_relation(model, tokenizer, new_text, other_text)
                if rel == "unrelated":
                    continue
                # Keep earlier list entry (lower index).
                keep_i, discard_i = (new_i, other_i) if new_i < other_i else (other_i, new_i)
                print(
                    f"[organize] {key} {rel}: keep «{bucket[keep_i]}» "
                    f"drop «{bucket[discard_i]}»"
                )
                drop_indices.add(discard_i)

        if drop_indices:
            store[key] = [t for i, t in enumerate(bucket) if i not in drop_indices]
            print(f"[organize] {key}: removed {len(drop_indices)} memor(ies)")

    store[SESSION_NEW_KEY] = {}
    save_memory(store, path)
    print(f"[organize] done. {count_memories(store)} memor(ies) remain.")
    return store
