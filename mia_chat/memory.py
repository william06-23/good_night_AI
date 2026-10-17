"""
Long-term memory for Mia: load/save JSON facts, importance gate, rewrite,
certainty scoring, bge-m3 retrieval (CPU), and good-night organization.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

# Default store next to this package: mia_chat/data/long_term_memory.json
DATA_DIR = Path(__file__).resolve().parent / "data"
MEMORY_PATH = DATA_DIR / "long_term_memory.json"

# Embedding retriever — keep on CPU so Qwen can own the GPU VRAM.
EMBED_MODEL_ID = r"C:\Users\William\William's_AI_project\bge-m3"
EMBED_DEVICE = "cpu"
RETRIEVE_TOP_K = 5
RETRIEVE_MIN_SCORE = 0.25

# Pair candidates for good-night organize (embedding similarity).
ORGANIZE_CANDIDATE_MIN = 0.45
# Treat as same fact for bump-instead-of-insert (high bar so different facts stay separate).
NEAR_DUP_MIN = 0.88

GATE_MAX_NEW_TOKENS = 64
REWRITE_MAX_NEW_TOKENS = 192
CERTAINTY_MAX_NEW_TOKENS = 64
ORGANIZE_MAX_NEW_TOKENS = 96
MAX_FACTS_PER_TURN = 5

GATE_SYSTEM = (
    "你是一个记忆筛选器。判断用户这句话是否包含值得长期记住的信息。"
    "值得记住：个人事实、偏好、计划、关系、重要情绪、承诺。"
    "不值得：寒暄、玩笑、一次性闲聊、无实质内容。"
    '只输出 JSON，格式严格为：{"important": true} 或 {"important": false}'
)

REWRITE_SYSTEM = (
    "你是一个记忆整理器。把下面对话里值得长期记住的信息改写成简短客观的陈述句。"
    "默认只输出 1 条事实。只有用户明确说了两个互不相关的独立信息时，才拆成多条（最多5条）。"
    "同一件事的细节、同义改写、顺带情绪不要拆成多条。"
    "没有可记内容则输出空数组。"
    "用中文或英文与原文一致即可，不要撒娇口吻，不要第一人称（用名字或“用户”）。"
    '只输出 JSON，格式严格为：{"facts": ["事实1", ...]}'
)

INTRO_SYSTEM = (
    "你是一个置信度评估器。根据用户如何陈述信息，给出 introduction 分（0到1的小数）。"
    "高分：明确断言、纠正、强调（“我叫…”，“其实是…”，“记住…”）。"
    "中分：普通提及、随口说出。"
    "低分：猜测、玩笑、犹豫、反问、不确定。"
    '只输出 JSON：{"intro": 0.0到1.0}'
)

ORGANIZE_SYSTEM = (
    "你是记忆冲突仲裁器。比较两条关于用户的事实，判断关系。"
    'relation 只能是 "same"、"contradict"、"unrelated" 之一。'
    "same：意思基本相同（改写/同义/同一事实）。"
    "contradict：两条不能同时为真（例如互斥身份、互斥日期）。"
    "unrelated：可以同时为真（例如喜欢蓝色和喜欢黄色；不同方面的偏好）。"
    '只输出 JSON：{"relation": "same"|"contradict"|"unrelated"}'
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _empty_store() -> dict:
    return {"facts": []}


def _normalize_fact(f: dict) -> dict:
    """Ensure newer fields exist on older JSON entries."""
    if "certainty" not in f or not isinstance(f.get("certainty"), (int, float)):
        f["certainty"] = 0.5
    else:
        f["certainty"] = _clamp(float(f["certainty"]))
    if "session_new" not in f:
        f["session_new"] = False
    if "updated_at" not in f:
        f["updated_at"] = f.get("created_at") or _now_iso()
    return f


def load_memory(path: Path = MEMORY_PATH) -> dict:
    """Load long-term memory JSON; create empty store if missing/corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        store = _empty_store()
        save_memory(store, path)
        return store
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "facts" not in data:
            return _empty_store()
        if not isinstance(data["facts"], list):
            data["facts"] = []
        data["facts"] = [_normalize_fact(f) for f in data["facts"] if isinstance(f, dict)]
        return data
    except (json.JSONDecodeError, OSError):
        return _empty_store()


def save_memory(store: dict, path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def format_facts_for_prompt(facts: list[dict] | list[str], max_facts: int = 40) -> str:
    """Compact block appended to Mia's system prompt from a fact list."""
    if not facts:
        return ""
    lines = []
    for f in facts[-max_facts:]:
        if isinstance(f, dict):
            text = (f.get("text") or "").strip()
            c = f.get("certainty")
            if text and isinstance(c, (int, float)):
                lines.append(f"- {text} (确信 {float(c):.2f})")
            elif text:
                lines.append(f"- {text}")
        else:
            text = str(f).strip()
            if text:
                lines.append(f"- {text}")
    if not lines:
        return ""
    return "已知长期记忆（与用户相关，聊天时可自然用到，不要逐条念出来）：\n" + "\n".join(lines)


class MemoryRetriever:
    """
    Dense retrieval over long-term facts with BAAI/bge-m3.

    Runs on EMBED_DEVICE (default CPU) so it does not compete with Qwen for VRAM.
    """

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
        self._matrix: torch.Tensor | None = None  # (N, D) on CPU, L2-normalized

    def rebuild(self, store: dict) -> None:
        """Re-embed all facts from the memory store."""
        facts = store.get("facts") or []
        self._ids = []
        self._texts = []
        texts = []
        for f in facts:
            text = (f.get("text") or "").strip()
            if not text:
                continue
            self._ids.append(f.get("id") or "")
            self._texts.append(text)
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
        self._matrix = torch.from_numpy(emb)  # stays on CPU
        print(f"[memory] retriever indexed {len(texts)} fact(s) on {self.device}")

    def add_fact_embedding(self, entry: dict) -> None:
        """Append one new fact embedding without full rebuild."""
        text = (entry.get("text") or "").strip()
        if not text:
            return
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        row = torch.from_numpy(emb)
        self._ids.append(entry.get("id") or "")
        self._texts.append(text)
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
        return torch.from_numpy(emb)  # (1, D)

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVE_TOP_K,
        min_score: float = RETRIEVE_MIN_SCORE,
    ) -> list[dict]:
        """Return top-k facts most similar to query (as {id, text, score})."""
        if self._matrix is None or not self._texts or not query.strip():
            return []
        q_t = self.encode_one(query)
        if q_t is None:
            return []
        scores = (q_t @ self._matrix.T).squeeze(0)  # cosine = dot if normalized
        k = min(top_k, scores.numel())
        vals, idxs = torch.topk(scores, k=k)
        out = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            if score < min_score:
                continue
            out.append(
                {
                    "id": self._ids[idx],
                    "text": self._texts[idx],
                    "score": float(score),
                }
            )
        return out

    def best_match(
        self,
        query: str,
        min_score: float = NEAR_DUP_MIN,
        exclude_ids: set[str] | None = None,
    ) -> dict | None:
        """Single best embedding match above min_score."""
        hits = self.retrieve(query, top_k=5, min_score=min_score)
        exclude_ids = exclude_ids or set()
        for h in hits:
            if h["id"] not in exclude_ids:
                return h
        return None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def is_duplicate(store: dict, fact_text: str) -> bool:
    """Exact normalized match only — different facts must not collapse via substring."""
    return find_string_duplicate(store, fact_text) is not None


def find_string_duplicate(store: dict, fact_text: str) -> dict | None:
    """Exact normalized text match only (no substring)."""
    new_n = _normalize(fact_text)
    if not new_n:
        return None
    for f in store.get("facts") or []:
        old = _normalize(f.get("text", ""))
        if old and new_n == old:
            return f
    return None


def add_fact(
    store: dict,
    text: str,
    certainty: float,
    source: str = "chat",
    session_new: bool = True,
) -> dict | None:
    """Append a fact. Returns the new entry (caller should avoid duplicates first)."""
    text = (text or "").strip()
    if not text:
        return None
    now = _now_iso()
    entry = {
        "id": str(uuid.uuid4()),
        "text": text,
        "certainty": _clamp(certainty),
        "created_at": now,
        "updated_at": now,
        "source": source,
        "session_new": session_new,
    }
    store.setdefault("facts", []).append(entry)
    return entry


def _extract_json_object(text: str) -> dict | None:
    """Parse first JSON object from model output (tolerates markdown fences)."""
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


def is_important(model, tokenizer, user_text: str, assistant_text: str = "") -> bool:
    """Gate: return True if the latest exchange looks worth long-term storage."""
    user_blob = f"用户：{user_text}"
    if assistant_text:
        user_blob += f"\n米娅：{assistant_text}"
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


def summarize_facts(model, tokenizer, user_text: str, assistant_text: str = "") -> list[str]:
    """Rewrite important content into one or more short factual sentences."""
    user_blob = f"用户：{user_text}"
    if assistant_text:
        user_blob += f"\n米娅：{assistant_text}"
    obj = _generate_json(
        model, tokenizer, REWRITE_SYSTEM, user_blob, REWRITE_MAX_NEW_TOKENS
    )
    if not obj:
        return []
    raw = obj.get("facts")
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
    # Backward-compatible single-fact key.
    elif isinstance(obj.get("fact"), str) and obj["fact"].strip():
        out.append(obj["fact"].strip())
    # Dedupe within this turn (normalized).
    seen: set[str] = set()
    unique: list[str] = []
    for t in out[:MAX_FACTS_PER_TURN]:
        key = _normalize(t)
        if key and key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def score_introduction(model, tokenizer, user_text: str, fact_text: str) -> float:
    """How strongly the user asserted this info (0–1). Defaults mid-low."""
    blob = f"用户原话：{user_text}\n提炼事实：{fact_text}"
    obj = _generate_json(model, tokenizer, INTRO_SYSTEM, blob, CERTAINTY_MAX_NEW_TOKENS)
    if not obj:
        return 0.35
    raw = obj.get("intro")
    try:
        return _clamp(float(raw))
    except (TypeError, ValueError):
        return 0.35


def score_context_match(
    fact_text: str,
    recalled_hits: list[dict] | None,
    retriever: MemoryRetriever | None,
) -> float:
    """
    How well the new fact aligns with this turn's recalled long-term context (0–1).
    Uses average similarity to recalled fact texts; 0.3 if nothing recalled.
    """
    if not recalled_hits:
        return 0.3
    if retriever is None:
        # Fall back to retrieve scores already on hits if query was user text,
        # but for fact-vs-recalled we average hit scores as a proxy.
        scores = [float(h.get("score", 0)) for h in recalled_hits]
        return _clamp(sum(scores) / len(scores)) if scores else 0.3

    q = retriever.encode_one(fact_text)
    if q is None:
        return 0.3
    sims = []
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


def compute_certainty(
    model,
    tokenizer,
    user_text: str,
    fact_text: str,
    recalled_hits: list[dict] | None = None,
    retriever: MemoryRetriever | None = None,
    intro: float | None = None,
) -> float:
    """
    New facts start low; intro strength + context match raise them a little.
    certainty = 0.06 + 0.28*intro + 0.22*context  (clamped ~0.05–0.55)
    """
    if intro is None:
        intro = score_introduction(model, tokenizer, user_text, fact_text)
    context = score_context_match(fact_text, recalled_hits, retriever)
    certainty = _clamp(0.06 + 0.28 * intro + 0.22 * context, 0.05, 0.55)
    print(f"[memory] certainty parts: intro={intro:.2f} context={context:.2f} -> {certainty:.2f}")
    return certainty


def bump_certainty(
    entry: dict,
    intro: float,
    base_rate: float = 0.22,
) -> float:
    """
    Raise certainty when the user re-hints the same info.

    Logistic-style approach to 1: delta = rate * (1 - old).
    Bigger steps when certainty is low, smaller as it nears 1.
    With typical intro≈0.6, ~10 hints move a mid/low fact close to ~0.9+.
    """
    old = _clamp(float(entry.get("certainty") or 0.5))
    intro = _clamp(intro)
    # Weak assertions move slower; strong ones use full base_rate.
    rate = base_rate * (0.4 + 0.6 * intro)
    new = _clamp(old + rate * (1.0 - old), 0.05, 0.99)
    entry["certainty"] = new
    entry["updated_at"] = _now_iso()
    entry["session_new"] = True  # re-touch so organize can reconsider pairs
    return new


def _fact_by_id(store: dict, fact_id: str) -> dict | None:
    for f in store.get("facts") or []:
        if f.get("id") == fact_id:
            return f
    return None


def _store_one_fact(
    model,
    tokenizer,
    store: dict,
    user_text: str,
    fact_text: str,
    path: Path,
    retriever: MemoryRetriever | None,
    recalled_hits: list[dict] | None,
    skip_ids: set[str] | None = None,
) -> dict | None:
    """Score + bump or insert one fact, independent of sibling facts in this turn."""
    intro = score_introduction(model, tokenizer, user_text, fact_text)
    skip_ids = skip_ids or set()

    existing = find_string_duplicate(store, fact_text)
    if existing is not None and existing.get("id") in skip_ids:
        existing = None
    if existing is None and retriever is not None:
        hit = retriever.best_match(
            fact_text, min_score=NEAR_DUP_MIN, exclude_ids=skip_ids
        )
        if hit:
            existing = _fact_by_id(store, hit["id"])

    if existing is not None:
        new_c = bump_certainty(existing, intro)
        save_memory(store, path)
        print(f"[memory] bumped certainty -> {new_c:.2f}: {existing.get('text')}")
        return existing

    certainty = compute_certainty(
        model,
        tokenizer,
        user_text,
        fact_text,
        recalled_hits=recalled_hits,
        retriever=retriever,
        intro=intro,
    )
    entry = add_fact(store, fact_text, certainty=certainty, session_new=True)
    if entry is None:
        return None
    save_memory(store, path)
    if retriever is not None:
        retriever.add_fact_embedding(entry)
    print(f"[memory] saved (certainty={certainty:.2f}): {fact_text}")
    return entry


def extract_and_store(
    model,
    tokenizer,
    store: dict,
    user_text: str,
    assistant_text: str = "",
    path: Path = MEMORY_PATH,
    retriever: MemoryRetriever | None = None,
    recalled_hits: list[dict] | None = None,
) -> list[dict]:
    """
    Gate → rewrite into independent facts → score/store each separately.
    Sibling facts from the same turn do not merge into each other.
    """
    if not user_text or not user_text.strip():
        return []
    if not is_important(model, tokenizer, user_text, assistant_text):
        print("[memory] not important — skip")
        return []
    fact_texts = summarize_facts(model, tokenizer, user_text, assistant_text)
    if not fact_texts:
        print("[memory] rewrite returned no facts — skip")
        return []

    print(f"[memory] extracted {len(fact_texts)} separate fact(s) from this turn")
    results: list[dict] = []
    # IDs created this turn — never bump one new fact into another from the same message.
    this_turn_ids: set[str] = set()
    for fact_text in fact_texts:
        entry = _store_one_fact(
            model,
            tokenizer,
            store,
            user_text,
            fact_text,
            path,
            retriever,
            recalled_hits,
            skip_ids=this_turn_ids,
        )
        if entry is not None:
            results.append(entry)
            fid = entry.get("id")
            if fid:
                this_turn_ids.add(fid)
    return results


def classify_relation(model, tokenizer, text_a: str, text_b: str) -> str:
    """Return 'same' | 'contradict' | 'unrelated'."""
    blob = f"事实A：{text_a}\n事实B：{text_b}"
    obj = _generate_json(model, tokenizer, ORGANIZE_SYSTEM, blob, ORGANIZE_MAX_NEW_TOKENS)
    if not obj:
        return "unrelated"
    rel = str(obj.get("relation", "unrelated")).strip().lower()
    if rel in {"same", "contradict", "unrelated"}:
        return rel
    return "unrelated"


def organize_memory(
    model,
    tokenizer,
    store: dict,
    retriever: MemoryRetriever | None = None,
    path: Path = MEMORY_PATH,
) -> dict:
    """
    After good night: compare session_new facts to older ones.
    same / contradict → keep higher certainty, drop the other.
    Rebuilds nothing itself; caller should retriever.rebuild(store).
    """
    facts = store.get("facts") or []
    new_facts = [f for f in facts if f.get("session_new")]
    if not new_facts:
        print("[organize] no new facts this session")
        return store

    print(f"[organize] reviewing {len(new_facts)} new fact(s)…")
    drop_ids: set[str] = set()

    # Ensure embedder index is current for candidate search.
    if retriever is not None:
        retriever.rebuild(store)

    for new in new_facts:
        if new.get("id") in drop_ids:
            continue
        new_text = (new.get("text") or "").strip()
        if not new_text:
            drop_ids.add(new["id"])
            continue

        candidates: list[dict] = []
        if retriever is not None:
            hits = retriever.retrieve(
                new_text, top_k=8, min_score=ORGANIZE_CANDIDATE_MIN
            )
            for h in hits:
                if h["id"] == new.get("id") or h["id"] in drop_ids:
                    continue
                other = _fact_by_id(store, h["id"])
                if other is not None:
                    candidates.append(other)
        else:
            candidates = [
                f
                for f in facts
                if f.get("id") != new.get("id") and f.get("id") not in drop_ids
            ]

        for other in candidates:
            if other.get("id") in drop_ids or new.get("id") in drop_ids:
                continue
            # Prefer comparing against settled (non-session) facts first; still OK vs peers.
            rel = classify_relation(
                model, tokenizer, new_text, (other.get("text") or "")
            )
            if rel == "unrelated":
                continue

            c_new = float(new.get("certainty") or 0)
            c_old = float(other.get("certainty") or 0)
            if c_new > c_old or (c_new == c_old and new.get("session_new")):
                keep, discard = new, other
            else:
                keep, discard = other, new

            print(
                f"[organize] {rel}: keep «{keep.get('text')}» "
                f"(c={float(keep.get('certainty') or 0):.2f}) "
                f"drop «{discard.get('text')}» "
                f"(c={float(discard.get('certainty') or 0):.2f})"
            )
            drop_ids.add(discard["id"])
            # Survivor may absorb a small bump when it was the older one confirmed by new talk.
            if keep is other and rel in {"same", "contradict"}:
                keep["certainty"] = _clamp(float(keep.get("certainty") or 0) + 0.05)
                keep["updated_at"] = _now_iso()

    if drop_ids:
        store["facts"] = [f for f in facts if f.get("id") not in drop_ids]
        print(f"[organize] removed {len(drop_ids)} fact(s)")

    # Clear session flags on survivors.
    for f in store.get("facts") or []:
        f["session_new"] = False

    save_memory(store, path)
    print(f"[organize] done. {len(store.get('facts') or [])} fact(s) remain.")
    return store
