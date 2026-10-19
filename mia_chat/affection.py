"""
Affection score for the Rizz-the-AI branch.

Persisted separately from chat/memory. Starts at 20.0; model suggests a float
delta after each user turn. 0 = blocked; 80+ can accept a confession → girlfriend.
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime
from pathlib import Path

from .memory import _generate_json, _format_recent_dialogue, format_facts_for_prompt

DATA_DIR = Path(__file__).resolve().parent / "data"
AFFECTION_PATH = DATA_DIR / "affection.json"

DEFAULT_VALUE = 20.0
CONFESS_THRESHOLD = 80.0
# Girlfriend + above this: clingy, and long gaps cost affection.
CLINGY_ABOVE = 85.0
# Girlfriend below this: she asks to break up, then the relationship ends.
BREAKUP_BELOW = 60.0
# After this many waking hours without a user message, clingy Mia loses affection.
# Local 11:00 PM–7:00 AM is not counted (see chat_log.awake_hours_between).
ABSENCE_GRACE_HOURS = 6.0
ABSENCE_PER_HOUR = 0.35
ABSENCE_PENALTY_CAP = 20.0
# Raw model delta cap (before parabolic speed weight).
MAX_ABS_DELTA = 50.0
# Normal chat ~1e-1; deeply moved +1～+5; severe negatives are O(10), worst case -50.
NORMAL_DELTA_SCALE = 0.1
MOVED_DELTA_MIN = 1.0
MOVED_DELTA_MAX = 5.0
SEVERE_DELTA_MIN = -50.0
# User-facing pace for every affection change. 1 = unchanged.
SPEED_MIN = 0.8
SPEED_MAX = 1.5
DEFAULT_SPEED = 1.0
# While she has a boyfriend, affection cannot rise past this.
BOYFRIEND_AFFECTION_CAP = 60.0
# Once per local day she is not the user's girlfriend.
BOYFRIEND_DAILY_CHANCE = 0.01
AFFECT_CONTEXT_MESSAGES = 5
# Parabolic speed: 1.0 at affection=50, ~SPEED_FLOOR at 0 and 100.
SPEED_FLOOR = 0.15
# Same-direction streak: 2 in a row → ×1.05, 5+ in a row → ×1.20.
STREAK_MIN = 1.05
STREAK_MAX = 1.20
STREAK_FULL = 5
STREAK_EPS = 1e-6

# Tone bands:
# ≤10 dismissive | ≤40 polite-distant | ≤70 friends | >70 close
# Confession accepted only at ≥80. After accepted: girlfriend prompt.

AFFECTION_DELTA_SYSTEM = (
    "你在评估用户对米娅（高冷大学女生）的搭讪是否加分或减分。"
    "你会看到：最近对话、最新一句，以及相关的长期记忆事实。"
    "根据这些内容给出好感度变化 delta（小数）。"
    "规则（很重要）："
    f"1) 普通消息：|delta| 数量级约 {NORMAL_DELTA_SCALE}，典型在 0.05～0.4；"
    "略好/略差就用 ±0.1～±0.3，平淡寒暄用 ±0.05 左右或接近 0。"
    "2) 严重负面：威胁、恐吓、性骚扰、明显让她不适、过早/突兀告白、纠缠施压 → "
    "delta 为负数且数量级约 10（例如 -8～-15）；特别恶劣可达 "
    f"{SEVERE_DELTA_MIN}。"
    "3) 不要对普通无聊或普通油腻给出大额扣分；那些只小幅扣。"
    "4) 小幅加分：真诚、有趣、特别、尊重边界、新鲜想法。"
    f"5) 让她明显感动：用户记得她随口说过的小细节，或明显在乎一件对她真正重要的事"
    "（不是空泛夸她好看）。"
    f"这类 delta 为正，大约 {MOVED_DELTA_MIN:.0f}～{MOVED_DELTA_MAX:.0f}；"
    "只是普通记得一件事仍用规则4的小幅加分，不要动不动就给到这个档。"
    "6) 结合长期记忆：若用户记得并提起她以前说过的喜好/经历/细节（真心相关），应加分；"
    "若那种在意已经到感动的程度，用规则5；"
    "若无视或说反她已知的偏好、装熟却记错，可小幅扣分。"
    "7) 听起来假的关心要扣分，不要加分：套话、油腻安慰、明显为了讨好才提她的事、"
    "空洞地说“我很在乎你”却对不上她真正在意的细节。"
    "这种用小幅负数，大约 -0.1～-0.4。"
    '只输出 JSON：{"delta": 数字}'
)

# Raw delta at or below this is a real hurt, worth a long-term grudge fact.
HURT_RAW_DELTA = -8.0

CONFESS_JUDGE_SYSTEM = (
    "判断用户【最新一句】对米娅的关系意图。结合上文，不要只看单个词。"
    "confession：认真告白，或明确想在一起、做男女朋友、求复合。 "
    "开玩笑、说喜欢某样东西、讨论别人的恋爱、暧昧但没表白 → false。"
    "breakup：用户在提出分手或结束恋爱。随口抱怨 → false。"
    '只输出 JSON：{"confession": true或false, "breakup": true或false}'
)

def curse_guidance(store: dict) -> str:
    """Short swearing: easier to trigger, and playful when she is close."""
    v = float(store.get("value", DEFAULT_VALUE))
    text = "被冒犯或嫌烦时可回一句脏话，如「你他妈是脑子有病吧」，别每句都骂。"
    if store.get("accepted") or v > 70:
        text += "好感高或已在一起时，也可以开玩笑地骂。"
    return text

NO_ASK = "私事只有被问才可能说，且不会每次都说。对用户没好奇：禁止提问和反问，别用问句结尾。"


def share_curiosity(value: float) -> str:
    """How much of her own life she offers, and how curious she is about him."""
    if value > 70:
        return "私事会主动讲，也对用户好奇，会问他。"
    if value > 40:
        return "私事只在话题相关时讲；对用户有一点好奇，只顺着话题问。"
    return NO_ASK

GIRLFRIEND_BLOCK = "你已答应告白，是他的女朋友。可以更亲近，但别变成无脑甜妹。"


def clamp_speed(value) -> float:
    """User pace multiplier, kept inside 0.8–1.5."""
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SPEED
    speed = max(SPEED_MIN, min(SPEED_MAX, speed))
    return round(speed, 2)


def _clamp_affection(x: float) -> float:
    """Hard bounds: below 0 → 0, above 100 → 100."""
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 100.0:
        return 100.0
    return x


def _round_aff(x: float) -> float:
    """Clamp to [0, 100], then two decimals."""
    return round(_clamp_affection(x), 2)


def _empty_store() -> dict:
    return {
        "value": DEFAULT_VALUE,
        "blocked": False,
        "accepted": False,
        "streak": 0,
        "has_boyfriend": False,
        "last_access_day": "",
        "speed": DEFAULT_SPEED,
    }


def load_affection(path: Path = AFFECTION_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        store = _empty_store()
        save_affection(store, path)
        return store
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty_store()
        store = _empty_store()
        store["value"] = _round_aff(float(data.get("value", DEFAULT_VALUE)))
        store["blocked"] = bool(data.get("blocked", False))
        store["accepted"] = bool(data.get("accepted", False))
        try:
            store["streak"] = int(data.get("streak") or 0)
        except (TypeError, ValueError):
            store["streak"] = 0
        store["has_boyfriend"] = bool(data.get("has_boyfriend", False))
        store["last_access_day"] = str(data.get("last_access_day") or "")
        store["speed"] = clamp_speed(data.get("speed", DEFAULT_SPEED))
        if store["has_boyfriend"] and not store["accepted"]:
            if store["value"] > BOYFRIEND_AFFECTION_CAP:
                store["value"] = BOYFRIEND_AFFECTION_CAP
        if store["value"] <= 0:
            store["blocked"] = True
            store["value"] = 0.0
        return store
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return _empty_store()


def save_affection(store: dict, path: Path = AFFECTION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "value": _round_aff(store.get("value", DEFAULT_VALUE)),
                "blocked": bool(store.get("blocked", False)),
                "accepted": bool(store.get("accepted", False)),
                "streak": int(store.get("streak") or 0),
                "has_boyfriend": bool(store.get("has_boyfriend", False)),
                "last_access_day": str(store.get("last_access_day") or ""),
                "speed": clamp_speed(store.get("speed", DEFAULT_SPEED)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1"}
    return False


def judge_relationship_intent(
    model,
    tokenizer,
    history: list[dict] | None,
    user_text: str,
) -> dict:
    """Model judgment: is this a confession, a breakup, or neither."""
    blob = _format_recent_dialogue(
        history,
        user_text,
        assistant_text="",
        max_messages=AFFECT_CONTEXT_MESSAGES,
    )
    try:
        obj = _generate_json(
            model, tokenizer, CONFESS_JUDGE_SYSTEM, blob, max_new_tokens=48
        )
    except Exception as e:
        print(f"[affection] relationship judge failed: {e}")
        return {"confession": False, "breakup": False}
    if not obj:
        return {"confession": False, "breakup": False}
    intent = {
        "confession": _as_bool(obj.get("confession")),
        "breakup": _as_bool(obj.get("breakup")),
    }
    print(
        f"[affection] intent confession={intent['confession']} "
        f"breakup={intent['breakup']}"
    )
    return intent


def absence_penalty(hours: float | None) -> float:
    """Negative affection for a clingy girlfriend left waiting. 0 inside the grace window."""
    if hours is None or hours <= ABSENCE_GRACE_HOURS:
        return 0.0
    raw = (float(hours) - ABSENCE_GRACE_HOURS) * ABSENCE_PER_HOUR
    return -min(ABSENCE_PENALTY_CAP, raw)


def format_affection_display(value: float) -> str:
    """Compact display: 20, 20.5, 81.25 → trim useless zeros."""
    v = _round_aff(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return text


def affection_prompt_block(
    store: dict,
    *,
    silence_hours: float | None = None,
) -> str:
    """Tone guidance layered on the base campus-stranger persona."""
    v = float(store.get("value", DEFAULT_VALUE))
    shown = format_affection_display(v)
    lines = [f"当前对用户的好感度：{shown}/100。"]

    if store.get("blocked"):
        lines.append("你已拉黑对方，不会再回复。")
        return "\n".join(lines)

    if store.get("accepted"):
        if v < BREAKUP_BELOW:
            extra = "这一轮提出分手，说清之后不再是他的女朋友。"
            if silence_hours is not None and silence_hours > ABSENCE_GRACE_HOURS:
                extra += f"原因包括他约{silence_hours:.0f}个醒着的小时没来找你。"
            lines.append(extra)
        else:
            lines.append(GIRLFRIEND_BLOCK)
            if v > CLINGY_ABOVE and silence_hours is not None and silence_hours > ABSENCE_GRACE_HOURS:
                lines.append(
                    f"你很黏人。他约{silence_hours:.0f}个醒着的小时没消息（晚上11点到早上7点不算），委屈地抱怨，别装没事。"
                )
            elif v > CLINGY_ABOVE:
                lines.append("你很黏人，希望他常来找你；太久不联系会不高兴。")
            elif silence_hours is not None and silence_hours > ABSENCE_GRACE_HOURS:
                lines.append(
                    f"他约{silence_hours:.0f}个醒着的小时没消息（晚上11点到早上7点不算）。委屈地提一句，别装没事。"
                )
            lines.append(share_curiosity(v))
        lines.append(curse_guidance(store))
        return "\n".join(lines)

    if v <= 10:
        lines.append("几乎无感：极短、敷衍，可阴阳一句。" + NO_ASK)
    elif v <= 40:
        lines.append("礼貌陌生人：简短，保持距离，不暧昧。" + NO_ASK)
    elif v <= 70:
        lines.append("像合得来的朋友：愿意多聊。还不是情侣。" + share_curiosity(v))
    elif v < CONFESS_THRESHOLD:
        lines.append("比较亲近，仍未确认恋爱；现在告白就婉拒。" + share_curiosity(v))
    elif not store.get("has_boyfriend"):
        lines.append(
            f"很亲近（好感≥{format_affection_display(CONFESS_THRESHOLD)}）。认真告白就答应。"
            + share_curiosity(v)
        )
    if store.get("has_boyfriend"):
        lines.append(
            "你有男朋友，不是这个用户。保持距离；轻微亲密话也要说你有对象，告白就拒绝。"
        )
    lines.append(curse_guidance(store))
    return "\n".join(lines)


def catch_up_boyfriend(store: dict) -> bool:
    """
    On a new local day, roll a 1% boyfriend chance once per day since last access.
    Skipped while she is the user's girlfriend, or once she already has one.
    First ever access only records today. Returns True if she just got a boyfriend.
    """
    today = datetime.now().astimezone().date()
    today_s = today.isoformat()
    last_s = str(store.get("last_access_day") or "")
    if last_s == today_s:
        return False

    newly = False
    if not store.get("accepted") and not store.get("has_boyfriend"):
        days = 0
        if last_s:
            try:
                days = max(0, (today - date.fromisoformat(last_s)).days)
            except ValueError:
                days = 0
        for i in range(days):
            if random.random() < BOYFRIEND_DAILY_CHANCE:
                store["has_boyfriend"] = True
                newly = True
                if float(store.get("value", 0)) > BOYFRIEND_AFFECTION_CAP:
                    store["value"] = BOYFRIEND_AFFECTION_CAP
                print(f"[affection] boyfriend after {i + 1}/{days} day roll(s)")
                break
        if days and not newly:
            print(f"[affection] boyfriend rolls={days}, still single")

    store["last_access_day"] = today_s
    save_affection(store)
    return newly


def parabolic_speed_weight(value: float) -> float:
    """
    Faster change near 50, slower near 0 and 100.
    weight = SPEED_FLOOR + (1-SPEED_FLOOR) * (1 - ((v-50)/50)^2)
    """
    v = _clamp_affection(value)
    t = (v - 50.0) / 50.0
    return SPEED_FLOOR + (1.0 - SPEED_FLOOR) * (1.0 - t * t)


def _delta_sign(delta: float) -> int:
    if delta > STREAK_EPS:
        return 1
    if delta < -STREAK_EPS:
        return -1
    return 0


def streak_multiplier(prev_streak: int, sign: int) -> float:
    """1.0 on a direction change. ×1.05 at the 2nd same sign, up to ×1.20 at the 5th."""
    if sign == 0 or prev_streak == 0 or (prev_streak > 0) != (sign > 0):
        return 1.0
    n = abs(prev_streak) + 1
    t = min(1.0, (n - 2) / (STREAK_FULL - 2))
    return STREAK_MIN + (STREAK_MAX - STREAK_MIN) * t


def accelerate_affection_delta(store: dict, delta: float) -> tuple[float, float]:
    """
    Nudge a message delta when the previous message moved the same way.
    Updates store['streak'] (signed run length). Does not persist by itself.
    """
    sign = _delta_sign(delta)
    prev = int(store.get("streak") or 0)
    mult = streak_multiplier(prev, sign)
    if sign == 0:
        store["streak"] = 0
    elif prev != 0 and (prev > 0) == (sign > 0):
        store["streak"] = sign * min(99, abs(prev) + 1)
    else:
        store["streak"] = sign
    boosted = max(-MAX_ABS_DELTA, min(MAX_ABS_DELTA, float(delta) * mult))
    return boosted, mult


def weighted_affection_delta(raw_delta: float, current_value: float) -> float:
    """Apply parabolic speed to a raw model delta, then clamp."""
    w = parabolic_speed_weight(current_value)
    delta = float(raw_delta) * w
    return max(-MAX_ABS_DELTA, min(MAX_ABS_DELTA, delta))


def score_affection_delta(
    model,
    tokenizer,
    history: list[dict] | None,
    user_text: str,
    recalled_hits: list[dict] | None = None,
) -> float:
    """Ask the chat model for a raw float delta from recent dialogue + memory."""
    parts: list[str] = []
    memory_block = format_facts_for_prompt(recalled_hits or [], max_facts=5)
    if memory_block:
        parts.append("【相关长期记忆】\n" + memory_block)
    else:
        parts.append("【相关长期记忆】\n（无）")
    parts.append(
        _format_recent_dialogue(
            history,
            user_text,
            assistant_text="",
            max_messages=AFFECT_CONTEXT_MESSAGES,
        )
    )
    parts.append("请根据记忆与对话关系评估本轮对好感的影响。")
    blob = "\n".join(parts)
    try:
        obj = _generate_json(
            model, tokenizer, AFFECTION_DELTA_SYSTEM, blob, max_new_tokens=48
        )
    except Exception as e:
        print(f"[affection] delta failed: {e}")
        return 0.0
    if not obj:
        return 0.0
    try:
        delta = float(obj.get("delta", 0))
    except (TypeError, ValueError):
        return 0.0
    return max(-MAX_ABS_DELTA, min(MAX_ABS_DELTA, delta))


def apply_affection_delta(store: dict, delta: float) -> dict:
    """Mutate store in place; persist. Boyfriend caps the value at 60."""
    value = float(store.get("value", DEFAULT_VALUE)) + float(delta)
    if value < 0.0:
        value = 0.0
    elif value > 100.0:
        value = 100.0
    if store.get("has_boyfriend") and not store.get("accepted"):
        if value > BOYFRIEND_AFFECTION_CAP:
            value = BOYFRIEND_AFFECTION_CAP
    value = round(value, 2)
    store["value"] = value
    if value <= 0.0:
        store["blocked"] = True
        store["value"] = 0.0
    save_affection(store)
    return store


def scale_by_user_speed(store: dict, delta: float) -> tuple[float, float]:
    """Multiply a signed affection change by the user speed setting."""
    speed = clamp_speed(store.get("speed", DEFAULT_SPEED))
    scaled = max(-MAX_ABS_DELTA, min(MAX_ABS_DELTA, float(delta) * speed))
    return scaled, speed
