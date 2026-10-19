"""
Fixed campus week for Mia, plus occasional one-off events stored as memories.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from .memory import (
    MEMORY_PATH,
    _generate_json,
    add_fact,
    find_string_duplicate,
    save_memory,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
CAMPUS_PATH = DATA_DIR / "campus_life.json"

# Proactive sharing starts once she is at least a friend (affection above 40).
FRIEND_ABOVE = 40.0
EVENT_KEEP_DAYS = 21
ROLL_CHANCE = 0.4

# (start_hour, end_hour, what she is doing). Local time. 23:00–07:00 is sleep.
_WEEKDAY_MORNING = (8, 12, "上午在上课，人在教学楼")
_LUNCH = (12, 14, "在食堂吃午饭")
_DINNER = (17, 19, "在食堂吃晚饭")
_DORM_NIGHT = (19, 23, "回宿舍了，看书或刷手机")

WEEKLY: dict[int, list[tuple[int, int, str]]] = {
    0: [
        _WEEKDAY_MORNING,
        _LUNCH,
        (14, 17, "下午在图书馆写新闻学作业"),
        _DINNER,
        _DORM_NIGHT,
    ],
    1: [
        (8, 12, "上午有课，在教学楼"),
        _LUNCH,
        (14, 16, "下午去学生会开短会"),
        (16, 17, "会后在校园里走走"),
        _DINNER,
        _DORM_NIGHT,
    ],
    2: [
        _WEEKDAY_MORNING,
        _LUNCH,
        (14, 17, "下午没课，在宿舍看电影或补觉"),
        _DINNER,
        (19, 23, "晚上去操场散步，然后回宿舍"),
    ],
    3: [
        (8, 10, "上午第一节有课"),
        (10, 12, "课间在咖啡店坐着"),
        _LUNCH,
        (14, 17, "下午在图书馆占座看书"),
        _DINNER,
        _DORM_NIGHT,
    ],
    4: [
        _WEEKDAY_MORNING,
        _LUNCH,
        (14, 17, "下午和同学做小组展示的准备"),
        _DINNER,
        (19, 23, "周五晚上，可能在宿舍，不太想出门"),
    ],
    5: [
        (8, 12, "周六早上睡懒觉，中午才出门"),
        (12, 14, "brunch，在校外或食堂"),
        (14, 18, "下午逛校园或去市区走走"),
        (18, 23, "晚上回宿舍，看剧"),
    ],
    6: [
        (8, 12, "周日早上还在宿舍"),
        (12, 14, "午饭，准备下周的东西"),
        (14, 18, "下午在图书馆赶作业"),
        (18, 23, "晚上回宿舍，早点休息"),
    ],
}

_WEEKDAY_NAMES = "一二三四五六日"

EVENT_SYSTEM = (
    "你在给大学生米娅编一件即将发生的校园小事。"
    "类型三选一：一场考试、一次社团或校园活动、一件她自己觉得还不错的小事。"
    "日期必须是给定候选日期之一。start_hour 和 end_hour 是整数，"
    "落在 8 到 21 点之间，且 end_hour 比 start_hour 至少大 1、最多大 4。"
    "不要和已有事件同一天、时段重叠。"
    "text 是一句客观陈述，主语是米娅，写清日期和她做了什么或将做什么。"
    '只输出 JSON：{"date":"YYYY-MM-DD","start_hour":14,"end_hour":16,"text":"米娅在……"}'
)


def _today() -> datetime:
    return datetime.now().astimezone()


def _empty() -> dict:
    return {"last_roll_date": "", "events": []}


def load_campus(path: Path = CAMPUS_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        store = _empty()
        save_campus(store, path)
        return store
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _empty()
        events = data.get("events")
        if not isinstance(events, list):
            events = []
        return {
            "last_roll_date": str(data.get("last_roll_date") or ""),
            "events": [e for e in events if isinstance(e, dict)],
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return _empty()


def save_campus(store: dict, path: Path = CAMPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def _prune(store: dict, today) -> None:
    cutoff = (today - timedelta(days=EVENT_KEEP_DAYS)).date().isoformat()
    store["events"] = [
        e
        for e in store.get("events") or []
        if str(e.get("date") or "") >= cutoff
    ]


def _overlaps(event: dict, day: str, hour: int) -> bool:
    if str(event.get("date")) != day:
        return False
    try:
        start = int(event.get("start_hour"))
        end = int(event.get("end_hour"))
    except (TypeError, ValueError):
        return False
    return start <= hour < end


def weekly_activity(now: datetime | None = None) -> str:
    now = now or _today()
    hour = now.hour
    if hour >= 23 or hour < 7:
        return "在宿舍睡觉"
    for start, end, text in WEEKLY.get(now.weekday(), []):
        if start <= hour < end:
            return text
    return "在学校里，没什么特别的安排"


def activity_now(store: dict | None = None, now: datetime | None = None) -> str:
    now = now or _today()
    store = store if store is not None else load_campus()
    day = now.date().isoformat()
    for event in store.get("events") or []:
        if _overlaps(event, day, now.hour):
            text = (event.get("text") or "").strip()
            if text:
                return text
    return weekly_activity(now)


def recent_event_lines(store: dict, days: int = 14, limit: int = 5) -> list[str]:
    today = _today().date()
    cutoff = today - timedelta(days=days)
    rows = []
    for event in store.get("events") or []:
        try:
            day = datetime.fromisoformat(str(event.get("date"))).date()
        except ValueError:
            continue
        if day < cutoff or day > today + timedelta(days=7):
            continue
        text = (event.get("text") or "").strip()
        if text:
            rows.append((day, text))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in rows[:limit]]


def schedule_prompt_block(affection: float, store: dict | None = None) -> str:
    """What she is doing now, and whether she may bring campus life up herself."""
    now = _today()
    store = store if store is not None else load_campus()
    label = f"周{_WEEKDAY_NAMES[now.weekday()]} {now.strftime('%H:%M')}"
    lines = [
        f"当前校园时间：{label}。",
        f"你此刻在做的事：{activity_now(store, now)}。",
        "被问到在干嘛时按这个回答，不要临时编一个完全不同的行程。",
    ]
    if float(affection) > FRIEND_ABOVE:
        lines.append(
            "你们至少已经算朋友：话题沾边时可以主动提一句自己的校园事"
            "（考试、活动、最近做过的事），偶尔提，不要每句话都说。"
        )
        recent = recent_event_lines(store)
        if recent:
            lines.append("你最近的校园事（可以自然提起，像“上周我也……”）：")
            lines.extend(f"- {text}" for text in recent)
    else:
        lines.append(
            "还没到朋友：只有被问到时才简短说自己在做什么，不要主动分享校园生活。"
        )
    return "\n".join(lines)


def _candidate_dates(today) -> list[str]:
    return [
        (today + timedelta(days=offset)).date().isoformat()
        for offset in range(0, 7)
    ]


def _event_conflicts(store: dict, day: str, start: int, end: int) -> bool:
    for event in store.get("events") or []:
        if str(event.get("date")) != day:
            continue
        try:
            other_start = int(event.get("start_hour"))
            other_end = int(event.get("end_hour"))
        except (TypeError, ValueError):
            continue
        if start < other_end and end > other_start:
            return True
    return False


def maybe_roll_campus_event(
    model,
    tokenizer,
    memory_store: dict,
    retriever=None,
    path: Path = CAMPUS_PATH,
    memory_path: Path = MEMORY_PATH,
) -> dict | None:
    """
    At most once per local day, sometimes invent one campus event and store it
    both on the schedule and in long-term memory.
    """
    now = _today()
    store = load_campus(path)
    _prune(store, now)
    today_s = now.date().isoformat()
    if store.get("last_roll_date") == today_s:
        save_campus(store, path)
        return None
    store["last_roll_date"] = today_s
    has_upcoming = any(
        str(e.get("date") or "") >= today_s for e in store.get("events") or []
    )
    if has_upcoming and random.random() > ROLL_CHANCE:
        save_campus(store, path)
        print("[campus] no new event today")
        return None

    dates = _candidate_dates(now)
    existing = [
        f"{e.get('date')} {e.get('start_hour')}-{e.get('end_hour')} {e.get('text')}"
        for e in store.get("events") or []
    ]
    blob = (
        "候选日期："
        + "、".join(dates)
        + "\n已有事件：\n"
        + ("\n".join(existing) if existing else "（无）")
    )
    try:
        obj = _generate_json(model, tokenizer, EVENT_SYSTEM, blob, max_new_tokens=120)
    except Exception as e:
        print(f"[campus] event roll failed: {e}")
        save_campus(store, path)
        return None
    if not obj:
        save_campus(store, path)
        return None
    day = str(obj.get("date") or "")
    text = (obj.get("text") or "").strip() if isinstance(obj.get("text"), str) else ""
    try:
        start = int(obj.get("start_hour"))
        end = int(obj.get("end_hour"))
    except (TypeError, ValueError):
        save_campus(store, path)
        return None
    if (
        day not in dates
        or not text
        or not (8 <= start < end <= 22)
        or (end - start) > 4
        or _event_conflicts(store, day, start, end)
    ):
        print("[campus] discarded invalid event")
        save_campus(store, path)
        return None

    event = {
        "date": day,
        "start_hour": start,
        "end_hour": end,
        "text": text,
    }
    store.setdefault("events", []).append(event)
    save_campus(store, path)

    if not find_string_duplicate(memory_store, text):
        entry = add_fact(
            memory_store,
            text,
            certainty=0.9,
            source="campus",
            session_new=False,
        )
        if entry is not None:
            save_memory(memory_store, memory_path)
            if retriever is not None:
                retriever.add_fact_embedding(entry)
    print(f"[campus] event {day} {start}-{end}: {text}")
    return event
