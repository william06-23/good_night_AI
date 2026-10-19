"""
Plain chat transcript for the user to read later.

Not used for AI recall, gating, or long-term facts — append-only JSONL.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CHAT_LOG_PATH = DATA_DIR / "chat_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return str(uuid.uuid4())


def append_log_event(
    role: str,
    content: str,
    session_id: str,
    path: Path = CHAT_LOG_PATH,
    *,
    blocked: bool = False,
) -> None:
    """Append one message (or session marker) as a JSON line."""
    content = (content or "").strip()
    if role != "session" and not content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": _now_iso(),
        "session_id": session_id,
        "role": role,
        "content": content,
    }
    if blocked and role == "user":
        row["blocked"] = True
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_log(path: Path = CHAT_LOG_PATH) -> None:
    """Erase the on-disk transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def start_session(session_id: str, path: Path = CHAT_LOG_PATH) -> None:
    append_log_event("session", "start", session_id, path=path)


def log_turn(
    user_text: str,
    assistant_text: str,
    session_id: str,
    path: Path = CHAT_LOG_PATH,
    *,
    blocked: bool = False,
) -> None:
    """Save one user + Mia exchange as-is. Blocked turns skip the assistant line."""
    append_log_event(
        "user", user_text, session_id, path=path, blocked=blocked
    )
    if not blocked and (assistant_text or "").strip():
        append_log_event("assistant", assistant_text, session_id, path=path)


def _ts_to_ms(ts_iso: str | None) -> int | None:
    if not ts_iso:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_iso).replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def awake_hours_between(start: datetime, end: datetime) -> float:
    """Hours between two times, skipping local 11:00 PM–7:00 AM."""
    if end <= start:
        return 0.0
    local_tz = datetime.now().astimezone().tzinfo
    start = start.astimezone(local_tz)
    end = end.astimezone(local_tz)
    total = 0.0
    day = start.date()
    last = end.date()
    while day <= last:
        awake_start = datetime.combine(day, time(7, 0), tzinfo=local_tz)
        awake_end = datetime.combine(day, time(23, 0), tzinfo=local_tz)
        seg_start = max(start, awake_start)
        seg_end = min(end, awake_end)
        if seg_end > seg_start:
            total += (seg_end - seg_start).total_seconds() / 3600.0
        day += timedelta(days=1)
    return total


def hours_since_last_user_message(path: Path = CHAT_LOG_PATH) -> float | None:
    """Waking hours since the previous user line. Sleep (11 PM–7 AM local) is skipped."""
    if not path.is_file():
        return None
    last_ts: str | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("role") == "user" and row.get("ts"):
                last_ts = str(row["ts"])
    if not last_ts:
        return None
    try:
        dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return awake_hours_between(dt, datetime.now(timezone.utc))


def load_messages(
    path: Path = CHAT_LOG_PATH,
    max_events: int = 2000,
) -> list[dict]:
    """
    Read the human transcript for the UI.

    Returns user/assistant rows plus session break markers.
    Not used for model context or recall.
    """
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = row.get("role")
            if role not in ("user", "assistant", "session"):
                continue
            ts_ms = _ts_to_ms(row.get("ts"))
            if role == "session":
                rows.append(
                    {
                        "role": "system",
                        "kind": "session",
                        "content": row.get("content") or "start",
                        "ts": ts_ms,
                    }
                )
            else:
                content = (row.get("content") or "").strip()
                if not content:
                    continue
                entry = {
                    "role": role,
                    "content": content,
                    "ts": ts_ms,
                }
                if role == "user" and row.get("blocked"):
                    entry["blocked"] = True
                rows.append(entry)
    if max_events > 0 and len(rows) > max_events:
        rows = rows[-max_events:]
    return rows
