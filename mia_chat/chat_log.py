"""
Plain chat transcript for the user to read later.

Not used for AI recall, gating, or long-term facts — append-only JSONL.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def start_session(session_id: str, path: Path = CHAT_LOG_PATH) -> None:
    append_log_event("session", "start", session_id, path=path)


def log_turn(
    user_text: str,
    assistant_text: str,
    session_id: str,
    path: Path = CHAT_LOG_PATH,
) -> None:
    """Save one user + Mia exchange as-is."""
    append_log_event("user", user_text, session_id, path=path)
    append_log_event("assistant", assistant_text, session_id, path=path)
