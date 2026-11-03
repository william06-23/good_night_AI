"""
CLI for the directed text game — no Vite/browser required.
"""

from __future__ import annotations

import time

from .engine import LINE_BEAT_SECONDS, TextGameEngine


def _drain_auto(engine: TextGameEngine, result: dict) -> None:
    """Follow auto_continue beats (CLI has no parallel interrupt typing)."""
    while result.get("auto_continue"):
        beat = float(result.get("beat_seconds") or LINE_BEAT_SECONDS)
        time.sleep(max(0.5, beat))
        try:
            result = engine.advance_scene()
        except (ValueError, RuntimeError) as e:
            print(f"[error] {e}")
            return
        _print_events(result.get("events") or [])


def _print_events(events: list) -> None:
    for ev in events or []:
        t = ev.get("type")
        if t == "cue":
            to = "、".join(ev.get("to") or []) or "—"
            print(f"[cue] {ev.get('speaker')} → {to}  ({ev.get('reason') or ''})")
        elif t == "wait_player":
            sug = "、".join(ev.get("suggested_to") or []) or "任选角色"
            print(f"[旁白点到你了] 建议对象：{sug}")
        elif t == "scene_start":
            print(f"\n—— 第 {ev.get('scene_index')}/{ev.get('max_scenes')} 幕 ——\n")
        elif t == "story_complete":
            print(ev.get("text") or "故事结束。")
        elif t in {"narrator", "dialogue", "player"} or ev.get("text"):
            speaker = ev.get("speaker") or "?"
            to = ev.get("to") or []
            text = (ev.get("text") or "").strip()
            if not text and t not in {"narrator", "dialogue", "player"}:
                continue
            if speaker == "Narrator" or t == "narrator":
                print(f"\n旁白：{text}\n")
            elif speaker == "Player" or t == "player":
                print(f"你 → {'、'.join(to)}：{text}")
            else:
                print(f"{speaker} → {'、'.join(to) or '?'}：{text}")


def _pick_addressees(cast: list[str], suggested: list[str] | None = None) -> list[str]:
    print("\n对话对象（可多选，输入编号，逗号分隔，例如 1,3）：")
    for i, name in enumerate(cast, 1):
        mark = " *" if suggested and name in suggested else ""
        print(f"  {i}. {name}{mark}")
    raw = input("对象 [默认建议或第1个]: ").strip()
    if not raw:
        if suggested:
            return [n for n in suggested if n in cast] or cast[:1]
        return cast[:1]
    out: list[str] = []
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(cast) and cast[idx] not in out:
                out.append(cast[idx])
        elif part in cast and part not in out:
            out.append(part)
    return out or cast[:1]


def main() -> None:
    engine = TextGameEngine()
    print("AI Text Game CLI (no browser)")
    print("Choose model: 1 = 7B   2 = 14B")
    choice = input("Model [1]: ").strip() or "1"
    model_key = "14b" if choice == "2" else "7b"
    print(f"Loading {model_key}…")
    engine.load(model_key)

    lang_raw = input("Language [zh/en] (default zh): ").strip().lower() or "zh"
    language = "en" if lang_raw.startswith("en") else "zh"
    theme = input(
        "Theme (~8–80 chars, fed into setup; Enter to skip): "
    ).strip() or None
    print("Generating story…")
    result = engine.generate_story(theme=theme, language=language)
    story = result["story"]
    print("\n===== SETUP =====")
    print(f"Player: {story['player'].get('name')} — {story['player'].get('identity')}")
    print(f"Goal: {story['player'].get('goal')}")
    print("\nBackground:\n", story.get("background", "")[:800], "…\n", sep="")
    for c in story.get("characters") or []:
        print(f"  - {c['name']} ({c['slot']}): {c.get('description', '')}")

    input("\nPress Enter to start scene 1…")
    intro = engine.start_intro()
    _print_events(intro.get("events") or [])
    _drain_auto(engine, intro)

    print(
        "\nCommands: Enter = advance cue | type text = your interrupt | "
        "quit = exit"
    )

    while True:
        story = engine.story.public_dict() if engine.story else {}
        if story.get("story_complete"):
            print("\n[story complete]")
            break

        scene = (
            f"[scene {story.get('scene_index')}/{story.get('max_scenes')} | "
            f"{story.get('message_count_in_scene')}/{story.get('scene_message_cap')}]"
        )
        waiting = story.get("waiting_for_player")
        prompt = f"\n{scene} {'(your turn) ' if waiting else ''}You / Enter=advance: "
        user = input(prompt).strip()

        if user.lower() in {"quit", "exit", "q", "晚安"}:
            print("Bye.")
            break

        if not user:
            if waiting:
                print("Narrator is waiting for you — type a line (or checkboxes via numbers after).")
                continue
            try:
                adv = engine.advance_scene()
            except (ValueError, RuntimeError) as e:
                print(f"[error] {e}")
                continue
            _print_events(adv.get("events") or [])
            _drain_auto(engine, adv)
            continue

        cast = story.get("cast_names") or []
        to = _pick_addressees(cast, story.get("suggested_to"))
        try:
            out = engine.chat(user, to=to)
        except (ValueError, RuntimeError) as e:
            print(f"[error] {e}")
            continue
        _print_events(out.get("events") or [])
        _drain_auto(engine, out)


if __name__ == "__main__":
    main()
