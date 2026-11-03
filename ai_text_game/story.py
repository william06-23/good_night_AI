"""
Directed text-game story: setup generator, narrator prose/cues, character lines.
Uses the same loaded instruct model with different prompts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .memory import (
    NARRATOR_KEY,
    PLAYER_KEY,
    _extract_json_object,
    _generate_json,
    format_facts_for_prompt,
)
from .prompts import normalize_lang, story_prompts

STORY_MAX_NEW_TOKENS = 1024
NARRATOR_MAX_NEW_TOKENS = 256
CUE_MAX_NEW_TOKENS = 128
CHARACTER_MAX_NEW_TOKENS = 256
MAX_SCENES = 5
SCENE_DIALOGUE_MIN = 8
SCENE_DIALOGUE_MAX = 15
# Prefer NPCs for this many dialogue lines before Player is normally cued.
PLAYER_CUE_AFTER_NPC_LINES = 3
# Soft length guidance for theme input (Chinese chars / short English phrases).
THEME_RECOMMENDED_MIN_CHARS = 8
THEME_RECOMMENDED_MAX_CHARS = 80
# Legacy alias used in a few places
SCENE_MESSAGE_CAP = SCENE_DIALOGUE_MAX

@dataclass
class CharacterCard:
    name: str
    slot: str
    description: str
    persona: str = ""
    # Detailed profile from the persona-generator pass (stored, reused on every line).
    persona_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def acting_brief(self, lang: str = "zh") -> str:
        """Text injected into the character-line prompt."""
        en = normalize_lang(lang) == "en"
        join_h = ", " if en else "、"
        parts = [
            (f"Role in plot: {self.description}" if en else f"剧情身份：{self.description}"),
            (f"Slot: {self.slot}" if en else f"槽位：{self.slot}"),
        ]
        if self.persona:
            parts.append(
                f"Persona summary: {self.persona}" if en else f"人设摘要：{self.persona}"
            )
        prof = self.persona_profile or {}
        labels = (
            {
                "summary": "Summary",
                "personality": "Personality",
                "speech_style": "Speech style",
                "hobbies": "Hobbies",
                "motivation": "Motivation",
                "quirks": "Quirks",
            }
            if en
            else {
                "summary": "总括",
                "personality": "性格",
                "speech_style": "说话习惯",
                "hobbies": "爱好",
                "motivation": "动机",
                "quirks": "小习惯/特点",
            }
        )
        if prof.get("summary"):
            parts.append(f"{labels['summary']}：{prof['summary']}")
        if prof.get("personality"):
            parts.append(f"{labels['personality']}：{prof['personality']}")
        if prof.get("speech_style"):
            parts.append(f"{labels['speech_style']}：{prof['speech_style']}")
        hobbies = prof.get("hobbies")
        if isinstance(hobbies, list) and hobbies:
            parts.append(f"{labels['hobbies']}：" + join_h.join(str(h) for h in hobbies))
        elif isinstance(hobbies, str) and hobbies.strip():
            parts.append(f"{labels['hobbies']}：{hobbies}")
        if prof.get("motivation"):
            parts.append(f"{labels['motivation']}：{prof['motivation']}")
        if prof.get("quirks"):
            parts.append(f"{labels['quirks']}：{prof['quirks']}")
        return "\n".join(parts)


@dataclass
class StoryState:
    background: str = ""
    player: dict[str, str] = field(default_factory=dict)
    character_dict: dict[str, CharacterCard] = field(default_factory=dict)
    scene_index: int = 1
    message_count_in_scene: int = 0  # all log lines this scene
    dialogue_count_in_scene: int = 0  # player+NPC lines only (not narrator)
    scene_log: list[dict[str, Any]] = field(default_factory=list)
    current_narration: str = ""
    waiting_for_player: bool = False
    suggested_to: list[str] = field(default_factory=list)
    story_complete: bool = False
    max_scenes: int = MAX_SCENES
    scene_dialogue_min: int = SCENE_DIALOGUE_MIN
    scene_dialogue_max: int = SCENE_DIALOGUE_MAX
    scene_message_cap: int = SCENE_DIALOGUE_MAX  # alias for UI
    theme: str = ""
    # Subset of cast present in the current scene (not everyone every scene).
    present_characters: list[str] = field(default_factory=list)
    # How the player's last line entered: "cued" | "interrupt" | "".
    last_player_mode: str = ""
    # Story content language: "zh" | "en"
    language: str = "zh"
    # Queued NPC reactions after a player line: [{name, mode, user_text}, ...]
    pending_reactions: list[dict[str, Any]] = field(default_factory=list)
    # After a line, next advance should open the next scene (cast + bridge).
    pending_scene_bridge: bool = False

    def cast_names(self) -> list[str]:
        return list(self.character_dict.keys())

    def present_names(self) -> list[str]:
        if self.present_characters:
            return list(self.present_characters)
        return self.cast_names()

    def public_dict(self) -> dict[str, Any]:
        return {
            "background": self.background,
            "player": self.player,
            "characters": [c.to_dict() for c in self.character_dict.values()],
            "scene_index": self.scene_index,
            "message_count_in_scene": self.message_count_in_scene,
            "dialogue_count_in_scene": self.dialogue_count_in_scene,
            "scene_log": list(self.scene_log[-80:]),
            "current_narration": self.current_narration,
            "waiting_for_player": self.waiting_for_player,
            "suggested_to": list(self.suggested_to),
            "story_complete": self.story_complete,
            "max_scenes": self.max_scenes,
            "scene_dialogue_min": self.scene_dialogue_min,
            "scene_dialogue_max": self.scene_dialogue_max,
            "scene_message_cap": self.scene_dialogue_max,
            "cast_names": self.cast_names(),
            "present_characters": list(self.present_names()),
            "theme": self.theme,
            "last_player_mode": self.last_player_mode,
            "language": normalize_lang(self.language),
            "pending_reactions": len(self.pending_reactions),
            "pending_scene_bridge": bool(self.pending_scene_bridge),
            "theme_hint": (
                f"Recommended theme length: about {THEME_RECOMMENDED_MIN_CHARS}–"
                f"{THEME_RECOMMENDED_MAX_CHARS} characters "
                "(one short phrase or 1–2 sentences)."
            ),
        }


def iter_chat_text(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.7,
) -> Iterator[str]:
    """Yield text chunks as the model generates (llama.cpp or transformers)."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(model, "create_chat_completion"):
        stream = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=True,
        )
        for chunk in stream:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                yield delta
        return

    import threading

    from transformers import TextIteratorStreamer

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0.01
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    def _run() -> None:
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_run)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join()


def generate_chat_text(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.7,
) -> str:
    return "".join(
        iter_chat_text(
            model,
            tokenizer,
            system,
            user,
            max_new_tokens,
            temperature=temperature,
        )
    ).strip()


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip())


def lookup_character(
    character_dict: dict[str, CharacterCard], speaker: str
) -> CharacterCard | None:
    """Resolve speaker via exact key, then for-loop / contains match."""
    raw = (speaker or "").strip()
    if not raw:
        return None
    if raw in character_dict:
        return character_dict[raw]
    target = _normalize_name(raw).casefold()
    for name, card in character_dict.items():
        if _normalize_name(name).casefold() == target:
            return card
    # Model sometimes returns "林晓（女主）" or "角色林晓"
    for name, card in character_dict.items():
        n = _normalize_name(name).casefold()
        if n and (n in target or target in n):
            return card
    return None


def resolve_cue_speaker(state: StoryState, speaker: str) -> tuple[str, CharacterCard | None]:
    """
    Map raw cue speaker → ('Player'|'Narrator'|canonical name, card|None).
    Returns which bucket to use for the next beat.
    """
    raw = (speaker or "").strip()
    if not raw:
        return PLAYER_KEY, None
    if raw in {"Player", "玩家", "User", "user", "YOU", "You"}:
        return PLAYER_KEY, None
    if raw in {"Narrator", "旁白", "narrator", "NARRATOR"}:
        return NARRATOR_KEY, None
    # In-story player name (e.g. 李晨) counts as Player
    pname = (state.player.get("name") or "").strip()
    if pname and (
        raw == pname
        or _normalize_name(raw).casefold() == _normalize_name(pname).casefold()
    ):
        return PLAYER_KEY, None
    card = lookup_character(state.character_dict, raw)
    if card is not None:
        return card.name, card
    return raw, None


def _validate_story_obj(
    obj: dict, *, language: str = "zh"
) -> tuple[str, dict[str, str], list[CharacterCard]]:
    background = str(obj.get("background") or "").strip()
    player_raw = obj.get("player") if isinstance(obj.get("player"), dict) else {}
    default_name = "Player" if normalize_lang(language) == "en" else "玩家"
    player = {
        "name": str(player_raw.get("name") or default_name).strip() or default_name,
        "identity": str(player_raw.get("identity") or "").strip(),
        "goal": str(player_raw.get("goal") or "").strip(),
    }
    chars_raw = obj.get("characters") if isinstance(obj.get("characters"), list) else []
    cards: list[CharacterCard] = []
    for item in chars_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        slot = str(item.get("slot") or "supporting").strip()
        if slot not in {"competitor", "goal_person", "supporting"}:
            slot = "supporting"
        if not name or name in (NARRATOR_KEY, PLAYER_KEY):
            continue
        cards.append(
            CharacterCard(
                name=name,
                slot=slot,
                description=str(item.get("description") or "").strip(),
                persona=str(item.get("persona") or item.get("description") or "").strip(),
            )
        )
    if not background:
        raise ValueError("Story missing background")
    # Dedupe by name keeping first
    seen: set[str] = set()
    unique: list[CharacterCard] = []
    for c in cards:
        if c.name in seen:
            continue
        seen.add(c.name)
        unique.append(c)
    unique = _ensure_min_cast(unique, language=language, player_name=player["name"])
    if len(unique) < 4:
        raise ValueError(f"Need 4–5 characters, got {len(unique)}")
    return background, player, unique[:5]


def _ensure_min_cast(
    cards: list[CharacterCard],
    *,
    language: str = "zh",
    player_name: str = "",
) -> list[CharacterCard]:
    """
    Model often returns 3 NPCs; pad to 4–5 and guarantee competitor + goal_person.
    """
    en = normalize_lang(language) == "en"
    out = list(cards)
    names = {c.name for c in out}
    if player_name:
        names.add(player_name)

    def _has(slot: str) -> bool:
        return any(c.slot == slot for c in out)

    # Promote/fix slots if the model omitted required roles.
    if not _has("goal_person") and out:
        out[0].slot = "goal_person"
    if not _has("competitor"):
        for c in out:
            if c.slot == "supporting":
                c.slot = "competitor"
                break
        else:
            if len(out) >= 2 and out[1].slot != "goal_person":
                out[1].slot = "competitor"

    fillers = (
        [
            ("Elena Voss", "supporting", "A sharp-eyed bystander who knows a secret about the goal."),
            ("Marcus Quinn", "supporting", "An opportunistic contact who sells information for favors."),
            ("Ivy Chen", "competitor", "A rival chasing the same prize as the player."),
            ("Noah Drake", "goal_person", "The person the player most needs to reach."),
        ]
        if en
        else [
            ("周晚星", "supporting", "知情的旁观者，掌握与目标有关的一条线索。"),
            ("沈渡", "supporting", "游走两头的中间人，爱用人情交换情报。"),
            ("韩弈", "competitor", "与玩家争夺同一目标的对手。"),
            ("苏念", "goal_person", "与玩家目标直接相关的关键人物。"),
        ]
    )

    # Prefer padding as supporting once required slots exist.
    for name, slot, desc in fillers:
        if len(out) >= 4:
            break
        if name in names:
            continue
        need_goal = not _has("goal_person")
        need_comp = not _has("competitor")
        use_slot = slot
        if need_goal:
            use_slot = "goal_person"
        elif need_comp:
            use_slot = "competitor"
        else:
            use_slot = "supporting"
        out.append(
            CharacterCard(
                name=name,
                slot=use_slot,
                description=desc,
                persona=desc,
            )
        )
        names.add(name)

    # One more supporting if still only 4 is fine; optionally add 5th for richer cast.
    if len(out) == 4:
        for name, slot, desc in fillers:
            if name in names:
                continue
            if slot != "supporting" and _has(slot):
                # still ok to add as supporting
                pass
            out.append(
                CharacterCard(
                    name=name,
                    slot="supporting",
                    description=desc,
                    persona=desc,
                )
            )
            break
    return out[:5]


def enrich_character_persona(
    model,
    tokenizer,
    background: str,
    card: CharacterCard,
    *,
    theme: str = "",
    language: str = "zh",
) -> CharacterCard:
    """Second-pass model: detailed persona from role + story background."""
    lang = normalize_lang(language)
    p = story_prompts(lang)
    if lang == "en":
        theme_block = f"Story theme: {theme.strip()}\n\n" if theme.strip() else ""
        user = (
            f"{theme_block}"
            f"Story background:\n{background}\n\n"
            f"Character name: {card.name}\n"
            f"Plot slot: {card.slot}\n"
            f"Plot description: {card.description}\n"
            "Generate the detailed persona JSON for this character."
        )
    else:
        theme_block = f"故事主题：{theme.strip()}\n\n" if theme.strip() else ""
        user = (
            f"{theme_block}"
            f"故事背景：\n{background}\n\n"
            f"角色名：{card.name}\n"
            f"剧情槽位：{card.slot}\n"
            f"剧情描述：{card.description}\n"
            "请生成该角色的详细人设 JSON。"
        )
    obj = _generate_json(model, tokenizer, p["persona_gen"], user, 384) or {}
    profile = {
        "personality": str(obj.get("personality") or "").strip(),
        "speech_style": str(obj.get("speech_style") or "").strip(),
        "hobbies": obj.get("hobbies")
        if isinstance(obj.get("hobbies"), list)
        else (
            [str(obj.get("hobbies"))]
            if isinstance(obj.get("hobbies"), str) and obj.get("hobbies").strip()
            else []
        ),
        "motivation": str(obj.get("motivation") or "").strip(),
        "quirks": str(obj.get("quirks") or "").strip(),
        "summary": str(obj.get("summary") or "").strip(),
    }
    card.persona_profile = profile
    card.persona = profile.get("summary") or card.persona or card.description[:80]
    return card


def generate_story_setup(
    model,
    tokenizer,
    theme: str | None = None,
    *,
    language: str = "zh",
) -> StoryState:
    lang = normalize_lang(language)
    p = story_prompts(lang)
    theme_s = (theme or "").strip()
    if lang == "en":
        user = (
            "Generate an English text-adventure opening setup. "
            "characters MUST list 4 or 5 NPCs: exactly 1 competitor, "
            "exactly 1 goal_person, and 2–3 supporting. Unique names."
        )
        if theme_s:
            user += (
                f"\n\n[REQUIRED THEME]\n{theme_s}\n"
                "Build background, cast, and player goal around this theme so it is unmistakable."
            )
        else:
            user += "\n(No theme given — invent a vivid original setting.)"
    else:
        user = (
            "请生成一个中文文字冒险开场设定。"
            "characters 必须有 4 或 5 个 NPC：恰好 1 个 competitor、"
            "恰好 1 个 goal_person，以及 2～3 个 supporting；名字互不重复。"
        )
        if theme_s:
            user += (
                f"\n\n【必须遵守的主题】\n{theme_s}\n"
                "请围绕以上主题构建 background、人物与玩家目标；"
                "设定读起来应能立刻认出这个主题。"
            )
        else:
            user += "\n（用户未指定主题，可自由发挥一个鲜明设定。）"

    last_err: Exception | None = None
    obj: dict | None = None
    background = ""
    player: dict[str, str] = {}
    cards: list[CharacterCard] = []
    for attempt in range(3):
        attempt_user = user
        if attempt > 0:
            attempt_user += (
                "\n\n[RETRY] Previous JSON had too few characters. "
                "Output 4–5 NPCs now."
                if lang == "en"
                else "\n\n【重试】上次人物不足。请这次输出 4～5 个 NPC。"
            )
        obj = _generate_json(
            model, tokenizer, p["story_gen"], attempt_user, STORY_MAX_NEW_TOKENS
        )
        if not obj:
            last_err = RuntimeError("Story generator returned no JSON")
            continue
        try:
            background, player, cards = _validate_story_obj(obj, language=lang)
            last_err = None
            break
        except ValueError as e:
            last_err = e
            print(f"[story] setup validate failed (try {attempt + 1}/3): {e}")
    if last_err is not None:
        raise last_err
    enriched: list[CharacterCard] = []
    for card in cards:
        try:
            enriched.append(
                enrich_character_persona(
                    model,
                    tokenizer,
                    background,
                    card,
                    theme=theme_s,
                    language=lang,
                )
            )
        except Exception as e:
            print(f"[story] persona gen failed for {card.name}: {e}")
            card.persona = card.persona or card.description[:80]
            enriched.append(card)
    return StoryState(
        background=background,
        player=player,
        character_dict={c.name: c for c in enriched},
        theme=theme_s,
        language=lang,
    )


def _fallback_present(state: StoryState) -> list[str]:
    names: list[str] = []
    by_slot: dict[str, list[str]] = {
        "goal_person": [],
        "competitor": [],
        "supporting": [],
    }
    for c in state.character_dict.values():
        by_slot.setdefault(c.slot, []).append(c.name)
    for slot in ("goal_person", "competitor"):
        names.extend(by_slot.get(slot, [])[:1])
    for n in by_slot.get("supporting", []):
        if len(names) >= 3:
            break
        if n not in names:
            names.append(n)
    if not names:
        names = state.cast_names()[:3]
    return names


def choose_present_cast(model, tokenizer, state: StoryState) -> list[str]:
    """Pick 2–4 cast members present this scene; store on state."""
    lang = normalize_lang(state.language)
    p = story_prompts(lang)
    cast_lines = [
        f"- {c.name} ({c.slot}): {c.description}"
        for c in state.character_dict.values()
    ]
    if lang == "en":
        theme_line = f"Theme: {state.theme}\n" if state.theme else ""
        user = (
            f"{theme_line}"
            f"Background summary:\n{state.background[:600]}\n\n"
            f"Scene {state.scene_index}/{state.max_scenes}.\n"
            f"Cast roster:\n" + "\n".join(cast_lines) + "\n\n"
            "Choose who appears in this scene."
        )
    else:
        theme_line = f"主题：{state.theme}\n" if state.theme else ""
        user = (
            f"{theme_line}"
            f"背景摘要：\n{state.background[:600]}\n\n"
            f"第 {state.scene_index}/{state.max_scenes} 幕。\n"
            f"角色名单：\n" + "\n".join(cast_lines) + "\n\n"
            "请选择本幕出场角色。"
        )
    obj = _generate_json(model, tokenizer, p["present_cast"], user, 128) or {}
    raw = obj.get("present") if isinstance(obj.get("present"), list) else []
    present: list[str] = []
    for name in raw:
        card = lookup_character(state.character_dict, str(name))
        if card and card.name not in present:
            present.append(card.name)
    if len(present) < 2:
        present = _fallback_present(state)
    state.present_characters = present[:4]
    return state.present_characters


def _theme_prefix(state: StoryState) -> str:
    if not state.theme:
        return ""
    if normalize_lang(state.language) == "en":
        return f"Story theme: {state.theme}\n\n"
    return f"故事主题：{state.theme}\n\n"


def _present_block(state: StoryState) -> str:
    names = state.present_names()
    en = normalize_lang(state.language) == "en"
    if not names:
        return (
            "Present this scene: (unspecified — describe general atmosphere)\n"
            if en
            else "本幕出场：（未指定，可写整体气氛）\n"
        )
    details = []
    for n in names:
        card = state.character_dict.get(n)
        if card:
            details.append(f"- {card.name} ({card.slot}): {card.description}")
        else:
            details.append(f"- {n}")
    header = (
        "Present cast (describe their actions/posture; no dialogue):\n"
        if en
        else "本幕出场角色（请描写他们的动作/姿态，勿写对白）：\n"
    )
    return header + "\n".join(details) + "\n"


def format_scene_log(
    log: list[dict[str, Any]], limit: int = 40, *, language: str = "zh"
) -> str:
    en = normalize_lang(language) == "en"
    p = story_prompts(language)
    lines: list[str] = []
    for entry in log[-limit:]:
        kind = entry.get("kind") or "line"
        speaker = entry.get("speaker") or "?"
        to = entry.get("to") or []
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        if kind == "narrator" or speaker == NARRATOR_KEY:
            lines.append(f"[{p['label_narrator']}] {text}")
        else:
            to_s = (", ".join(to) if en else "、".join(to)) if to else p["label_none"]
            lines.append(f"[{speaker} → {to_s}] {text}")
    return "\n".join(lines) if lines else p["label_no_dialogue"]


def prior_dialogue_block(state: StoryState, *, limit: int = 40) -> str:
    """
    Everything already spoken before the current speaker's turn.
    (Lines are appended only after generation, so scene_log is always 'before now'.)
    """
    lang = normalize_lang(state.language)
    body = format_scene_log(state.scene_log, limit, language=lang)
    if lang == "en":
        return (
            "Messages you have already heard in this scene "
            "(stay in continuity; answer what was just said when relevant):\n"
            f"{body}"
        )
    return (
        "你已经听到的本场对白/旁白（请承接上文，相关时要回应刚才的话）：\n"
        f"{body}"
    )


def compose_narrator_intro(state: StoryState) -> tuple[str, str, dict[str, Any]]:
    lang = normalize_lang(state.language)
    p = story_prompts(lang)
    system = p["narrator_intro"]
    if lang == "en":
        user = (
            f"{_theme_prefix(state)}"
            f"Background:\n{state.background}\n\n"
            f"Player: {state.player.get('name')} — {state.player.get('identity')}\n"
            f"Goal: {state.player.get('goal')}\n"
            f"Scene {state.scene_index}/{state.max_scenes}.\n"
            f"{_present_block(state)}"
        )
    else:
        user = (
            f"{_theme_prefix(state)}"
            f"背景：\n{state.background}\n\n"
            f"玩家：{state.player.get('name')} — {state.player.get('identity')}\n"
            f"目标：{state.player.get('goal')}\n"
            f"当前第 {state.scene_index}/{state.max_scenes} 幕。\n"
            f"{_present_block(state)}"
        )
    debug = {
        "kind": "narrator_intro",
        "system": system,
        "user": user,
        "background": state.background,
        "theme": state.theme,
        "player": dict(state.player),
        "present": list(state.present_names()),
        "persona": "",
        "memory": "",
        "scene_log": format_scene_log(state.scene_log, language=lang),
        "narration": state.current_narration,
    }
    return system, user, debug


def iter_narrator_intro(model, tokenizer, state: StoryState) -> Iterator[str]:
    system, user, _ = compose_narrator_intro(state)
    yield from iter_chat_text(
        model, tokenizer, system, user, NARRATOR_MAX_NEW_TOKENS
    )


def prepare_narrator_intro(
    model, tokenizer, state: StoryState
) -> tuple[dict[str, Any], Iterator[str]]:
    system, user, debug = compose_narrator_intro(state)
    return debug, iter_chat_text(
        model, tokenizer, system, user, NARRATOR_MAX_NEW_TOKENS
    )


def narrator_intro(model, tokenizer, state: StoryState) -> str:
    return "".join(iter_narrator_intro(model, tokenizer, state)).strip()


def compose_narrator_continue(state: StoryState) -> tuple[str, str, dict[str, Any]]:
    lang = normalize_lang(state.language)
    p = story_prompts(lang)
    none = p["label_none"]
    system = p["narrator_continue"]
    if lang == "en":
        user = (
            f"{_theme_prefix(state)}"
            f"Background summary:\n{state.background[:500]}\n\n"
            f"{_present_block(state)}"
            f"Current narration:\n{state.current_narration or none}\n\n"
            f"Recent scene:\n{format_scene_log(state.scene_log, language=lang)}\n"
        )
    else:
        user = (
            f"{_theme_prefix(state)}"
            f"背景摘要：\n{state.background[:500]}\n\n"
            f"{_present_block(state)}"
            f"当前旁白：\n{state.current_narration or none}\n\n"
            f"最近场景：\n{format_scene_log(state.scene_log, language=lang)}\n"
        )
    debug = {
        "kind": "narrator_continue",
        "system": system,
        "user": user,
        "background": state.background[:500],
        "theme": state.theme,
        "player": dict(state.player),
        "present": list(state.present_names()),
        "persona": "",
        "memory": "",
        "scene_log": format_scene_log(state.scene_log, language=lang),
        "narration": state.current_narration or none,
    }
    return system, user, debug


def iter_narrator_continue(model, tokenizer, state: StoryState) -> Iterator[str]:
    system, user, _ = compose_narrator_continue(state)
    yield from iter_chat_text(
        model,
        tokenizer,
        system,
        user,
        NARRATOR_MAX_NEW_TOKENS,
        temperature=0.6,
    )


def prepare_narrator_continue(
    model, tokenizer, state: StoryState
) -> tuple[dict[str, Any], Iterator[str]]:
    system, user, debug = compose_narrator_continue(state)
    return debug, iter_chat_text(
        model,
        tokenizer,
        system,
        user,
        NARRATOR_MAX_NEW_TOKENS,
        temperature=0.6,
    )


def narrator_continue(model, tokenizer, state: StoryState) -> str:
    return "".join(iter_narrator_continue(model, tokenizer, state)).strip()


def compose_narrator_bridge(state: StoryState) -> tuple[str, str, dict[str, Any]]:
    lang = normalize_lang(state.language)
    p = story_prompts(lang)
    prev = max(1, state.scene_index - 1)
    system = p["narrator_bridge"]
    if lang == "en":
        user = (
            f"{_theme_prefix(state)}"
            f"Background:\n{state.background}\n\n"
            f"Player goal: {state.player.get('goal')}\n"
            f"Scene {prev} has ended; open scene {state.scene_index}.\n"
            f"{_present_block(state)}"
            f"Previous scene log:\n{format_scene_log(state.scene_log, 30, language=lang)}"
        )
    else:
        user = (
            f"{_theme_prefix(state)}"
            f"背景：\n{state.background}\n\n"
            f"玩家目标：{state.player.get('goal')}\n"
            f"第 {prev} 幕已结束，现在拉开第 {state.scene_index} 幕。\n"
            f"{_present_block(state)}"
            f"上一幕记录：\n{format_scene_log(state.scene_log, 30, language=lang)}"
        )
    debug = {
        "kind": "narrator_bridge",
        "system": system,
        "user": user,
        "background": state.background,
        "theme": state.theme,
        "player": dict(state.player),
        "present": list(state.present_names()),
        "persona": "",
        "memory": "",
        "scene_log": format_scene_log(state.scene_log, 30, language=lang),
        "narration": state.current_narration,
    }
    return system, user, debug


def iter_narrator_bridge(model, tokenizer, state: StoryState) -> Iterator[str]:
    system, user, _ = compose_narrator_bridge(state)
    yield from iter_chat_text(
        model, tokenizer, system, user, NARRATOR_MAX_NEW_TOKENS
    )


def prepare_narrator_bridge(
    model, tokenizer, state: StoryState
) -> tuple[dict[str, Any], Iterator[str]]:
    system, user, debug = compose_narrator_bridge(state)
    return debug, iter_chat_text(
        model, tokenizer, system, user, NARRATOR_MAX_NEW_TOKENS
    )


def narrator_bridge(model, tokenizer, state: StoryState) -> str:
    return "".join(iter_narrator_bridge(model, tokenizer, state)).strip()


def clamp_addressees(
    to_list: list[str] | None,
    state: StoryState,
    *,
    allow_empty: bool = False,
) -> list[str]:
    allowed = set(state.cast_names()) | {PLAYER_KEY}
    out: list[str] = []
    for raw in to_list or []:
        name = (raw or "").strip()
        if name in {"玩家", "player", "Player"}:
            name = PLAYER_KEY
        if name in allowed and name not in out:
            out.append(name)
        else:
            card = lookup_character(state.character_dict, name)
            if card and card.name not in out:
                out.append(card.name)
    if not out and not allow_empty:
        return [PLAYER_KEY]
    return out


def narrator_cue(model, tokenizer, state: StoryState) -> dict[str, Any]:
    lang = normalize_lang(state.language)
    p = story_prompts(lang)
    present = state.present_names()
    cast_lines = []
    for name in present:
        c = state.character_dict.get(name)
        if c:
            cast_lines.append(f"- {c.name} ({c.slot}): {c.description}")
    if not cast_lines:
        for c in state.character_dict.values():
            cast_lines.append(f"- {c.name} ({c.slot}): {c.description}")
    none = p["label_none"]
    if lang == "en":
        user = (
            f"{_theme_prefix(state)}"
            f"Background:\n{state.background}\n\n"
            f"Player name: {state.player.get('name')} (use Player when cueing the player)\n"
            f"Present cast (speaker must be one of these, or Player):\n"
            + "\n".join(cast_lines)
            + "\n\n"
            f"Scene {state.scene_index}/{state.max_scenes}.\n"
            f"Current narration:\n{state.current_narration or none}\n\n"
            f"Recent scene:\n{format_scene_log(state.scene_log, language=lang)}\n"
        )
        user += (
            f"\nDialogue lines this scene (excluding narrator): {state.dialogue_count_in_scene}/"
            f"{state.scene_dialogue_min}~{state.scene_dialogue_max}."
        )
        dcount = state.dialogue_count_in_scene
        if dcount < PLAYER_CUE_AFTER_NPC_LINES:
            user += (
                f"\nOpening phase (first {PLAYER_CUE_AFTER_NPC_LINES} dialogue lines): "
                "prefer present NPCs; do not pick Player."
            )
        if dcount < state.scene_dialogue_min:
            user += "\nBelow minimum dialogue; end_scene must be false; do not pick Narrator."
        elif dcount >= state.scene_dialogue_max:
            user += "\nHit dialogue max; set end_scene=true."
    else:
        user = (
            f"{_theme_prefix(state)}"
            f"背景：\n{state.background}\n\n"
            f"玩家名：{state.player.get('name')}（调度时用 Player）\n"
            f"本幕出场角色（只能从中选 speaker，或选 Player）：\n"
            + "\n".join(cast_lines)
            + "\n\n"
            f"第 {state.scene_index}/{state.max_scenes} 幕。\n"
            f"当前旁白：\n{state.current_narration or none}\n\n"
            f"最近场景：\n{format_scene_log(state.scene_log, language=lang)}\n"
        )
        dcount = state.dialogue_count_in_scene
        user += (
            f"\n本幕对白条数（不含旁白）：{dcount}/"
            f"{state.scene_dialogue_min}~{state.scene_dialogue_max}。"
        )
        if dcount < PLAYER_CUE_AFTER_NPC_LINES:
            user += (
                f"\n开场阶段（前 {PLAYER_CUE_AFTER_NPC_LINES} 句对白）：请优先点出场 NPC，不要点 Player。"
            )
        if dcount < state.scene_dialogue_min:
            user += "\n尚未达到最少对白条数，end_scene 必须为 false，且不要选 Narrator。"
        elif dcount >= state.scene_dialogue_max:
            user += "\n已达本幕对白上限，请设 end_scene=true。"

    obj = _generate_json(model, tokenizer, p["narrator_cue"], user, CUE_MAX_NEW_TOKENS) or {}
    raw_speaker = str(obj.get("speaker") or "").strip()
    resolved, card = resolve_cue_speaker(state, raw_speaker or PLAYER_KEY)

    def _pick_present() -> CharacterCard:
        for slot in ("goal_person", "competitor", "supporting"):
            for name in present:
                c = state.character_dict.get(name)
                if c and c.slot == slot:
                    return c
        for name in present:
            c = state.character_dict.get(name)
            if c:
                return c
        return next(iter(state.character_dict.values()))

    if resolved == NARRATOR_KEY and state.cast_names():
        pick = _pick_present()
        resolved, card = pick.name, pick
        raw_speaker = f"{raw_speaker or 'Narrator'}→override:{pick.name}"

    if (
        resolved not in (PLAYER_KEY, NARRATOR_KEY)
        and present
        and resolved not in present
    ):
        pick = _pick_present()
        raw_speaker = f"{raw_speaker}→absent→{pick.name}"
        resolved, card = pick.name, pick

    if resolved == PLAYER_KEY and dcount < PLAYER_CUE_AFTER_NPC_LINES and present:
        pick = _pick_present()
        resolved, card = pick.name, pick
        raw_speaker = f"{raw_speaker or 'Player'}→npc_first:{pick.name}"

    end_scene = bool(obj.get("end_scene"))
    if dcount < state.scene_dialogue_min:
        end_scene = False
    if dcount >= state.scene_dialogue_max:
        end_scene = True
    to = clamp_addressees(
        obj.get("to") if isinstance(obj.get("to"), list) else [],
        state,
        allow_empty=False,
    )
    allowed_to = set(present) | {PLAYER_KEY}
    to = [n for n in to if n in allowed_to] or [PLAYER_KEY]
    return {
        "speaker": resolved,
        "raw_speaker": raw_speaker,
        "to": to,
        "end_scene": end_scene,
        "reason": str(obj.get("reason") or "").strip(),
        "resolve": "ok" if (resolved in (PLAYER_KEY, NARRATOR_KEY) or card) else "miss",
        "dialogue_count": dcount,
        "present": list(present),
    }


def build_character_user_prompt(
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    memory_block: str,
    *,
    player_turn_mode: str | None = None,
) -> str:
    lang = normalize_lang(state.language)
    en = lang == "en"
    join = ", " if en else "、"
    to_s = join.join(to) if to else PLAYER_KEY
    none = story_prompts(lang)["label_none"]
    reaction = ""
    if player_turn_mode == "interrupt":
        reaction = (
            "\n[Reaction: player interrupt] The player suddenly cut into the scene. "
            "React in-character (surprise, annoyance, curiosity, going along, etc.). "
            "Show you were interrupted — do not treat this as a normal cued turn.\n"
            if en
            else (
                "\n【反应情境：玩家打断】玩家突然插话打断了当前场面。"
                "请按人设做出即时反应（惊讶、不悦、好奇、顺着话头等均可），"
                "体现出被打断，不要当成正常轮到你说话。\n"
            )
        )
    elif player_turn_mode == "cued":
        reaction = (
            "\n[Reaction: after cued player] The director already pointed at the player; "
            "they spoke in turn. Reply naturally as a normal conversational turn — "
            "do not act interrupted.\n"
            if en
            else (
                "\n【反应情境：轮到玩家后接话】导演已点过玩家，玩家按顺序发言。"
                "请自然接话回应，语气像正常对话轮次，不要表现得像被打断。\n"
            )
        )
    heard = prior_dialogue_block(state)
    if en:
        return (
            f"Your name: {card.name}\n"
            f"{card.acting_brief(lang)}\n"
            f"You are speaking to: {to_s}\n"
            f"{reaction}\n"
            f"{_theme_prefix(state)}"
            f"Story background summary:\n{state.background[:800]}\n\n"
            f"Current narration:\n{state.current_narration or none}\n\n"
            f"{heard}\n\n"
            f"{memory_block}\n\n"
            "You heard the lines above. Deliver this one line like a movie/TV character — "
            "spoken dialogue only, in persona, continuing from what you just heard:"
        )
    return (
        f"你的名字：{card.name}\n"
        f"{card.acting_brief(lang)}\n"
        f"你正在对以下人说话：{to_s}\n"
        f"{reaction}\n"
        f"{_theme_prefix(state)}"
        f"故事背景摘要：\n{state.background[:800]}\n\n"
        f"当前旁白：\n{state.current_narration or none}\n\n"
        f"{heard}\n\n"
        f"{memory_block}\n\n"
        "以上是你已经听到的内容。请严格按人设说这一句对白，承接上文，"
        "口吻像电影/连续剧台词，只要说出口的话："
    )


def compose_character_line(
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    recalled_hits: list[dict] | None = None,
    *,
    player_turn_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    lang = normalize_lang(state.language)
    p = story_prompts(lang)
    memory_block = format_facts_for_prompt(recalled_hits or [])
    user = build_character_user_prompt(
        state,
        card,
        to,
        memory_block,
        player_turn_mode=player_turn_mode,
    )
    who = f"You are {card.name}." if lang == "en" else f"你就是「{card.name}」。"
    system = p["character_line"] + f"\n{who}"
    none = p["label_none"]
    debug = {
        "kind": "character_line",
        "system": system,
        "user": user,
        "speaker": card.name,
        "slot": card.slot,
        "to": list(to),
        "player_turn_mode": player_turn_mode or "",
        "persona": card.acting_brief(lang),
        "persona_profile": dict(card.persona_profile or {}),
        "memory": memory_block,
        "recalled": list(recalled_hits or []),
        "background": state.background[:800],
        "theme": state.theme,
        "narration": state.current_narration or none,
        "scene_log": format_scene_log(state.scene_log, language=lang),
        "prior_dialogue": prior_dialogue_block(state),
        "present": list(state.present_names()),
    }
    return system, user, debug


def iter_character_line(
    model,
    tokenizer,
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    recalled_hits: list[dict] | None = None,
    *,
    player_turn_mode: str | None = None,
) -> Iterator[str]:
    system, user, _ = compose_character_line(
        state,
        card,
        to,
        recalled_hits=recalled_hits,
        player_turn_mode=player_turn_mode,
    )
    yield from iter_chat_text(
        model,
        tokenizer,
        system,
        user,
        CHARACTER_MAX_NEW_TOKENS,
    )


def prepare_character_line(
    model,
    tokenizer,
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    recalled_hits: list[dict] | None = None,
    *,
    player_turn_mode: str | None = None,
) -> tuple[dict[str, Any], Iterator[str]]:
    system, user, debug = compose_character_line(
        state,
        card,
        to,
        recalled_hits=recalled_hits,
        player_turn_mode=player_turn_mode,
    )
    return debug, iter_chat_text(
        model,
        tokenizer,
        system,
        user,
        CHARACTER_MAX_NEW_TOKENS,
    )


def generate_character_line(
    model,
    tokenizer,
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    recalled_hits: list[dict] | None = None,
    *,
    player_turn_mode: str | None = None,
) -> str:
    return "".join(
        iter_character_line(
            model,
            tokenizer,
            state,
            card,
            to,
            recalled_hits=recalled_hits,
            player_turn_mode=player_turn_mode,
        )
    ).strip()


def append_log(
    state: StoryState,
    *,
    speaker: str,
    text: str,
    to: list[str] | None = None,
    kind: str = "dialogue",
) -> dict[str, Any]:
    entry = {
        "speaker": speaker,
        "to": list(to or []),
        "text": (text or "").strip(),
        "kind": kind,
        "scene_index": state.scene_index,
    }
    state.scene_log.append(entry)
    state.message_count_in_scene += 1
    if kind == "dialogue":
        state.dialogue_count_in_scene += 1
    return entry
