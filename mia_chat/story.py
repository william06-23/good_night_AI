"""
Directed text-game story: setup generator, narrator prose/cues, character lines.
Uses the same loaded instruct model with different prompts.
"""

from __future__ import annotations

import json
import re
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

STORY_MAX_NEW_TOKENS = 1024
NARRATOR_MAX_NEW_TOKENS = 256
CUE_MAX_NEW_TOKENS = 128
CHARACTER_MAX_NEW_TOKENS = 256
MAX_SCENES = 5
SCENE_MESSAGE_CAP = 15

STORY_GEN_SYSTEM = (
    "你是文字冒险开场编剧。只输出 JSON，不要 markdown。"
    "用中文写一个可玩的短故事开场设定。"
    "篇幅分配大约：background 约60%，characters 合计约30%，player 约10%。"
    "characters 必须包含：恰好1个 slot=competitor，恰好1个 slot=goal_person，"
    "以及2或3个 slot=supporting。名字互不重复。"
    "competitor：与玩家目标相近或冲突的对手。"
    "goal_person：与玩家目标直接相关的人（如对象、目标人物等）。"
    "supporting：制造障碍或互动的配角。"
    "每个人物给 description（剧情角色）和 persona（说话性格口吻，一两句）。"
    '严格格式：{"background":"...","player":{"name":"...","identity":"...","goal":"..."},'
    '"characters":[{"name":"...","slot":"competitor|goal_person|supporting",'
    '"description":"...","persona":"..."}]}'
)

NARRATOR_INTRO_SYSTEM = (
    "你是旁白。根据故事背景与玩家目标，用中文写3～4句简短旁白，"
    "交代开场情境并拉开第一幕。不要写角色对白，不要输出 JSON。"
)

NARRATOR_BRIDGE_SYSTEM = (
    "你是旁白。上一幕刚结束。用中文：先一两句收束上一幕，再写3～4句拉开下一幕。"
    "不要写角色对白，不要输出 JSON。"
)

NARRATOR_CUE_SYSTEM = (
    "你是导演兼旁白调度。根据故事背景、角色名单、当前幕次与最近场景记录，"
    "决定下一位发言者。"
    "speaker 只能是：某个角色的【精确姓名】、或 Player、或 Narrator。"
    "to 是发言对象姓名列表（可多人），对象必须来自角色名单或 Player；"
    "若 speaker 是 Narrator，to 可为 []。"
    "若本幕该结束，设 end_scene=true（例如剧情自然告一段落时）。"
    '只输出 JSON：{"speaker":"...","to":["..."],"end_scene":false,"reason":"..."}'
)

CHARACTER_LINE_SYSTEM = (
    "你正在扮演故事中的一个角色，用中文说一两句符合性格的对白。"
    "不要跳出角色，不要提AI。不要输出 JSON。"
)


@dataclass
class CharacterCard:
    name: str
    slot: str
    description: str
    persona: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class StoryState:
    background: str = ""
    player: dict[str, str] = field(default_factory=dict)
    character_dict: dict[str, CharacterCard] = field(default_factory=dict)
    scene_index: int = 1
    message_count_in_scene: int = 0
    scene_log: list[dict[str, Any]] = field(default_factory=list)
    current_narration: str = ""
    waiting_for_player: bool = False
    suggested_to: list[str] = field(default_factory=list)
    story_complete: bool = False
    max_scenes: int = MAX_SCENES
    scene_message_cap: int = SCENE_MESSAGE_CAP
    theme: str = ""

    def cast_names(self) -> list[str]:
        return list(self.character_dict.keys())

    def public_dict(self) -> dict[str, Any]:
        return {
            "background": self.background,
            "player": self.player,
            "characters": [c.to_dict() for c in self.character_dict.values()],
            "scene_index": self.scene_index,
            "message_count_in_scene": self.message_count_in_scene,
            "scene_log": list(self.scene_log[-80:]),
            "current_narration": self.current_narration,
            "waiting_for_player": self.waiting_for_player,
            "suggested_to": list(self.suggested_to),
            "story_complete": self.story_complete,
            "max_scenes": self.max_scenes,
            "scene_message_cap": self.scene_message_cap,
            "cast_names": self.cast_names(),
        }


def generate_chat_text(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    *,
    temperature: float = 0.7,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(model, "create_chat_completion"):
        out = model.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        return (out["choices"][0]["message"].get("content") or "").strip()

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0.01
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", (name or "").strip())


def lookup_character(
    character_dict: dict[str, CharacterCard], speaker: str
) -> CharacterCard | None:
    """Resolve speaker via exact key, then for-loop normalized match."""
    raw = (speaker or "").strip()
    if not raw:
        return None
    if raw in character_dict:
        return character_dict[raw]
    target = _normalize_name(raw).casefold()
    for name, card in character_dict.items():
        if _normalize_name(name).casefold() == target:
            return card
    return None


def _validate_story_obj(obj: dict) -> tuple[str, dict[str, str], list[CharacterCard]]:
    background = str(obj.get("background") or "").strip()
    player_raw = obj.get("player") if isinstance(obj.get("player"), dict) else {}
    player = {
        "name": str(player_raw.get("name") or "玩家").strip() or "玩家",
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
    if len(cards) < 4:
        raise ValueError(f"Need 4–5 characters, got {len(cards)}")
    # Dedupe by name keeping first
    seen: set[str] = set()
    unique: list[CharacterCard] = []
    for c in cards:
        if c.name in seen:
            continue
        seen.add(c.name)
        unique.append(c)
    return background, player, unique[:5]


def generate_story_setup(
    model, tokenizer, theme: str | None = None
) -> StoryState:
    user = "请生成一个中文文字冒险开场设定。"
    if theme and theme.strip():
        user += f"\n主题/偏好：{theme.strip()}"
    obj = _generate_json(
        model, tokenizer, STORY_GEN_SYSTEM, user, STORY_MAX_NEW_TOKENS
    )
    if not obj:
        raise RuntimeError("Story generator returned no JSON")
    background, player, cards = _validate_story_obj(obj)
    state = StoryState(
        background=background,
        player=player,
        character_dict={c.name: c for c in cards},
        theme=(theme or "").strip(),
    )
    return state


def format_scene_log(log: list[dict[str, Any]], limit: int = 20) -> str:
    lines: list[str] = []
    for entry in log[-limit:]:
        kind = entry.get("kind") or "line"
        speaker = entry.get("speaker") or "?"
        to = entry.get("to") or []
        text = (entry.get("text") or "").strip()
        if not text:
            continue
        if kind == "narrator" or speaker == NARRATOR_KEY:
            lines.append(f"【旁白】{text}")
        else:
            to_s = "、".join(to) if to else "（无）"
            lines.append(f"【{speaker} → {to_s}】{text}")
    return "\n".join(lines) if lines else "（尚无对白）"


def narrator_intro(model, tokenizer, state: StoryState) -> str:
    user = (
        f"背景：\n{state.background}\n\n"
        f"玩家：{state.player.get('name')} — {state.player.get('identity')}\n"
        f"目标：{state.player.get('goal')}\n"
        f"当前第 {state.scene_index}/{state.max_scenes} 幕。"
    )
    return generate_chat_text(
        model,
        tokenizer,
        NARRATOR_INTRO_SYSTEM,
        user,
        NARRATOR_MAX_NEW_TOKENS,
    )


def narrator_bridge(model, tokenizer, state: StoryState) -> str:
    user = (
        f"背景：\n{state.background}\n\n"
        f"玩家目标：{state.player.get('goal')}\n"
        f"刚结束的是第 {state.scene_index} 幕，即将进入第 {state.scene_index + 1} 幕。\n"
        f"本幕记录：\n{format_scene_log(state.scene_log, 30)}"
    )
    return generate_chat_text(
        model,
        tokenizer,
        NARRATOR_BRIDGE_SYSTEM,
        user,
        NARRATOR_MAX_NEW_TOKENS,
    )


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
        if name == "玩家":
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
    cast_lines = []
    for c in state.character_dict.values():
        cast_lines.append(f"- {c.name} ({c.slot}): {c.description}")
    user = (
        f"背景：\n{state.background}\n\n"
        f"玩家名：{state.player.get('name')}（调度时用 Player）\n"
        f"角色名单：\n" + "\n".join(cast_lines) + "\n\n"
        f"第 {state.scene_index}/{state.max_scenes} 幕，"
        f"本幕已有 {state.message_count_in_scene}/{state.scene_message_cap} 条记录。\n"
        f"当前旁白：\n{state.current_narration or '（无）'}\n\n"
        f"最近场景：\n{format_scene_log(state.scene_log)}\n"
    )
    if state.message_count_in_scene >= state.scene_message_cap:
        user += "\n提示：本幕条数已达软上限，可考虑 end_scene=true。"

    obj = _generate_json(
        model, tokenizer, NARRATOR_CUE_SYSTEM, user, CUE_MAX_NEW_TOKENS
    ) or {}
    speaker = str(obj.get("speaker") or "Player").strip()
    if speaker in {"玩家", "User", "user"}:
        speaker = PLAYER_KEY
    if speaker in {"旁白", "narrator"}:
        speaker = NARRATOR_KEY
    end_scene = bool(obj.get("end_scene"))
    if state.message_count_in_scene >= state.scene_message_cap:
        end_scene = True
    to = clamp_addressees(
        obj.get("to") if isinstance(obj.get("to"), list) else [],
        state,
        allow_empty=(speaker == NARRATOR_KEY),
    )
    return {
        "speaker": speaker,
        "to": to,
        "end_scene": end_scene,
        "reason": str(obj.get("reason") or "").strip(),
    }


def build_character_user_prompt(
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    memory_block: str,
) -> str:
    to_s = "、".join(to) if to else PLAYER_KEY
    return (
        f"你的名字：{card.name}\n"
        f"身份槽：{card.slot}\n"
        f"设定：{card.description}\n"
        f"性格口吻：{card.persona}\n"
        f"你正在对以下人说话：{to_s}\n\n"
        f"故事背景摘要：\n{state.background[:800]}\n\n"
        f"当前旁白：\n{state.current_narration or '（无）'}\n\n"
        f"最近场景：\n{format_scene_log(state.scene_log)}\n\n"
        f"{memory_block}\n\n"
        "请说出这一句对白："
    )


def generate_character_line(
    model,
    tokenizer,
    state: StoryState,
    card: CharacterCard,
    to: list[str],
    recalled_hits: list[dict] | None = None,
) -> str:
    memory_block = format_facts_for_prompt(recalled_hits or [])
    user = build_character_user_prompt(state, card, to, memory_block)
    system = CHARACTER_LINE_SYSTEM + f"\n你就是「{card.name}」。"
    return generate_chat_text(
        model,
        tokenizer,
        system,
        user,
        CHARACTER_MAX_NEW_TOKENS,
    )


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
    return entry
