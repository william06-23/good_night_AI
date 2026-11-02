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
SCENE_DIALOGUE_MIN = 8
SCENE_DIALOGUE_MAX = 15
# Legacy alias used in a few places
SCENE_MESSAGE_CAP = SCENE_DIALOGUE_MAX

STORY_GEN_SYSTEM = (
    "你是文字冒险开场编剧。只输出 JSON，不要 markdown。"
    "用中文写一个可玩的短故事开场设定。"
    "篇幅分配大约：background 约60%，characters 合计约30%，player 约10%。"
    "characters 必须包含：恰好1个 slot=competitor，恰好1个 slot=goal_person，"
    "以及2或3个 slot=supporting。名字互不重复。"
    "competitor：与玩家目标相近或冲突的对手。"
    "goal_person：与玩家目标直接相关的人（如对象、目标人物等）。"
    "supporting：制造障碍或互动的配角。"
    "每个人物给 description（剧情中的身份/作用，一两句即可；详细性格稍后另生成）。"
    "persona 字段可先填空字符串。"
    '严格格式：{"background":"...","player":{"name":"...","identity":"...","goal":"..."},'
    '"characters":[{"name":"...","slot":"competitor|goal_person|supporting",'
    '"description":"...","persona":""}]}'
)

PERSONA_GEN_SYSTEM = (
    "你是角色人设设计师。根据故事背景与该角色的剧情身份（slot+description），"
    "用中文生成详细人设，只输出 JSON。"
    "要具体、可表演，包含性格、说话习惯、爱好、动机、与玩家的可能关系。"
    '格式：{"personality":"...","speech_style":"...","hobbies":["..."],'
    '"motivation":"...","quirks":"...","summary":"一两句总括，供扮演时置顶"}'
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
    "你是导演调度。根据故事背景、角色名单、当前幕次与最近场景记录，"
    "决定下一位发言者。"
    "本幕结构：旁白开场之后，只允许角色名单中的【精确姓名】或 Player 发言；"
    "禁止再选 Narrator（旁白只在开场/换幕时出现）。"
    "优先轮流安排不同角色与 Player，尤其是 goal_person 与 competitor。"
    "to 是发言对象列表（可多人），必须来自角色名单或 Player；角色发言时 to 通常含 Player。"
    "对白条数未满最少场次前，end_scene 必须为 false；"
    "达到建议条数后，若剧情告一段落可将 end_scene 设为 true。"
    '只输出 JSON：{"speaker":"...","to":["..."],"end_scene":false,"reason":"..."}'
)

NARRATOR_CONTINUE_SYSTEM = (
    "你是旁白。用中文补 1～2 句简短环境/气氛描写，承接当前场景，不要重复开场，"
    "不要写角色对白，不要输出 JSON。"
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
    persona: str = ""
    # Detailed profile from the persona-generator pass (stored, reused on every line).
    persona_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def acting_brief(self) -> str:
        """Text injected into the character-line prompt."""
        parts = [
            f"剧情身份：{self.description}",
            f"槽位：{self.slot}",
        ]
        if self.persona:
            parts.append(f"人设摘要：{self.persona}")
        prof = self.persona_profile or {}
        if prof.get("summary"):
            parts.append(f"总括：{prof['summary']}")
        if prof.get("personality"):
            parts.append(f"性格：{prof['personality']}")
        if prof.get("speech_style"):
            parts.append(f"说话习惯：{prof['speech_style']}")
        hobbies = prof.get("hobbies")
        if isinstance(hobbies, list) and hobbies:
            parts.append("爱好：" + "、".join(str(h) for h in hobbies))
        elif isinstance(hobbies, str) and hobbies.strip():
            parts.append(f"爱好：{hobbies}")
        if prof.get("motivation"):
            parts.append(f"动机：{prof['motivation']}")
        if prof.get("quirks"):
            parts.append(f"小习惯/特点：{prof['quirks']}")
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

    def cast_names(self) -> list[str]:
        return list(self.character_dict.keys())

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


def enrich_character_persona(
    model, tokenizer, background: str, card: CharacterCard
) -> CharacterCard:
    """Second-pass model: detailed persona from role + story background."""
    user = (
        f"故事背景：\n{background}\n\n"
        f"角色名：{card.name}\n"
        f"剧情槽位：{card.slot}\n"
        f"剧情描述：{card.description}\n"
        "请生成该角色的详细人设 JSON。"
    )
    obj = _generate_json(
        model, tokenizer, PERSONA_GEN_SYSTEM, user, 384
    ) or {}
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
    # Dedicated persona pass for each cast member (stored on the card).
    enriched: list[CharacterCard] = []
    for card in cards:
        try:
            enriched.append(
                enrich_character_persona(model, tokenizer, background, card)
            )
        except Exception as e:
            print(f"[story] persona gen failed for {card.name}: {e}")
            card.persona = card.persona or card.description[:80]
            enriched.append(card)
    state = StoryState(
        background=background,
        player=player,
        character_dict={c.name: c for c in enriched},
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


def narrator_continue(model, tokenizer, state: StoryState) -> str:
    user = (
        f"背景摘要：\n{state.background[:500]}\n\n"
        f"当前旁白：\n{state.current_narration or '（无）'}\n\n"
        f"最近场景：\n{format_scene_log(state.scene_log)}\n"
    )
    return generate_chat_text(
        model,
        tokenizer,
        NARRATOR_CONTINUE_SYSTEM,
        user,
        NARRATOR_MAX_NEW_TOKENS,
        temperature=0.6,
    )


def narrator_bridge(model, tokenizer, state: StoryState) -> str:
    # Called after scene_index has already advanced to the new scene.
    prev = max(1, state.scene_index - 1)
    user = (
        f"背景：\n{state.background}\n\n"
        f"玩家目标：{state.player.get('goal')}\n"
        f"第 {prev} 幕已结束，现在拉开第 {state.scene_index} 幕。\n"
        f"上一幕记录：\n{format_scene_log(state.scene_log, 30)}"
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
        f"第 {state.scene_index}/{state.max_scenes} 幕。\n"
        f"当前旁白：\n{state.current_narration or '（无）'}\n\n"
        f"最近场景：\n{format_scene_log(state.scene_log)}\n"
    )
    dcount = state.dialogue_count_in_scene
    user += (
        f"\n本幕对白条数（不含旁白）：{dcount}/"
        f"{state.scene_dialogue_min}~{state.scene_dialogue_max}。"
    )
    if dcount < state.scene_dialogue_min:
        user += "\n尚未达到最少对白条数，end_scene 必须为 false，且不要选 Narrator。"
    elif dcount >= state.scene_dialogue_max:
        user += "\n已达本幕对白上限，请设 end_scene=true。"

    obj = _generate_json(
        model, tokenizer, NARRATOR_CUE_SYSTEM, user, CUE_MAX_NEW_TOKENS
    ) or {}
    raw_speaker = str(obj.get("speaker") or "").strip()
    resolved, card = resolve_cue_speaker(state, raw_speaker or PLAYER_KEY)

    def _pick_cast() -> CharacterCard:
        for c in state.character_dict.values():
            if c.slot == "goal_person":
                return c
        return next(iter(state.character_dict.values()))

    # Mid-scene: never Narrator (opening narrator already happened).
    if resolved == NARRATOR_KEY and state.cast_names():
        pick = _pick_cast()
        resolved, card = pick.name, pick
        raw_speaker = f"{raw_speaker or 'Narrator'}→override:{pick.name}"

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
    return {
        "speaker": resolved,
        "raw_speaker": raw_speaker,
        "to": to,
        "end_scene": end_scene,
        "reason": str(obj.get("reason") or "").strip(),
        "resolve": "ok" if (resolved in (PLAYER_KEY, NARRATOR_KEY) or card) else "miss",
        "dialogue_count": dcount,
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
        f"{card.acting_brief()}\n"
        f"你正在对以下人说话：{to_s}\n\n"
        f"故事背景摘要：\n{state.background[:800]}\n\n"
        f"当前旁白：\n{state.current_narration or '（无）'}\n\n"
        f"最近场景：\n{format_scene_log(state.scene_log)}\n\n"
        f"{memory_block}\n\n"
        "请严格按人设说这一句对白："
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
    if kind == "dialogue":
        state.dialogue_count_in_scene += 1
    return entry
