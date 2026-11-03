# AI Text Game — agent context

Branch **`AI_text_game`**. Prefer small, focused changes. Weights under `models/` or `AI_TEXT_GAME_MODELS_DIR` (legacy: `MIA_MODELS_DIR`).

Python package: **`ai_text_game/`**. Girlfriend-chat UI/CLI are not on this branch.

## Product

Directed text adventure. One Qwen plays story generator, persona designer, narrator/director, NPC voices. bge-m3 retrieves Narrator/Player/NPC memories. Theme + **language** (`zh`|`en`) drive setup, personas, narration, and dialogue. Present cast 2–4 per scene; Narrator opens; 8–15 dialogue lines; NPC-first early cues; cued vs interrupt reactions. Max 5 scenes. UI: `web/src/StoryApp.jsx` (中文/EN toggle top-right).

## Launchers

- `run_web.bat` / `run_api.bat` / `run_story.bat`
- `local_env.bat.example` → gitignored `local_env.bat` for `PY` + `AI_TEXT_GAME_MODELS_DIR`

## Key modules

- `ai_text_game/engine.py` — `TextGameEngine` (alias `MiaEngine`)
- `ai_text_game/story.py` + `prompts.py` — bilingual story prompts
- `ai_text_game/memory.py` — multi-perspective memory; language via `set_memory_language`
- `ai_text_game/api.py` — FastAPI
- `ai_text_game/story_cli.py` — CLI

## Conventions

1. Committed bats stay portable; personal paths only in `local_env.bat`.
2. Push **`AI_text_game` only** unless asked otherwise.
3. Single root `README.md` for run instructions (no nested package README).
