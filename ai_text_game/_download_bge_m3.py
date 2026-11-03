"""One-shot download of BAAI/bge-m3 into models/bge-m3 (or AI_TEXT_GAME_MODELS_DIR)."""

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]


def _models_dir() -> Path:
    for key in ("AI_TEXT_GAME_MODELS_DIR", "MIA_MODELS_DIR"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser()
    return REPO_ROOT / "models"


MODELS_DIR = _models_dir()


def main() -> None:
    from huggingface_hub import snapshot_download

    dest = MODELS_DIR / "bge-m3"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BAAI/bge-m3 → {dest}")
    snapshot_download(repo_id="BAAI/bge-m3", local_dir=str(dest))
    print("Done.")


if __name__ == "__main__":
    main()
