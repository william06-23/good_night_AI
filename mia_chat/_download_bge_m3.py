"""One-shot download of BAAI/bge-m3 into models/bge-m3 (or MIA_MODELS_DIR)."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MIA_MODELS_DIR", REPO_ROOT / "models")).expanduser()
DEST = str(MODELS_DIR / "bge-m3")

path = snapshot_download(repo_id="BAAI/bge-m3", local_dir=DEST)
print("DONE", path)
