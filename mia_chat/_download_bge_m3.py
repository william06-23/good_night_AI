"""One-shot download of BAAI/bge-m3 next to the other local models."""
from huggingface_hub import snapshot_download

DEST = r"C:\Users\William\William's_AI_project\bge-m3"

path = snapshot_download(repo_id="BAAI/bge-m3", local_dir=DEST)
print("DONE", path)
