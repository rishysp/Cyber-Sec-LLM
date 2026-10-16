"""Stage 1b - one-time download of the fp16 Llama 3.1 8B Instruct weights (~16 GB, resumable).
Run inside the 'data' venv:  pip install huggingface_hub
Default repo is Unsloth's public mirror (no token needed). The Llama 3.1 Community License still applies.
To use Meta's gated repo instead: accept the license on huggingface.co, run `huggingface-cli login`,
then  BASE_REPO=meta-llama/Llama-3.1-8B-Instruct python 05_download_base_model.py
"""
import os, pathlib
from huggingface_hub import snapshot_download

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
dest = ROOT / "models" / "llama31-8b-instruct"
dest.mkdir(parents=True, exist_ok=True)

repo = os.environ.get("BASE_REPO", "unsloth/Meta-Llama-3.1-8B-Instruct")
snapshot_download(
    repo_id=repo,
    local_dir=str(dest),
    allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    ignore_patterns=["*.bin", "original/*", "*.gguf"],
)
size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e9
print(f"downloaded {repo} -> {dest}  ({size:.1f} GB)")
