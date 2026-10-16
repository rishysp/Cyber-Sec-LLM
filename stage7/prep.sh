#!/usr/bin/env bash
# Prepares the docker build context. Run from ~/secrag/stage7 inside WSL, with the `rag` venv ACTIVE.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p app
cp ../stage5/{common,context,graph,llm,retriever,server}.py app/

# Pin exactly the versions installed in THIS venv (bm25.pkl is a pickle, so versions matter). Uses the running python's own
# metadata, so it works even when the venv has no pip and `pip` on PATH is a different interpreter.
python - <<'PY' > requirements.txt
import importlib.metadata as md, sys
need = ["chromadb", "fastembed", "flashrank", "langgraph", "fastapi", "uvicorn", "rank-bm25", "numpy", "tokenizers", "requests"]
extra = ["langchain-core", "onnxruntime", "pydantic", "starlette", "httpx"]
missing = []
for name in need + extra:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        if name in need: missing.append(name)
if missing:
    sys.stderr.write("MISSING in this venv: " + ", ".join(missing) + "\n"); sys.exit(1)
PY
echo "python: $(python --version)   venv: ${VIRTUAL_ENV:-none}"
echo "orchestrator files:"; ls app; echo; echo "pinned requirements:"; cat requirements.txt
[ "$(wc -l < requirements.txt)" -ge 10 ] || { echo "requirements.txt looks too short - is the rag venv active?"; exit 1; }
echo "HOST_UID=$(id -u)"  > .env
echo "HOST_GID=$(id -g)" >> .env
echo; echo "wrote .env (container runs as your user so ~/secrag/index stays writable by you)"
