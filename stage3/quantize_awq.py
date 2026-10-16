"""Stage 3b - AWQ 4-bit (weights int4, activations 16-bit) with llm-compressor, output loadable by vLLM.

AWQ (Activation-aware Weight Quantization): before rounding weights to 4 bits it finds per-channel scales that
protect the ~1% of weight channels that matter most (judged by activation magnitude on calibration data).
Calibration data = OUR OWN domain text (training pairs rendered with the chat template), so it works offline and
matches the real traffic better than generic web text.

Run in a separate 'quant' venv (llm-compressor pins its own torch/transformers):  python quantize_awq.py
Smoke test first:  python quantize_awq.py --samples 16 --seq 256 --out ~/secrag/models/smoke-awq
"""
import argparse, json, os, pathlib, random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models" / "llama31-8b-secure-merged"))
ap.add_argument("--out", default=str(ROOT / "models" / "llama31-8b-secure-awq"))
ap.add_argument("--scheme", default="W4A16", help="W4A16 (symmetric, most compatible) or W4A16_ASYM")
ap.add_argument("--samples", type=int, default=256)
ap.add_argument("--seq", type=int, default=512)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto")      # stays on CPU; layers are onloaded one by one
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

# ---- calibration set: random training pairs, rendered with the same chat template used in training ----
rng = random.Random(1337)
rows = [json.loads(l) for l in open(ROOT / "data" / "train.jsonl", encoding="utf-8")]
rng.shuffle(rows)
texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows[: args.samples]]
ds = Dataset.from_list([{"text": t} for t in texts])
ds = ds.map(lambda s: tok(s["text"], padding=False, max_length=args.seq, truncation=True, add_special_tokens=False),
            remove_columns=ds.column_names)
print(f"calibration samples={len(ds)} max_len={args.seq} scheme={args.scheme}")

recipe = [AWQModifier(ignore=["lm_head"], scheme=args.scheme, targets=["Linear"])]
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.seq, num_calibration_samples=len(ds))

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(out), save_compressed=True)
tok.save_pretrained(str(out))
size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
print(f"saved quantized model to {out} ({size:.1f} GB)")
