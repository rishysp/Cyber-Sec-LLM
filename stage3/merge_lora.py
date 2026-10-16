"""Stage 3a - merge the LoRA adapter into the ORIGINAL bf16 base weights, one shard at a time.

Why not PeftModel.merge_and_unload()? It needs the whole 16 GB model in RAM. This streams shard by shard
(~5 GB peak RAM) and does the matrix math on the GPU, so it fits a laptop with ~11 GB of WSL memory.
    merged_W = W + scale * (B @ A),   scale = alpha / r   (or alpha / sqrt(r) for rsLoRA)
Note (interview point): QLoRA trains the adapter against the 4-bit (NF4) base but we merge into the full-precision
base. The mismatch is small and standard practice; we measure the end result after quantization anyway.

Run in the 'ft' venv:  python merge_lora.py
"""
import argparse, json, math, os, pathlib, re, shutil
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(ROOT / "models" / "llama31-8b-instruct"))
ap.add_argument("--adapter", default="/mnt/c/secrag_out/qlora/adapter")
ap.add_argument("--out", default=str(ROOT / "models" / "llama31-8b-secure-merged"))
args = ap.parse_args()
base, adapter, out = pathlib.Path(args.base), pathlib.Path(args.adapter), pathlib.Path(args.out)

cfg = json.load(open(adapter / "adapter_config.json"))
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / math.sqrt(r) if cfg.get("use_rslora") else alpha / r
print(f"LoRA r={r} alpha={alpha} rslora={bool(cfg.get('use_rslora'))} -> scale={scale}")

lora = load_file(str(adapter / "adapter_model.safetensors"))     # ~160 MB
pat = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$")
pairs = {}
for k, v in lora.items():
    m = pat.match(k)
    if not m:
        print("note: skipping adapter key", k)
        continue
    pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = v
assert pairs and all(len(p) == 2 for p in pairs.values()), "adapter keys not understood"
print(f"{len(pairs)} LoRA-targeted weight matrices (expect 224 = 32 layers x 7)")

out.mkdir(parents=True, exist_ok=True)
shards = sorted(base.glob("*.safetensors"))
applied = 0
for shard in shards:
    tensors = {}
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        meta = f.metadata() or {"format": "pt"}
        for name in f.keys():
            t = f.get_tensor(name)
            if name in pairs:
                A = pairs[name]["A"].to("cuda", torch.float32)
                B = pairs[name]["B"].to("cuda", torch.float32)
                W = t.to("cuda", torch.float32)
                W += scale * (B @ A)
                t = W.to(t.dtype).cpu()
                applied += 1
                del A, B, W
            tensors[name] = t.contiguous()
    save_file(tensors, str(out / shard.name), metadata=meta)
    del tensors
    torch.cuda.empty_cache()
    print("wrote", shard.name, flush=True)
assert applied == len(pairs), f"applied {applied} of {len(pairs)} - name mismatch"

for p in base.iterdir():                                          # config, tokenizer, index, chat template
    if p.is_file() and p.suffix != ".safetensors":
        shutil.copy2(p, out / p.name)

# ---------- verify: same tensors/shapes as base, sane sizes, merged weights differ and are finite ----------
checked = False
for shard in shards:
    a, b = out / shard.name, shard
    assert a.stat().st_size > 1e9, f"{a.name} suspiciously small ({a.stat().st_size} bytes)"
    with safe_open(str(a), "pt") as fa, safe_open(str(b), "pt") as fb:
        assert set(fa.keys()) == set(fb.keys()), f"tensor names differ in {a.name}"
        for n in fa.keys():
            assert fa.get_slice(n).get_shape() == fb.get_slice(n).get_shape(), f"shape differs: {n}"
        if not checked:
            for n in fa.keys():
                if n in pairs:
                    ta, tb = fa.get_tensor(n).float(), fb.get_tensor(n).float()
                    print(f"sample {n}: max|merged-base|={(ta - tb).abs().max():.5f} finite={bool(torch.isfinite(ta).all())}")
                    checked = True
                    break
total = sum(p.stat().st_size for p in out.glob("*.safetensors")) / 1e9
print(f"OK: merged model at {out}  ({total:.1f} GB of weights)")
