"""Stage 2b - understand WHAT the fine-tune learned before we merge and quantize it.
1. Validation loss per source (attack / nist / owasp / code), tuned adapter vs the same model with the adapter disabled.
2. A few greedy sample generations, including one vulnerability type that is not in the training library.
3. A first AST syntax preview on generated Python (the full metric comes in the evaluation stage).
Run in the 'ft' venv:  python eval_by_source.py
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, ast, collections, json, pathlib, re
from unsloth import FastLanguageModel
import torch

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="/mnt/c/secrag_out/qlora/adapter")
ap.add_argument("--max-seq", type=int, default=1024)
ap.add_argument("--out", default="/mnt/c/secrag_out/per_source_eval.json")
args = ap.parse_args()

SYSTEM = ("You are an offline cybersecurity assistant. Answer accurately and concisely using MITRE ATT&CK, "
          "OWASP and NIST SP 800-53 terminology. If you are unsure, say so.")

model, tok = FastLanguageModel.from_pretrained(model_name=args.adapter, max_seq_length=args.max_seq,
                                               dtype=None, load_in_4bit=True)
model.eval()
rows = [json.loads(l) for l in open(ROOT / "data" / "val.jsonl", encoding="utf-8")]


def encode(msgs):
    prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return (tok(prefix, add_special_tokens=False)["input_ids"],
            tok(full, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def answer_nll(ids, n_prompt):
    x = torch.tensor([ids], device="cuda")
    logits = model(input_ids=x).logits[0, n_prompt - 1:-1].float()      # only answer positions
    loss = torch.nn.functional.cross_entropy(logits, x[0, n_prompt:], reduction="sum")
    n = len(ids) - n_prompt
    del logits
    torch.cuda.empty_cache()
    return loss.item(), n


def run():
    tot = collections.defaultdict(lambda: [0.0, 0])
    for r in rows:
        p, f = encode(r["messages"])
        if len(f) > args.max_seq:
            continue
        l, n = answer_nll(f, len(p))
        for k in ("ALL", r["source"]):
            tot[k][0] += l
            tot[k][1] += n
    return {k: round(v[0] / v[1], 3) for k, v in sorted(tot.items())}


tuned = run()
with model.disable_adapter():
    base = run()
print(f"\n{'source':<8}{'base':>8}{'tuned':>8}{'change':>9}")
for k in tuned:
    print(f"{k:<8}{base[k]:>8.3f}{tuned[k]:>8.3f}{(tuned[k] - base[k]) / base[k] * 100:>8.0f}%")
json.dump({"base": base, "tuned": tuned}, open(args.out, "w"), indent=2)

# ---------- sample generations ----------
FastLanguageModel.for_inference(model)


def ask(q, max_new=260):
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


held_out_code = next(r["messages"][1]["content"] for r in rows if r["source"] == "code")
open_redirect = ("Fix the following Python code so it is no longer vulnerable to open redirect (CWE-601).\n\n"
                 "```python\nfrom flask import redirect, request\n\ndef go():\n    return redirect(request.args.get(\"next\"))\n```")
tests = {
    "ATT&CK (known style)": "What is MITRE ATT&CK technique T1059.001 (PowerShell)?",
    "NIST (known style)": "What does NIST SP 800-53 control AC-2 (Account Management) require?",
    "code, held-out pattern": held_out_code,
    "code, pattern NOT in training library": open_redirect,
}
for name, q in tests.items():
    a = ask(q)
    print(f"\n===== {name} =====\n{a}")
    blocks = re.findall(r"```(?:python)?\n(.*?)```", a, re.S)
    if blocks:
        ok = []
        for b in blocks:
            try:
                ast.parse(b)
                ok.append(True)
            except SyntaxError:
                ok.append(False)
        print(f"[AST preview] code blocks={len(blocks)} parse_ok={ok}")
