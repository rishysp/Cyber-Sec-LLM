"""Stage 4 - honest serving benchmark against the local OpenAI-compatible vLLM endpoint.

Measures, per scenario (short prompt vs RAG-sized prompt):
  TTFT      = time from sending the request to the first streamed content token
  decode    = tokens/s after the first token  ((completion_tokens - 1) / (last_token_time - first_token_time))
Every request starts with a unique nonce so vLLM's automatic prefix cache cannot make TTFT look better than a
cold request really is. Output length is forced to >= --min-tokens so decode speed is measured over a fair window.

Run (needs only 'requests', so the 'data' venv is fine):  python bench_vllm.py
"""
import argparse, json, os, pathlib, random, statistics, time, uuid
import requests

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1")
ap.add_argument("--model", default="secrag-llama")
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=160)
ap.add_argument("--min-tokens", type=int, default=128)
ap.add_argument("--rep", type=float, default=1.0, help="repetition_penalty")
ap.add_argument("--long-chars", type=int, default=6000, help="approx. size of the RAG-style context (~4 chars/token)")
ap.add_argument("--out", default="/mnt/c/secrag_out/bench_vllm.json")
args = ap.parse_args()

SYSTEM = ("You are an offline cybersecurity assistant. Answer accurately and concisely using MITRE ATT&CK, "
          "OWASP and NIST SP 800-53 terminology. If you are unsure, say so.")
QUESTION = "What is MITRE ATT&CK technique T1059.001 (PowerShell)?"
chunks = [json.loads(l)["text"] for l in open(ROOT / "data" / "processed" / "chunks.jsonl", encoding="utf-8")]


def messages(kind, seed):
    system = f"[request {uuid.uuid4().hex}] " + SYSTEM                 # unique prefix -> no prefix-cache hit
    if kind == "short":
        return [{"role": "system", "content": system}, {"role": "user", "content": QUESTION}]
    rng = random.Random(seed)
    ctx, n = [], 0
    while n < args.long_chars:
        c = rng.choice(chunks)
        ctx.append(c)
        n += len(c)
    user = "Use only the context below to answer.\n\n" + "\n\n".join(ctx) + f"\n\nQuestion: {QUESTION}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def one(msgs):
    body = {"model": args.model, "messages": msgs, "stream": True, "temperature": 0,
            "max_tokens": args.max_tokens, "min_tokens": args.min_tokens, "repetition_penalty": args.rep,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = last = None
    usage = None
    with requests.post(args.url + "/chat/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                now = time.perf_counter()
                first = first or now
                last = now
    n = usage["completion_tokens"] if usage else 0
    return {"ttft_ms": (first - t0) * 1000, "tok_s": (n - 1) / (last - first) if n > 1 and last > first else None,
            "total_s": last - t0, "prompt_tokens": usage["prompt_tokens"] if usage else None, "completion_tokens": n}


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


results = {}
for kind in ("short", "long"):
    for i in range(args.warmup):
        one(messages(kind, 1000 + i))
    runs = [one(messages(kind, i)) for i in range(args.runs)]
    ttft = [r["ttft_ms"] for r in runs]
    tps = [r["tok_s"] for r in runs if r["tok_s"]]
    results[kind] = {
        "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in runs),
        "ttft_ms_p50": round(pct(ttft, 0.5)), "ttft_ms_p95": round(pct(ttft, 0.95)),
        "decode_tok_s_median": round(statistics.median(tps), 1), "decode_tok_s_min": round(min(tps), 1),
        "total_s_p50": round(pct([r["total_s"] for r in runs], 0.5), 2), "runs": runs}
    print(f"\n[{kind}] prompt~{results[kind]['prompt_tokens_median']:.0f} tokens | TTFT p50={results[kind]['ttft_ms_p50']} ms "
          f"p95={results[kind]['ttft_ms_p95']} ms | decode median={results[kind]['decode_tok_s_median']} tok/s "
          f"(min {results[kind]['decode_tok_s_min']}) | total p50={results[kind]['total_s_p50']} s")

json.dump({"config": vars(args), "results": results}, open(args.out, "w"), indent=2)
print("\nsaved", args.out)
