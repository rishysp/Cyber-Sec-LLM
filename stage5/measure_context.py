"""Stage 5f - measure (not assume) what the retrieval layer buys you. Retrieval only: no LLM needed, GPU untouched.

  context reduction : naive baseline (top-10 dense chunks, full text, no routing) vs optimized (route -> hybrid ->
                      rerank top-3 -> extractive trim to a token budget), counted with the LLaMA tokenizer
  retrieval quality : hit@k on 18 hand-written queries (a SMALL smoke set; the real set is built in Stage 6)
                      dense-only vs BM25-only vs hybrid+rerank  -> shows what each component contributes
  latency           : retrieval milliseconds p50/p95 on CPU

Run:  python measure_context.py [--budget 450]
"""
import argparse, json, statistics, time
from common import *
from retriever import HybridRetriever
from graph import classify_rules, prefer_for, format_context
from context import compress

CASES = [  # (question, kind, key)  kind 'id' = chunk id prefix, 'title' = any substring in the chunk title
    ("What is MITRE ATT&CK technique T1059.001?", "id", "attack:T1059.001:"),
    ("Explain T1003 OS Credential Dumping", "id", "attack:T1003:"),
    ("Describe the Phishing technique T1566", "id", "attack:T1566:"),
    ("What is Valid Accounts T1078 and how is it abused?", "id", "attack:T1078:"),
    ("How do adversaries dump credentials from LSASS memory?", "id", "attack:T1003.001:"),
    ("Which technique covers abusing scheduled tasks for persistence?", "id", "attack:T1053.005:"),
    ("How does registry run key persistence work?", "id", "attack:T1547.001:"),
    ("What does NIST SP 800-53 AC-2 require?", "id", "nist:AC-2:"),
    ("Which control covers boundary protection, SC-7?", "id", "nist:SC-7:"),
    ("Which 800-53 control requires multi-factor authentication for users?", "id", "nist:IA-2"),
    ("Which NIST control covers audit record review and analysis?", "id", "nist:AU-6:"),
    ("Which NIST control requires flaw remediation and patching?", "id", "nist:SI-2"),
    ("Which control enforces least privilege?", "id", "nist:AC-6"),
    ("How do I prevent SQL injection?", "title", ["sql injection"]),
    ("How should passwords be stored securely?", "title", ["password storage"]),
    ("How do I defend against cross-site scripting?", "title", ["cross site scripting", "cross-site scripting", "xss"]),
    ("How do I prevent CSRF attacks?", "title", ["cross-site request forgery", "csrf"]),
    ("How should session management be implemented securely?", "title", ["session management"]),
]


def _n(s): return norm_ref(s).lower()


def hit(chunks, kind, key):
    for c in chunks:
        if kind == "id" and _n(c["id"]).startswith(_n(key)): return True
        if kind == "title" and any(k in c["title"].lower() for k in key): return True
    return False


def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=int, default=450)
    ap.add_argument("--out", default="/mnt/c/secrag_out/context_eval.json"); a = ap.parse_args()
    count = get_counter()
    R = HybridRetriever()
    R.search("warmup", n=1)
    rows = []
    for q, kind, key in CASES:
        base = [R.chunks[i] for i in R.dense(q, 10)]
        base_ctx = "\n\n".join(c["text"] for c in base)
        intent, sources = classify_rules(q)
        res, t = R.search(q, sources=sources, n=3, prefer=prefer_for(intent))
        parts = compress(res, q, a.budget, count)
        opt_ctx = format_context(res, parts)
        d3 = [R.chunks[i] for i in R.dense(q, 3, sources)]
        b3 = [R.chunks[i] for i in R.sparse(q, 3, sources)]
        rows.append({"q": q, "intent": intent, "pinned": t["pinned"] > 0, "base_tok": count(base_ctx), "opt_tok": count(opt_ctx),
                     "retrieval_ms": round(t["total_ms"]), "got": [label(c) for c in res],
                     "hit": hit(res, kind, key), "dense3": hit(d3, kind, key), "bm25_3": hit(b3, kind, key),
                     "naive10": hit(base, kind, key)})
        r = rows[-1]
        print(f"{intent:7s} {'PIN' if r['pinned'] else '   '} base={r['base_tok']:5d} opt={r['opt_tok']:4d} hit={r['hit']!s:5s} "
              f"{r['retrieval_ms']:4d}ms  {q[:56]}")
        if not r["hit"]: print("        expected", key, "| got", r["got"])

    def summ(rs):
        n = len(rs)
        if not n: return {}
        bt = statistics.mean(r["base_tok"] for r in rs); ot = statistics.mean(r["opt_tok"] for r in rs)
        lat = [r["retrieval_ms"] for r in rs]
        return {"n": n, "mean_baseline_tokens": round(bt), "mean_optimized_tokens": round(ot),
                "context_reduction_pct": round(100 * (1 - ot / bt), 1),
                "hit@3 dense-only": f"{sum(r['dense3'] for r in rs)}/{n}", "hit@3 bm25-only": f"{sum(r['bm25_3'] for r in rs)}/{n}",
                "hit@3 hybrid+rerank (+pin)": f"{sum(r['hit'] for r in rs)}/{n}", "hit@10 naive dense": f"{sum(r['naive10'] for r in rs)}/{n}",
                "retrieval_ms_p50": round(pct(lat, 50)), "retrieval_ms_p95": round(pct(lat, 95)), "retrieval_ms_max": max(lat)}
    out = {"budget": a.budget, "ALL": summ(rows), "UNPINNED_only (real search)": summ([r for r in rows if not r["pinned"]]),
           "PINNED_only (exact-id lookup)": summ([r for r in rows if r["pinned"]])}
    print("\n", json.dumps(out, indent=2))
    json.dump({"summary": out, "rows": rows}, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
