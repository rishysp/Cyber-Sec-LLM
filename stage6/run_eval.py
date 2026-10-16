"""Stage 6b - run the evaluation through the REAL pipeline (router -> hybrid retrieval -> compression -> vLLM -> AST gate).
vLLM must be running. Deterministic metrics only (no LLM judge): a judge-based pass is ragas_eval.py.

  python run_eval.py                      # everything
  python run_eval.py --limit 20           # quick check
  python run_eval.py --server http://localhost:8100/v1 --ttft-n 30    # also measure TTFT through the HTTP server
"""
import argparse, collections, json, os, pathlib, random, re, statistics, sys, time
import requests

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
sys.path.insert(0, str(ROOT / "stage5")); sys.path.insert(0, str(HERE))
from common import get_counter, norm_ref, label
import checks

ap = argparse.ArgumentParser()
ap.add_argument("--eval-set", default=str(ROOT / "data" / "processed" / "eval_set.jsonl"))
ap.add_argument("--code-pairs", default=str(ROOT / "data" / "processed" / "pairs_code.jsonl"))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--code-per-pattern", type=int, default=2)
ap.add_argument("--no-code", action="store_true")
ap.add_argument("--server", default="")
ap.add_argument("--ttft-n", type=int, default=30)
ap.add_argument("--out", default="/mnt/c/secrag_out/eval_results.json")
ap.add_argument("--items-out", default="/mnt/c/secrag_out/eval_items.jsonl")
args = ap.parse_args()

ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b|\b[A-Z]{2}-\d{1,2}(?:\(\d{1,2}\))?(?!\w)")
nid = lambda s: norm_ref(s).lower()


def hit(chunk, gold):
    keys = gold["key"] if isinstance(gold["key"], list) else [gold["key"]]
    if gold["kind"] == "id":
        return any(nid(chunk["id"]).startswith(nid(k)) for k in keys)
    return any(k in chunk["title"].lower() for k in keys)


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 2) if xs else None


def rate(xs): return f"{sum(xs)}/{len(xs)} ({100*sum(xs)/len(xs):.0f}%)" if xs else "n/a"


def main():
    from retriever import HybridRetriever
    from graph import build_graph
    count = get_counter()
    g = build_graph(HybridRetriever(), count)
    g.invoke({"question": "warmup: what is T1059.001?"})

    items = [json.loads(l) for l in open(args.eval_set, encoding="utf-8")]
    all_items = items
    if args.limit: items = random.Random(3).sample(items, min(args.limit, len(items)))
    res, dump = [], []
    for k, it in enumerate(items, 1):
        t = time.perf_counter(); s = g.invoke({"question": it["question"]}); dt = time.perf_counter() - t
        ans = s["answer"]
        rank = next((i + 1 for i, c in enumerate(s["chunks"]) if hit(c, it["gold"])), None)
        ids = {nid(x) for x in ID_RE.findall(ans)}
        ctx_ids = {nid(x) for x in ID_RE.findall(s["context"] + " " + it["question"])}
        r = {"id": it["id"], "type": it["type"], "src": it["src"], "intent": s["intent"], "rank": rank,
             "pinned": s["timings"].get("pinned", 0) > 0, "retrieval_ms": s["timings"]["total_ms"], "e2e_s": dt,
             "ctx_tokens": s["ctx_tokens"], "completion_tokens": s["usage"]["completion_tokens"],
             "terms_ok": (any(t_ in ans.lower() for t_ in it["any_terms"]) if it["any_terms"] else None),
             "n_ids": len(ids), "ids_grounded": (ids <= ctx_ids) if ids else None}
        res.append(r)
        dump.append({"id": it["id"], "type": it["type"], "question": it["question"], "answer": ans,
                     "contexts": s["ctx_list"], "retrieved": [label(c) for c in s["chunks"]]})
        print(f"[{k:3d}/{len(items)}] {it['type']:6s} rank={rank} {dt:4.1f}s  {it['question'][:60]}", flush=True)

    code_res = []
    if not args.no_code and os.path.exists(args.code_pairs):
        rows = [json.loads(l) for l in open(args.code_pairs, encoding="utf-8")]
        by = collections.defaultdict(list)
        for r_ in rows:
            if r_.get("source") == "code": by[r_["group"].split("/")[1]].append(r_)  # all 17 patterns; 14 were in training (seen), 3 held out
        rng = random.Random(5)
        for pid, rs in sorted(by.items()):
            for r_ in rng.sample(rs, min(args.code_per_pattern, len(rs))):
                t = time.perf_counter(); s = g.invoke({"question": r_["instruction"]}); dt = time.perf_counter() - t
                code = checks.last_code(s["answer"])
                c = {"pid": pid, "held_out": pid in checks.HELD_OUT, "intent_code": s["intent"] == "code", "attempts": s["attempts"],
                     "ast_final": bool(s.get("ast_ok")), "ast_first_try": bool(s.get("ast_ok")) and s["attempts"] == 1,
                     "pattern_ok": checks.pattern_ok(pid, code), "behavior_ok": (checks.sqli_behavior_ok(code) if pid == "sqli" else None),
                     "e2e_s": dt, "retrieved": [label(x) for x in s["chunks"]]}
                code_res.append(c)
                dump.append({"id": f"code-{pid}", "type": "code", "question": r_["instruction"], "answer": s["answer"],
                             "contexts": s["ctx_list"], "retrieved": c["retrieved"]})
                print(f"[code {pid:7s}] ast={c['ast_final']} pattern={c['pattern_ok']} behavior={c['behavior_ok']} {dt:4.1f}s", flush=True)

    # ---------------- summary ----------------
    S = {"n_text": len(res), "n_code": len(code_res)}
    types = ["attack", "nist", "owasp"]
    S["router_accuracy"] = rate([r["intent"] == r["type"] for r in res])
    S["router_accuracy_by_type"] = {t: rate([r["intent"] == t for r in res if r["type"] == t]) for t in types}
    S["retrieval_hit@3"] = {t: rate([r["rank"] is not None for r in res if r["type"] == t]) for t in types}
    S["retrieval_hit@3"]["ALL"] = rate([r["rank"] is not None for r in res])
    S["retrieval_hit@3_unpinned_only"] = rate([r["rank"] is not None for r in res if not r["pinned"]])
    S["retrieval_precision@1"] = rate([r["rank"] == 1 for r in res])
    S["MRR"] = round(statistics.mean((1 / r["rank"]) if r["rank"] else 0 for r in res), 3) if res else None
    S["answer_keyword_hit"] = rate([r["terms_ok"] for r in res if r["terms_ok"] is not None])
    S["id_grounding (answers citing ids: all ids present in context)"] = rate([r["ids_grounded"] for r in res if r["ids_grounded"] is not None])
    S["mean_context_tokens"] = round(statistics.mean(r["ctx_tokens"] for r in res)) if res else None
    S["latency_text_e2e_s"] = {"p50": pct([r["e2e_s"] for r in res], 50), "p95": pct([r["e2e_s"] for r in res], 95),
                               "share_under_2.5s": rate([r["e2e_s"] < 2.5 for r in res])}
    S["mean_completion_tokens_text"] = round(statistics.mean(r["completion_tokens"] for r in res), 1) if res else None
    if code_res:
        S["code"] = {"router_sends_to_code_path": rate([c["intent_code"] for c in code_res]),
                     "ast_pass_first_try": rate([c["ast_first_try"] for c in code_res]),
                     "ast_pass_after_repair": rate([c["ast_final"] for c in code_res]),
                     "vulnerable_pattern_gone_AND_safe_construct_present": rate([c["pattern_ok"] for c in code_res]),
                     "  ...SEEN-in-training patterns (optimistic)": rate([c["pattern_ok"] for c in code_res if not c["held_out"]]),
                     "  ...HELD-OUT patterns (yaml/tar/cookie)": rate([c["pattern_ok"] for c in code_res if c["held_out"]]),
                     "sqli_behavioral_test (payload blocked, normal query works)": rate([c["behavior_ok"] for c in code_res if c["pid"] == "sqli" and c["behavior_ok"] is not None]),
                     "sqli_not_runnable_by_harness (signature changed)": sum(1 for c in code_res if c["pid"] == "sqli" and c["behavior_ok"] is None),
                     "latency_code_e2e_s": {"p50": pct([c["e2e_s"] for c in code_res], 50), "p95": pct([c["e2e_s"] for c in code_res], 95)}}
        S["code"]["pattern_ok_by_pid"] = {p: rate([c["pattern_ok"] for c in code_res if c["pid"] == p]) for p in sorted({c["pid"] for c in code_res})}

    if args.server:
        tt = []
        for it in random.Random(7).sample(all_items, min(args.ttft_n, len(all_items))):
            t0, first, tot = time.perf_counter(), None, None
            with requests.post(f"{args.server}/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": it["question"]}]},
                               stream=True, timeout=300) as r_:
                for line in r_.iter_lines():
                    if not line or line == b"data: [DONE]": continue
                    j = json.loads(line[6:])
                    if j.get("choices") and j["choices"][0]["delta"].get("content") and first is None:
                        first = time.perf_counter() - t0
            tt.append((first or 0, time.perf_counter() - t0))
        S["server_stream"] = {"n": len(tt), "TTFT_ms_p50": round(pct([a for a, _ in tt], 50) * 1000), "TTFT_ms_p95": round(pct([a for a, _ in tt], 95) * 1000),
                              "total_s_p50": pct([b for _, b in tt], 50), "total_s_p95": pct([b for _, b in tt], 95)}

    print("\n" + json.dumps(S, indent=2))
    json.dump({"summary": S, "text": res, "code": code_res}, open(args.out, "w"), indent=2)
    with open(args.items_out, "w", encoding="utf-8") as f:
        for d in dump: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("saved", args.out, "and", args.items_out)

if __name__ == "__main__":
    main()
