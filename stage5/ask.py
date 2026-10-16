"""Quick CLI: python ask.py "What is T1059.001?"   (prints route, retrieved sources, token counts, timings, answer)"""
import os, sys, time
from common import get_counter
from retriever import HybridRetriever
from graph import build_graph
from common import label

count = get_counter()
g = build_graph(HybridRetriever(), count)
q = " ".join(sys.argv[1:]) or "What is MITRE ATT&CK technique T1059.001?"
t = time.perf_counter(); out = g.invoke({"question": q}); dt = time.perf_counter() - t
print("intent:", out["intent"], "| sources filter:", out["sources"])
print("retrieved (label, rerank score):", [(label(c), None if c.get("score") is None else round(c["score"], 3)) for c in out["chunks"]])
print("context tokens:", out["ctx_tokens"], "| retrieval ms:", {k: round(v) for k, v in out["timings"].items()})
print("ast_ok:", out.get("ast_ok"), "| attempts:", out["attempts"], "| completion tokens:", out["usage"]["completion_tokens"])
print(f"total {dt:.2f}s\n\n{out['answer']}\n\n[raw answer repr] {out['answer']!r}")

if os.environ.get("SHOW_CTX"):
    print("\n----- CONTEXT SENT TO MODEL -----\n" + out["context"])
