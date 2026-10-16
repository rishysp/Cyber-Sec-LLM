"""Diagnose WHY a chunk is (not) retrieved: ranks in dense / BM25 / fused lists and the reranker's scores.
Usage:  python debug_query.py "How do I prevent SQL injection in Python?" "prepared statements"
        (2nd arg = a phrase that should appear in the title/text of the chunk you expected to see)
"""
import sys
from common import *
from retriever import HybridRetriever, rrf
from graph import classify_rules, prefer_for

q = sys.argv[1]; want = (sys.argv[2] if len(sys.argv) > 2 else "").lower()
R = HybridRetriever()
intent, sources = classify_rules(q); prefer = prefer_for(intent)
d, b = R.dense(q, 100, sources), R.sparse(q, 100, sources)
fused = rrf([d, b])
if prefer:
    fused = [i for i in fused if R.chunks[i]["title"].startswith(prefer)] + [i for i in fused if not R.chunks[i]["title"].startswith(prefer)]
print(f"intent={intent} sources={sources} prefer={prefer!r} RERANK_CANDS={RERANK_CANDS}\n")
def show(name, lst, n=8):
    print(f"--- {name} top {n}")
    for r, i in enumerate(lst[:n]): print(f"  {r+1:2d}. {R.chunks[i]['id']:34s} {label(R.chunks[i])[:70]}")
show("dense", d); show("bm25", b); show("fused (after prefer-partition)", fused, RERANK_CANDS + 2)
if want:
    print(f"\n--- chunks containing {want!r} in title/text")
    for i, c in enumerate(R.chunks):
        if sources and c["source"] not in sources: continue
        if want in c["title"].lower() or want in c["text"][:300].lower():
            rd = d.index(i) + 1 if i in d else None; rb = b.index(i) + 1 if i in b else None
            rf = fused.index(i) + 1 if i in fused else None
            print(f"  {c['id']:34s} dense_rank={rd} bm25_rank={rb} fused_rank={rf} in_candidates={bool(rf and rf <= RERANK_CANDS)} | {label(c)[:60]}")
from flashrank import RerankRequest
cand = fused[:12]
rr = R.ranker.rerank(RerankRequest(query=q, passages=[{"id": i, "text": R.chunks[i]["text"]} for i in cand]))
print("\n--- reranker on top-12 fused (score, label)")
for x in rr: print(f"  {x['score']:.5f}  {R.chunks[int(x['id'])]['id']:30s} {label(R.chunks[int(x['id'])])[:60]}")
