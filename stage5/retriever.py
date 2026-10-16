"""Stage 5b - hybrid retriever: exact-ID pin + dense (Chroma) + BM25 -> RRF fusion -> FlashRank rerank. CPU only."""
import pickle, re, time
import numpy as np
from common import *

ID_ATTACK = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)
ID_NIST = re.compile(r"\b[A-Z]{2}-\d{1,2}(?:\(\d{1,2}\))?(?!\w)", re.I)


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion: score = sum 1/(k+rank). Needs no score calibration between BM25 and cosine."""
    s = {}
    for lst in rank_lists:
        for r, i in enumerate(lst):
            s[i] = s.get(i, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])]


class HybridRetriever:
    def __init__(self):
        import chromadb
        from chromadb.config import Settings
        from fastembed import TextEmbedding
        from flashrank import Ranker
        d = pickle.load(open(INDEX / "bm25.pkl", "rb"))
        self.bm25, self.chunks = d["bm25"], d["chunks"]
        self.idx = {c["id"]: i for i, c in enumerate(self.chunks)}
        self.src = np.array([c["source"] for c in self.chunks])
        self.by_ref, self.by_mit = {}, {}                   # 'T1059.001' / 'AC-2' -> chunk indices (ID pinning)
        for i, c in enumerate(self.chunks):
            if c["source"] in ("attack", "nist"):
                ref = c["id"].split(":")[1]
                if ref.endswith("-mit"): self.by_mit.setdefault(norm_ref(ref[:-4]), []).append(i)
                else: self.by_ref.setdefault(norm_ref(ref), []).append(i)
        client = chromadb.PersistentClient(path=str(INDEX / "chroma"), settings=Settings(anonymized_telemetry=False))
        self.col = client.get_collection("secrag")
        self.emb = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE), local_files_only=True)
        self.ranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(CACHE), max_length=RERANK_LEN)

    # -- pieces (public so the evaluation can compare dense-only vs hybrid) --
    def dense(self, q, k=20, sources=None):
        qv = list(self.emb.embed([QUERY_PREFIX + q]))[0].tolist()
        where = None
        if sources and len(sources) == 1: where = {"source": sources[0]}
        elif sources and len(sources) > 1: where = {"source": {"$in": list(sources)}}
        r = self.col.query(query_embeddings=[qv], n_results=k, where=where, include=["distances"])
        return [self.idx[i] for i in r["ids"][0]]

    def sparse(self, q, k=20, sources=None):
        s = self.bm25.get_scores(tok(q))
        if sources: s = np.where(np.isin(self.src, list(sources)), s, -1.0)
        top = np.argsort(-s)[:k]
        return [int(i) for i in top if s[i] > 0]

    def pin(self, q, sources=None):
        """If the query names a technique/control id that exists in the corpus, that record goes first."""
        out = []
        want_mit = re.search(r"mitigat|prevent|defen[cs]e|detect|counter", q, re.I)
        for m in ID_ATTACK.findall(q) + ID_NIST.findall(q):
            ref = norm_ref(m)
            extra = self.by_mit.get(ref, [])[:1] if want_mit else []
            for i in self.by_ref.get(ref, [])[:2] + extra:
                if i not in out and (not sources or self.chunks[i]["source"] in sources):
                    out.append(i)
        return out[:4]

    def search(self, q, sources=None, k=20, n=3, prefer=None):
        t0 = time.perf_counter()
        pinned = self.pin(q, sources)
        if pinned:   # the user named an exact technique/control id: serve that record, skip search + rerank entirely
            res = [dict(self.chunks[i], score=None) for i in pinned[:max(n, len(pinned))]]
            ms = (time.perf_counter() - t0) * 1e3
            return res, {"dense_ms": 0, "bm25_ms": 0, "rerank_ms": 0, "total_ms": ms, "pinned": len(pinned)}
        d = self.dense(q, k, sources); t1 = time.perf_counter()
        b = self.sparse(q, k, sources); t2 = time.perf_counter()
        fused = rrf([d, b])
        if prefer:   # stable partition: prescriptive docs (e.g. cheat sheets) are reranked before overview pages
            fused = [i for i in fused if self.chunks[i]["title"].startswith(prefer)] + \
                    [i for i in fused if not self.chunks[i]["title"].startswith(prefer)]
        cand = fused[:RERANK_CANDS]
        passages = [{"id": i, "text": self.chunks[i]["text"]} for i in cand]
        from flashrank import RerankRequest
        ranked = self.ranker.rerank(RerankRequest(query=q, passages=passages)) if passages else []
        t3 = time.perf_counter()
        order = pinned + [int(r["id"]) for r in ranked]
        res = [dict(self.chunks[i], score=None) for i in order[:n]]
        for r, ch in zip(ranked, res[len(pinned):]):
            ch["score"] = float(r["score"])
        return res, {"dense_ms": (t1-t0)*1e3, "bm25_ms": (t2-t1)*1e3, "rerank_ms": (t3-t2)*1e3,
                     "total_ms": (t3-t0)*1e3, "pinned": len(pinned)}
