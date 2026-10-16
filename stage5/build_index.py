"""Stage 5a - build the hybrid index ONCE (needs internet only for the first model download).

  dense : BGE-small embeddings (fastembed/ONNX, CPU) -> ChromaDB persistent collection (cosine)
  sparse: BM25Okapi over an ID-preserving tokenizer -> pickle
  rerank: downloads the FlashRank cross-encoder into the cache so serving is fully offline

Run:  python build_index.py
"""
import json, pickle, time
import numpy as np
from common import *

def main():
    from fastembed import TextEmbedding
    from flashrank import Ranker
    import chromadb
    from chromadb.config import Settings
    from rank_bm25 import BM25Okapi

    CACHE.mkdir(parents=True, exist_ok=True); INDEX.mkdir(parents=True, exist_ok=True)
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    for c in chunks:                      # sanitize headings/text before embedding; ids stay unchanged
        c["title"], c["text"] = clean(c["title"]), clean(c["text"])
    print(f"{len(chunks)} chunks")

    t = time.time()
    emb = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE))
    vecs = np.array(list(emb.embed([c["text"] for c in chunks], batch_size=64)), dtype="float32")
    print(f"embedded {vecs.shape} in {time.time()-t:.0f}s")

    client = chromadb.PersistentClient(path=str(INDEX / "chroma"), settings=Settings(anonymized_telemetry=False))
    try: client.delete_collection("secrag")
    except Exception: pass
    col = client.create_collection("secrag", metadata={"hnsw:space": "cosine"})
    B = 2000
    for i in range(0, len(chunks), B):
        part = chunks[i:i+B]
        col.add(ids=[c["id"] for c in part], embeddings=vecs[i:i+B].tolist(),
                metadatas=[{"source": c["source"], "title": c["title"][:200]} for c in part])
    print("chroma count:", col.count())

    bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    with open(INDEX / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    Ranker(model_name=RERANK_MODEL, cache_dir=str(CACHE), max_length=384)   # triggers the one-time download
    print("reranker cached. Index built at", INDEX)

if __name__ == "__main__":
    main()
