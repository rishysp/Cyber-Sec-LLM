"""Stage 5 - patch the EXISTING index in place (minutes, not 13): strip kramdown junk like '{: style="..."}' from
headings/text, re-embed ONLY the chunks that changed, refresh the chroma entries and rebuild the BM25 pickle.
Run once:  python reclean_index.py   (idempotent; build_index.py now does the same cleaning for fresh builds)
"""
import json, pickle
import numpy as np
from common import *


def main():
    from fastembed import TextEmbedding
    import chromadb
    from chromadb.config import Settings
    from rank_bm25 import BM25Okapi

    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    changed = []
    for i, c in enumerate(chunks):
        t, x = clean(c["title"]), clean(c["text"])
        if t != c["title"] or x != c["text"]:
            changed.append(i)
        c["title"], c["text"] = t, x
    print(f"{len(changed)} of {len(chunks)} chunks contained junk and will be re-embedded")

    client = chromadb.PersistentClient(path=str(INDEX / "chroma"), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection("secrag")
    if changed:
        emb = TextEmbedding(EMBED_MODEL, cache_dir=str(CACHE), local_files_only=True)
        for s in range(0, len(changed), 500):
            idx = changed[s:s + 500]
            v = np.array(list(emb.embed([chunks[i]["text"] for i in idx], batch_size=64)), dtype="float32")
            col.update(ids=[chunks[i]["id"] for i in idx], embeddings=v.tolist(),
                       metadatas=[{"source": chunks[i]["source"], "title": chunks[i]["title"][:200]} for i in idx])
            print(f"  re-embedded {min(s + 500, len(changed))}/{len(changed)}")
    with open(INDEX / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": BM25Okapi([tok(c["text"]) for c in chunks]), "chunks": chunks}, f)
    print("chroma count:", col.count(), "| bm25 rebuilt with cleaned text")


if __name__ == "__main__":
    main()
