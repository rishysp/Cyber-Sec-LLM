"""Stage 1d - quality control + leakage-safe split.
Split is by GROUP (technique family / control base / cheat sheet), not by row, so validation loss measures
generalisation to unseen techniques/controls instead of memorised near-duplicates of training rows.
"""
import collections, hashlib, json, os, pathlib, re, statistics

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
P = ROOT / "data" / "processed"
D = ROOT / "data"

SYSTEM = ("You are an offline cybersecurity assistant. Answer accurately and concisely using MITRE ATT&CK, "
          "OWASP and NIST SP 800-53 terminology. If you are unsure, say so.")

rows = [json.loads(l) for l in open(P / "pairs_raw.jsonl", encoding="utf-8")]
seen, keep, dropped = set(), [], collections.Counter()
for r in rows:
    q, a = r["instruction"].strip(), r["output"].strip()
    if not (15 <= len(q) <= 400 and 60 <= len(a) <= 3500):
        dropped["length"] += 1
        continue
    if sum(ch.isascii() for ch in a) / len(a) < 0.97:
        dropped["non_ascii"] += 1
        continue
    h = hashlib.md5(re.sub(r"\W+", " ", (q + a).lower()).encode()).hexdigest()
    if h in seen:
        dropped["duplicate"] += 1
        continue
    seen.add(h)
    keep.append(r)


def is_val(group):
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 100 < 6   # ~6% of groups, deterministic


train, val = [], []
for r in keep:
    (val if is_val(r["group"]) else train).append(r)

assert not ({r["group"] for r in train} & {r["group"] for r in val}), "group leakage!"


def to_chat(r):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": r["instruction"]},
                         {"role": "assistant", "content": r["output"]}],
            "source": r["source"], "group": r["group"], "kind": r["kind"]}


for name, data in (("train", train), ("val", val)):
    with open(D / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(to_chat(r), ensure_ascii=False) + "\n")

(D / "val_groups.txt").write_text("\n".join(sorted({r["group"] for r in val})))
lens = [len(r["instruction"]) + len(r["output"]) for r in keep]
print("dropped:", dict(dropped))
print(f"kept={len(keep)} train={len(train)} val={len(val)}")
print("by source (train):", dict(collections.Counter(r["source"] for r in train)))
print("by source (val):  ", dict(collections.Counter(r["source"] for r in val)))
print(f"chars per pair: median={statistics.median(lens):.0f} p95={sorted(lens)[int(0.95 * len(lens))]}  (~4 chars/token)")
