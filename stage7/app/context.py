"""Stage 5c - context budgeting. Keeps the prompt small so TTFT (prefill) and total latency stay low.

compress(): per retrieved chunk, keep the header line + the sentences that best overlap the query (lead sentence
gets a bonus because definitions come first), up to a token budget, in original order. Extractive, so nothing is
paraphrased or invented. This is where the 'context reduction' claim is measured, against a naive baseline.
"""
import re
from common import tok

_STOP = set("the a an of to in and or for is are what how does do with on by as at be it this that from can which".split())
_SENT = re.compile(r"(?<=[.!?])\s+")


def _units(body: str):
    out = []
    for line in body.split("\n"):
        line = line.strip()
        if line:
            out.extend(s for s in _SENT.split(line) if s)
    return out


def compress(chunks, query, budget_tokens, count):
    """chunks: list of dicts with 'text' (first line = header). count: fn(str)->int tokens. Returns list of str."""
    if not chunks: return []
    q = {t for t in tok(query) if t not in _STOP}
    per = max(60, budget_tokens // len(chunks))
    out = []
    for ch in chunks:
        head, _, body = ch["text"].partition("\n")
        units = _units(body)
        scored = []
        for pos, u in enumerate(units):
            ut = set(tok(u))
            s = len(q & ut) / (len(q) or 1) + (0.5 if pos == 0 else 0.0)
            scored.append((s, pos, u))
        keep, used = [], count(head)
        for s, pos, u in sorted(scored, key=lambda x: (-x[0], x[1])):
            c = count(u)
            if used + c > per and keep: continue
            keep.append((pos, u)); used += c
            if used >= per: break
        out.append(head + "\n" + " ".join(u for _, u in sorted(keep)))
    return out
