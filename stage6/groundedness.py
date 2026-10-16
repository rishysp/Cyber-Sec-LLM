"""Stage 6d - deterministic groundedness check, no LLM judge (so no self-judging bias, runs in seconds).

For each answer sentence: what fraction of its content words appear in the retrieved context?
  supported   = >= 70% of its content words occur in the context
  The headline number is the share of answer sentences that are supported, plus the share of ANSWERS where every sentence is.

HONEST LIMITS: this is a lexical proxy, NOT Ragas faithfulness. It catches invented content (new facts, new IDs); it does not
catch a sentence that reuses context words but says something wrong. Because the model tends to paraphrase/echo its context,
scores will be high; the number shows 'the answer stays inside the evidence', not 'the answer is correct'.

  python groundedness.py
"""
import argparse, json, re, statistics
ap = argparse.ArgumentParser()
ap.add_argument("--items", default="/mnt/c/secrag_out/eval_items.jsonl")
ap.add_argument("--thr", type=float, default=0.7)
ap.add_argument("--show-worst", type=int, default=5)
a = ap.parse_args()

STOP = set("the a an and or of to in on for with by is are was were be been it its this that these those as at from can may will should "
           "such not no use used using also any all more most other than then into their there which when where what how do does".split())
W = lambda t: [w for w in re.findall(r"[a-z0-9][a-z0-9\-\._]*", t.lower()) if len(w) > 2 and w not in STOP]
rows = [json.loads(l) for l in open(a.items, encoding="utf-8")]
rows = [r for r in rows if r["type"] != "code" and r["contexts"] and r["answer"].strip()]

per_answer, sent_ok, sent_all, worst = [], 0, 0, []
for r in rows:
    ctx = set(W(" ".join(r["contexts"]) + " " + r["question"]))
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", r["answer"]) if len(W(s)) >= 4]
    if not sents: continue
    scores = []
    for s in sents:
        ws = W(s); sc = sum(w in ctx for w in ws) / len(ws); scores.append(sc)
        sent_all += 1; sent_ok += sc >= a.thr
        worst.append((sc, r["id"], s[:140]))
    per_answer.append(all(sc >= a.thr for sc in scores))
if not sent_all: raise SystemExit("no scorable answers - is eval_items.jsonl from a full run_eval.py run?")
out = {"answers_scored": len(per_answer), "sentences_scored": sent_all,
       f"sentence_supported_(>={a.thr:.0%}_words_in_context)": f"{sent_ok}/{sent_all} ({100*sent_ok/sent_all:.0f}%)",
       "answers_fully_supported": f"{sum(per_answer)}/{len(per_answer)} ({100*sum(per_answer)/len(per_answer):.0f}%)",
       "mean_sentence_support": round(statistics.mean(sc for sc, _, _ in worst), 3)}
print(json.dumps(out, indent=2))
print("\nleast-supported sentences (read these - they are where hallucination would show up):")
for sc, i, s in sorted(worst)[:a.show_worst]: print(f"  {sc:.2f}  [{i}]  {s}")
