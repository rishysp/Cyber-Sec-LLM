"""Stage 6c - Ragas faithfulness (groundedness) + context precision, fully offline: the judge is the local vLLM model.

  ~/venvs/ragas/bin/python ragas_eval.py --n 40

HONEST LIMITS (say these in the interview):
  * the judge is the SAME 8B model that wrote the answers -> self-preference bias, and an 8B judge is a noisy judge.
    Treat the numbers as a relative signal (before/after a change), not an absolute quality claim.
  * Ragas needs the judge to emit valid JSON; some samples come back NaN. We report coverage and only average the valid ones.
  * 'LLMContextPrecisionWithoutReference' judges whether each retrieved chunk was useful for answering, given the answer.
"""
import argparse, json, math, random, statistics
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--items", default="/mnt/c/secrag_out/eval_items.jsonl")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--url", default="http://localhost:8000/v1")
ap.add_argument("--out", default="/mnt/c/secrag_out/ragas_results.json")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.items, encoding="utf-8")]
rows = [r for r in rows if r["type"] != "code" and r["contexts"] and r["answer"].strip()]
rows = random.Random(1).sample(rows, min(a.n, len(rows)))

llm = LangchainLLMWrapper(ChatOpenAI(base_url=a.url, api_key="none", model="secrag-llama", temperature=0, max_tokens=768, timeout=300))
ds = EvaluationDataset(samples=[SingleTurnSample(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"]) for r in rows])
res = evaluate(ds, metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()], llm=llm,
               run_config=RunConfig(max_workers=1, timeout=300, max_retries=2), raise_exceptions=False, show_progress=True)
df = res.to_pandas()

out = {"n_samples": len(rows), "judge": "secrag-llama (self-judged, 8B AWQ)"}
for col in [c for c in df.columns if c in ("faithfulness", "llm_context_precision_without_reference")]:
    v = [x for x in df[col].tolist() if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out[col] = {"valid": f"{len(v)}/{len(rows)}", "mean": round(statistics.mean(v), 3) if v else None,
                "median": round(statistics.median(v), 3) if v else None}
print(json.dumps(out, indent=2))
df.insert(0, "id", [r["id"] for r in rows])
out["per_item"] = json.loads(df.drop(columns=[c for c in ("retrieved_contexts", "response") if c in df.columns]).to_json(orient="records"))
json.dump(out, open(a.out, "w"), indent=2)
print("saved", a.out)
