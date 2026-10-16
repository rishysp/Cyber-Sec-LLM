"""Stage 5e - OpenAI-compatible RAG endpoint for Open WebUI.  GET /v1/models, POST /v1/chat/completions.

Text questions : route + retrieve + build prompt, then STREAM vLLM's tokens straight through (TTFT = retrieval + prefill).
Code questions : full graph (generate -> AST gate -> one repair attempt), returned whole (or as a single SSE chunk).
Open WebUI also sends housekeeping prompts (chat titles, tags) that start with '### Task:'; those bypass RAG.

Run:  python server.py --port 8100
"""
import argparse, json, time, uuid
import requests, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from common import *
from retriever import HybridRetriever
from graph import build_graph, strip_lead
import llm

MODEL_ID = "secrag-rag"
count = get_counter()
retr = HybridRetriever()
prep = build_graph(retr, count, prepare_only=True)
full = build_graph(retr, count, prepare_only=False)
app = FastAPI(title="secrag")


def sse(obj): return f"data: {json.dumps(obj)}\n\n"
def chunk(cid, text=None, finish=None):
    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": MODEL_ID,
            "choices": [{"index": 0, "delta": ({"content": text} if text else {}), "finish_reason": finish}]}
def footer(state):
    src = ", ".join(dict.fromkeys(label(c) for c in state["chunks"]))
    extra = ""
    if state["intent"] == "code":
        extra = f"\n\nAST check: {'passed' if state.get('ast_ok') else 'FAILED - ' + state.get('ast_error', '')} (syntax only, not a security proof)"
    return f"\n\n---\nSources: {src}  |  route: {state['intent']}{extra}"


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health(): return {"ok": True}


@app.post("/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    stream = bool(body.get("stream"))
    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    if isinstance(user, list):                                   # multi-part content -> join text parts
        user = " ".join(p.get("text", "") for p in user if isinstance(p, dict))
    cid = "chatcmpl-" + uuid.uuid4().hex[:12]
    mt = int(body.get("max_tokens") or 256)

    if "### Task:" in user:                                      # Open WebUI housekeeping -> plain LLM, no RAG
        r = llm_raw(msgs, min(mt, 128), stream)
        return r if stream else JSONResponse(r)

    t0 = time.perf_counter()
    if not stream:
        s = full.invoke({"question": user})
        text = s["answer"] + footer(s)
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": s["usage"]}

    s0 = prep.invoke({"question": user})
    if s0["intent"] == "code":
        s = full.invoke({"question": user})
        def one():
            yield sse(chunk(cid, s["answer"] + footer(s))); yield sse(chunk(cid, finish="stop")); yield "data: [DONE]\n\n"
        return StreamingResponse(one(), media_type="text/event-stream")

    def gen():
        r = llm.chat(s0["messages"], max_tokens=min(mt, 260), stream=True)
        buf, flushed, sent_footer = "", False, False      # buf: first few words are held back so a leading
        for line in r.iter_lines():                       # 'From the context [...]:' line can be removed
            if not line: continue
            line = line.decode()
            if line.strip() == "data: [DONE]":
                if buf and not flushed:
                    yield sse(chunk(cid, strip_lead(buf))); flushed = True
                if not sent_footer: yield sse(chunk(cid, footer(s0)))
                yield "data: [DONE]\n\n"; break
            j = json.loads(line[6:])
            ch = j["choices"][0] if j.get("choices") else None
            delta = ch["delta"].get("content") if ch else None
            if delta and not flushed:
                buf += delta
                low, tgt = buf.lstrip().lower(), "from the context"
                if (len(low) < len(tgt) and tgt.startswith(low)) or \
                   (low.startswith(tgt) and "\n" not in buf and len(buf) < 200):
                    continue            # hold ONLY while the answer could be the 'From the context ...:' lead-in
                flushed = True          # anything else streams immediately (keeps TTFT honest)
                text = strip_lead(buf)
                if text: yield sse(chunk(cid, text))
                continue
            if ch and ch.get("finish_reason") and not sent_footer:   # inject sources BEFORE the finish chunk
                if buf and not flushed:
                    yield sse(chunk(cid, strip_lead(buf))); flushed = True
                yield sse(chunk(cid, footer(s0))); sent_footer = True
            j["model"] = MODEL_ID
            yield sse(j)
        print(f"[{s0['intent']}] ctx={s0['ctx_tokens']}tok retrieval={s0['timings']['total_ms']:.0f}ms "
              f"total={time.perf_counter()-t0:.2f}s q={user[:60]!r}")
    return StreamingResponse(gen(), media_type="text/event-stream")


def llm_raw(msgs, max_tokens, stream):
    body = {"model": LLM_NAME, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.2, "stream": stream}
    if not stream:
        return requests.post(f"{VLLM_URL}/chat/completions", json=body, timeout=300).json()
    def gen():
        with requests.post(f"{VLLM_URL}/chat/completions", json=body, stream=True, timeout=300) as r:
            for line in r.iter_lines():
                if line: yield line.decode() + "\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
