"""Smoke test of the RAG server (what Open WebUI will call). Run after server.py is up:  python test_api.py
Checks: /v1/models, streamed text answer (prints TTFT + total), non-streamed answer, code-remediation (AST gate), and the
Open WebUI '### Task:' bypass.
"""
import json, sys, time
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100/v1"
print("models:", [m["id"] for m in requests.get(f"{BASE}/models", timeout=10).json()["data"]])


def stream(q):
    t0, ttft, text = time.perf_counter(), None, ""
    with requests.post(f"{BASE}/chat/completions", json={"model": "secrag-rag", "stream": True,
                       "messages": [{"role": "user", "content": q}]}, stream=True, timeout=300) as r:
        for line in r.iter_lines():
            if not line or line == b"data: [DONE]": continue
            j = json.loads(line[6:])
            d = (j["choices"][0]["delta"].get("content") if j.get("choices") else None)
            if d:
                ttft = ttft or time.perf_counter() - t0
                text += d
    return text, ttft, time.perf_counter() - t0


for q in ["What is MITRE ATT&CK technique T1059.001?", "How should passwords be stored securely?"]:
    text, ttft, tot = stream(q)
    print(f"\n[stream] {q}\n  TTFT {ttft*1000:.0f} ms | total {tot:.2f}s\n  {text}")

q = "What does NIST SC-7 require?"
t = time.perf_counter()
r = requests.post(f"{BASE}/chat/completions", json={"stream": False, "messages": [{"role": "user", "content": q}]}, timeout=300).json()
print(f"\n[non-stream] {q}  ({time.perf_counter()-t:.2f}s)\n  {r['choices'][0]['message']['content']}")

code = ('Fix this code:\n```python\nimport sqlite3\n\ndef get_user(conn, name):\n'
        '    q = "SELECT * FROM users WHERE name = \'" + name + "\'"\n    return conn.execute(q).fetchall()\n```')
t = time.perf_counter()
r = requests.post(f"{BASE}/chat/completions", json={"stream": False, "messages": [{"role": "user", "content": code}]}, timeout=300).json()
print(f"\n[code fix] ({time.perf_counter()-t:.2f}s)\n  {r['choices'][0]['message']['content']}")

r = requests.post(f"{BASE}/chat/completions", json={"stream": False, "messages": [{"role": "user",
                  "content": "### Task:\nGenerate a concise 3-5 word title for this chat: SQL injection prevention"}]}, timeout=120).json()
print("\n[webui task bypass]", r["choices"][0]["message"]["content"][:120])
