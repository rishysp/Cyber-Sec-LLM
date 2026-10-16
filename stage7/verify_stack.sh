#!/usr/bin/env bash
# Smoke test of the running stack. Usage: ./verify_stack.sh
set -u
ok(){ printf "  [ok]   %s\n" "$1"; }; bad(){ printf "  [FAIL] %s\n" "$1"; FAIL=1; }; FAIL=0
echo "containers:"; docker ps --format '  {{.Names}}  {{.Status}}' | grep secrag || bad "no secrag containers running"
echo "endpoints:"
curl -sf http://127.0.0.1:8000/health -o /dev/null && ok "vLLM /health" || bad "vLLM /health (airgap mode hides this port - expected then)"
curl -sf http://127.0.0.1:8100/health -o /dev/null && ok "orchestrator /health" || bad "orchestrator /health (airgap mode hides this port - expected then)"
curl -sf http://127.0.0.1:3000/health -o /dev/null && ok "Open WebUI /health" || bad "Open WebUI /health"
echo "model list via the orchestrator (Open WebUI shows exactly this):"
curl -sf http://127.0.0.1:8100/v1/models | python3 -c "import sys,json; print('  ', [m['id'] for m in json.load(sys.stdin)['data']])" || bad "/v1/models"
echo "one RAG question (expect an answer plus a Sources footer):"
curl -sf http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"secrag-rag","stream":false,"messages":[{"role":"user","content":"What does NIST SP 800-53 AC-2 require?"}]}' \
  | python3 -c "import sys,json; print('  ', json.load(sys.stdin)['choices'][0]['message']['content'][:500].replace('\n','\n   '))" || bad "chat completion"
echo "GPU memory:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/   /'
[ "$FAIL" = 0 ] && echo "ALL GOOD" || echo "SOME CHECKS FAILED"
