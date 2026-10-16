# Offline Security Intelligence: LLM Fine-Tuning and Hybrid RAG

A security question-answering assistant that runs **entirely offline at runtime** on a single 8 GB laptop GPU. A fine-tuned LLaMA 3.1 8B answers questions about MITRE ATT&CK, NIST controls and OWASP cheat sheets. Answers are grounded in a local hybrid search index, cite their sources, and code answers are parse-checked before they are returned.

> **Status:** working end to end and measured. Several performance targets were missed; they are reported honestly in [Results](#results) and [Known limitations](#known-limitations).

---

## Contents

1. [Highlights](#highlights)
2. [Architecture](#architecture)
3. [Results](#results)
4. [Repository layout](#repository-layout)
5. [Requirements](#requirements)
6. [Quick start](#quick-start)
7. [Daily operations](#daily-operations)
8. [How a question is answered](#how-a-question-is-answered)
9. [Pipeline stages and how to reproduce them](#pipeline-stages-and-how-to-reproduce-them)
10. [Evaluation](#evaluation)
11. [Design decisions](#design-decisions)
12. [Configuration reference](#configuration-reference)
13. [Troubleshooting](#troubleshooting)
14. [Known limitations](#known-limitations)
15. [Roadmap](#roadmap)
16. [Licensing and data sources](#licensing-and-data-sources)

---

## Highlights

- **Fully local.** Model, search index, API and chat UI all run on one machine. All services bind to `127.0.0.1`.
- **Fits an 8 GB GPU.** LLaMA 3.1 8B fine-tuned with 4-bit QLoRA (peak 6.62 GB), merged, then quantized to AWQ W4A16 (5.7 GB) and served with vLLM.
- **Hybrid retrieval.** Dense embeddings (BGE-small, ChromaDB) plus BM25, fused with reciprocal rank fusion, reranked with FlashRank. Exact IDs such as `T1059.001` or `AC-2` skip search and are pinned directly.
- **Grounded and cited.** Answers use a compressed context (about 81% smaller than a top-10 dense baseline) and end with the source records.
- **Code is checked.** Generated code is parsed with Python's AST; a failure triggers one repair attempt.
- **Honest evaluation.** A 76-question evaluation set, held-out code patterns, a deterministic groundedness proxy, and a written list of missed targets.
- **One command to run.** Docker Compose starts vLLM, the orchestrator and Open WebUI.

---

## Architecture

```
                     127.0.0.1 only, no internet needed at runtime

  Browser ──► Open WebUI :3000 ──► Orchestrator :8100 (CPU) ──► vLLM :8000 (GPU)
                  ▲                    │   ▲                       AWQ 4-bit
                  │                    │   │                       LLaMA 3.1 8B
                  └── answer + sources ┘   └── Local index
                                               ChromaDB + BM25
                                               6,138 chunks
```

Inside the orchestrator, every question passes through four stages:

```
 question ─► Router ─► Hybrid retrieval ─► Context compressor ─► vLLM ─► Answer gate ─► response
            (rules)   (BM25 + dense,       (450-token budget)           (AST check for code,
                       RRF, rerank;                                      echo strip, Sources)
                       ID pinning)
```

| Service | Port | Runs on | Role |
| --- | --- | --- | --- |
| `vllm` | 8000 | GPU | Serves the AWQ model as `secrag-llama` |
| `orchestrator` | 8100 | CPU | Router, retrieval, compression, LangGraph pipeline; exposes the OpenAI-compatible model `secrag-rag` |
| `open-webui` | 3000 | CPU | Chat UI; sees one model, `secrag-rag` |

---

## Results

Measured on an RTX 3080 Laptop GPU (8 GB), Windows 11 with WSL2. Evaluation set: 76 questions (30 hand-written, 46 generated).

### Retrieval

| Metric | Result |
| --- | --- |
| Hit@3 | 97% (74 of 76; 66 of 68 without ID pinning) |
| Precision@1 | 88% |
| MRR | 0.923 |
| Mean context size | 387 tokens |
| Context reduction vs top-10 dense baseline | about 81% |

### Speed

| Metric | Result |
| --- | --- |
| Decode speed | 63.5 tok/s (short prompt), 60.3 tok/s (long prompt) |
| End-to-end latency, text answers | p50 1.63 s, p95 2.77 s (93% under 2.5 s) |
| End-to-end latency, code answers | p50 2.25 s, p95 3.08 s |
| Time to first token, vLLM alone | 77 ms (p50, short prompt) |
| Time to first token, through the full pipeline | p50 690 ms, p95 1150 ms |
| VRAM while serving | 7,453 MiB |

### Answer quality

| Metric | Result |
| --- | --- |
| Expected-keyword hit | 82% (51 of 62) |
| Source IDs named correctly | 31 of 31 |
| Router direct hits | 48 of 76 (63%); misses fall back to the `general` route and still search everything |
| Sentences supported by context (lexical proxy) | 75% (142 of 189) |
| Answers with every sentence supported | 63% (48 of 76) |

### Code answers

| Metric | Result |
| --- | --- |
| Parses on first try | 65% |
| Parses after one repair | 91% (31 of 34) |
| Security pattern check passes | 91% overall; 27 of 28 on seen patterns, 4 of 6 on held-out patterns |
| SQL injection behavioral test (in-memory SQLite) | 2 of 2 |

### Training

| Item | Value |
| --- | --- |
| Data | 2,101 training pairs, 160 validation pairs |
| Validation loss | 2.229 to 1.417 over 2 epochs |
| Peak VRAM | 6.62 GB |
| Training time | 65.5 minutes |

### Targets that were missed

| Target | Measured |
| --- | --- |
| Groundedness above 90% | 75% of sentences (proxy) |
| First token under 250 ms | Met by vLLM alone (77 ms); not met by the full pipeline (690 ms) |
| p95 latency under 2.5 s | 2.77 s for text answers |
| VRAM under 6.5 GB | 7.4 GB while serving |

Notes on measurement:

- The groundedness number comes from a **lexical proxy** (fraction of an answer's content words found in the retrieved context, per sentence). It is **not** Ragas faithfulness. A Ragas run with a self-judge produced only 5 of 40 valid faithfulness scores and 19 of 40 valid context-precision scores, so it is not reported.
- Code-pattern results on "seen" patterns are optimistic: those patterns appear in training data. The held-out patterns (YAML loading, tar extraction, cookie flags) are the fairer test.

---

## Repository layout

```
secrag/
├── scripts/            Data preparation helpers
├── stage2/             QLoRA fine-tuning (Unsloth)
├── stage3/             Adapter merge and AWQ quantization
├── stage4/             vLLM serving config and benchmarks
├── stage5/             Orchestrator: retrieval, router, LangGraph, API server
│   ├── common.py       Shared helpers, ID normalization
│   ├── context.py      Extractive context compression
│   ├── graph.py        LangGraph pipeline: generate, AST check, one repair, echo strip
│   ├── llm.py          vLLM client
│   ├── retriever.py    BM25 + dense fusion, ID pinning, FlashRank rerank
│   └── server.py       OpenAI-compatible API (port 8100)
├── stage6/             Evaluation
│   ├── build_eval_set.py
│   ├── run_eval.py
│   ├── checks.py       17 code pattern rules + SQL injection behavior test
│   ├── groundedness.py Lexical groundedness proxy
│   └── ragas_eval.py   Ragas attempt (not reportable, see Results)
├── stage7/             Docker stack
│   ├── docker-compose.yml
│   ├── docker-compose.airgap.yml   Optional internal-only network (untested)
│   ├── Dockerfile.orchestrator
│   ├── prep.sh         Copies app code, pins package versions, writes .env
│   └── verify_stack.sh Smoke test
├── index/              ChromaDB + BM25 data (built locally, not committed)
└── models/             Merged, AWQ-quantized model (not committed)
```

Large artifacts (model weights, the index, caches) are **not** part of the repository. Rebuild them with the stage instructions below.

---

## Requirements

| Item | Used here |
| --- | --- |
| GPU | NVIDIA RTX 3080 Laptop, 8 GB VRAM (any CUDA GPU with at least 8 GB should work) |
| OS | Windows 11 with WSL2 Ubuntu |
| Docker | Docker Desktop with the WSL2 backend and GPU support |
| Python (outside Docker) | 3.12 in a virtual environment (3.14 lacks wheels for some packages) |
| Disk | Roughly 20 GB free for model, index, images and checkpoints |
| Network | Needed **only** to download models, datasets and images the first time |

The author's setup keeps WSL and Docker data on a virtual disk (V:) stored on an external drive. Before starting WSL or Docker, attach it from an Administrator PowerShell:

```powershell
diskpart /s D:\attach-wsl-disk.txt
```

This is specific to that setup; skip it if your WSL lives on an internal drive.

---

## Quick start

These steps assume the model is at `~/secrag/models` and the index is built at `~/secrag/index`. See [Pipeline stages](#pipeline-stages-and-how-to-reproduce-them) if you need to produce them.

```bash
cd ~/secrag/stage7

# 1. Copy app code into the build folder and pin package versions
./prep.sh

# 2. Build the orchestrator image
docker compose build orchestrator

# 3. Start everything (vLLM takes 1-3 minutes to load the model)
docker compose up -d
docker compose ps          # wait until all three show "healthy"

# 4. Smoke test
./verify_stack.sh

# 5. Open the chat UI
#    http://localhost:3000   then choose the model "secrag-rag"
```

Try these questions first:

| Type | Example |
| --- | --- |
| Exact ID | What is MITRE ATT&CK technique T1059.001? |
| Control ID | Explain NIST control AC-2. |
| Concept | How do I prevent SQL injection according to the OWASP cheat sheets? |
| Code | Write a Python function that safely runs a SQL query with user input. |
| General | What is the difference between authentication and authorization? |

### Query the API directly

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"secrag-rag","messages":[{"role":"user","content":"What is T1059.001?"}]}'
```

---

## Daily operations

Run these in `~/secrag/stage7`.

| Goal | Command |
| --- | --- |
| Start | `docker compose up -d` |
| Check status | `docker compose ps` |
| Follow logs | `docker compose logs -f --tail 50 vllm` (or `orchestrator`) |
| UI logs | `docker logs secrag-webui --tail 50` |
| Pause (keeps containers) | `docker compose stop` then `docker compose start` |
| Shut down | `docker compose down` |
| Shut down and delete chat history | `docker compose down -v` |
| Check GPU | `nvidia-smi` |
| Start with internal-only networking (untested) | `docker compose -f docker-compose.yml -f docker-compose.airgap.yml up -d` |

Before powering off the laptop: `docker compose down`, then run `wsl --shutdown` in PowerShell.

After changing orchestrator code:

```bash
./prep.sh && docker compose build orchestrator && docker compose up -d
```

### Proving it works offline

1. Start the stack and wait for `healthy`.
2. Turn Wi-Fi off and unplug Ethernet.
3. Ask a question in Open WebUI. A sourced answer with no error shows the full path works without internet.

---

## How a question is answered

1. **Open WebUI** sends the message to the orchestrator as an OpenAI-style chat request. Title, tag, autocomplete and follow-up generation are disabled so the UI makes no extra model calls.
2. **Router.** Keyword rules pick one of `attack`, `nist`, `owasp`, `code` or `general`. Ties and misses go to `general`. Rules are used deliberately: they are instant, testable and need no model call.
3. **Retrieval.** BM25 and dense search run on the same index. BM25 uses a tokenizer that keeps IDs (`T1059.001`, `AC-2`) intact. Results are merged with reciprocal rank fusion (k = 60).
4. **ID pinning.** If the question names an exact technique or control ID, that record is returned directly, skipping search and rerank. For `owasp` and `code` routes, OWASP Cheat Sheet sources are preferred. Code questions also get a vulnerability hint (for example SQL injection) to improve retrieval.
5. **Rerank and compress.** FlashRank (`ms-marco-MiniLM-L-12-v2`) reorders 8 candidates. The compressor then keeps question-matching sentences within a 450-token budget.
6. **Generation.** One prompt goes to vLLM, which serves the AWQ model at about 60 tokens per second.
7. **Answer gate.** For code, the output is parsed with Python's AST; one repair attempt runs on failure. Echoed prompt instructions are removed, and a `Sources:` line is added from the retrieved records.
8. **Response** streams back to the UI. Streaming uses a prepare-only graph so the first token arrives sooner.

---

## Pipeline stages and how to reproduce them

| Stage | What it does | Key settings |
| --- | --- | --- |
| 1. Data | Builds instruction pairs from ATT&CK, NIST and OWASP. Group-level split puts about 6% of text groups into validation by hash. Code patterns yaml, tar and cookie are held out. 269 code pairs over 17 patterns. | Tokenizer needs `no_truncation()` when measuring length |
| 2. Fine-tune | Unsloth 4-bit QLoRA on LLaMA 3.1 8B | r = 16, alpha = 16, all 7 projection matrices, lr 2e-4 cosine, warmup 0.05, adamw_8bit, effective batch 16, 2 epochs, max sequence 1024, gradient checkpointing |
| 3. Merge and quantize | Shard-by-shard merge of the adapter into bf16 base weights, then AWQ W4A16 with llm-compressor | 256 calibration samples, sequence 512, `lm_head` ignored; output 5.7 GB |
| 4. Serve | vLLM v0.29.0 in Docker | See [Configuration](#configuration-reference) |
| 5. Orchestrate | Index build, router, retrieval, LangGraph, API server | 6,138 chunks: ATT&CK 1,729, NIST 1,303, OWASP 3,106 |
| 6. Evaluate | Retrieval, latency, router, answer and code checks | See [Evaluation](#evaluation) |
| 7. Deploy | Docker Compose with health checks and Open WebUI | Offline mode on |

Practical notes learned while building:

- Write heavy files (checkpoints, swap, build output) to the internal drive, not an external one. An earlier run on an external drive stalled and left an empty checkpoint.
- Use `nohup` for long builds so a closed terminal does not kill them.
- Check that `pip` is the virtual environment's pip; use `python -m pip`.
- An 8,000-character filter had silently dropped long OWASP sections. Verify chunk counts per source after every index build.
- NIST IDs are zero-padded; normalize IDs before matching.

Each stage's scripts live in its `stageN/` folder with the settings above as defaults. Check the actual arguments in each script before running.

---

## Evaluation

Run from the project root with the stack up (the evaluation calls vLLM):

```bash
cd ~/secrag
~/venvs/rag/bin/python stage6/build_eval_set.py      # writes data/processed/eval_set.jsonl
~/venvs/rag/bin/python stage6/run_eval.py            # retrieval, latency, router, code checks
~/venvs/rag/bin/python stage6/groundedness.py --show-worst
```

Outputs go to `/mnt/c/secrag_out/` (`eval_results.json`, `eval_items.jsonl`).

| Component | What it measures |
| --- | --- |
| `build_eval_set.py` | 76 items: 30 hand-written, 46 generated |
| `run_eval.py` | Hit@3, precision@1, MRR, context size, latency percentiles, time to first token (sampled over all items), router accuracy, keyword hit, source-ID grounding, code checks |
| `checks.py` | 17 regex security-pattern rules, validated against the 269 training code pairs, plus a behavioral SQL injection test on in-memory SQLite |
| `groundedness.py` | Per-sentence support: share of content words found in the context (threshold 0.7). Prints the least-supported sentences |

Ragas (v0.3.9) was tried with the local model as judge and abandoned because most scores were invalid, so groundedness is reported from the deterministic proxy instead.

---

## Design decisions

| Decision | Reason |
| --- | --- |
| AWQ W4A16 with vLLM's Marlin kernel | Best quality-per-byte for an 8 GB card; fast decode |
| Decode-only CUDA graphs, no `torch.compile` | The default compile setup left a 0.09 GiB KV cache (3072 tokens need 0.38 GiB). Eager mode gave about 36 tok/s; decode-only graphs give 60 to 66 |
| Hybrid search instead of dense only | Dense finds meaning, BM25 finds exact IDs and rare terms; either alone missed cases |
| ID pinning | Questions naming a technique or control should return that record, not a similar one |
| Rule-based router | Instant, deterministic, easy to test; misses degrade to a search over everything |
| Small compressed context | KV cache on 8 GB is scarce; each prompt token costs speed and memory |
| Fine-tune plus retrieval | Fine-tune supplies vocabulary and style; retrieval supplies facts that can be cited and updated without retraining |
| Deterministic groundedness proxy | A local judge model was unreliable; a transparent metric can be explained and reproduced |
| Services bound to `127.0.0.1` | Nothing outside the machine can reach them |

---

## Configuration reference

### vLLM

```
-cc.mode=none
-cc.cudagraph_mode=full_decode_only
--max-model-len=3072
--gpu-memory-utilization=0.85
--max-num-seqs=2
--max-num-batched-tokens=2048
--served-model-name=secrag-llama
--dtype=float16
```

Image: `vllm/vllm-openai:v0.29.0-cu129`. A GPU throttle reason of `0x4` is the laptop's roughly 90 W software power cap and is normal.

### Orchestrator

| Variable | Value |
| --- | --- |
| `SEC_ROOT` | `/srv/secrag` |
| `MODEL_DIR` | `/model` |
| `VLLM_URL` | `http://vllm:8000/v1` |
| user | `${HOST_UID}:${HOST_GID}` (written to `.env` by `prep.sh`) |

Mounts: index (read-write), `rag_cache` (read-only), model folder (read-only).

### Open WebUI

`OPENAI_API_BASE_URL=http://orchestrator:8100/v1`, `OFFLINE_MODE=true`, title, tag, autocomplete and follow-up generation disabled, `WEBUI_AUTH=false` (takes effect only on a fresh data volume). Chat history lives in the `openwebui-data` volume.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| WSL files or Docker missing after boot | External virtual disk not attached | Attach it with `diskpart`, then restart WSL and Docker Desktop |
| Containers vanished after a crash | Docker state lost | `docker compose up -d`; rebuild images if they are gone too |
| Orchestrator `unhealthy` | Image built without packages | `./prep.sh` (it errors on missing versions), then `docker compose build --no-cache orchestrator` |
| vLLM `unhealthy` or slow to start | Model still loading, or GPU memory in use | Wait about 3 minutes; check `nvidia-smi`; read `docker compose logs vllm` |
| Tiny KV cache or out-of-memory | Compile mode or other GPU programs | Keep the flags in the vLLM config above; close other GPU apps |
| Port 8000, 8100 or 3000 in use | Leftover process | `docker compose down`, then `docker ps` |
| Open WebUI shows no model | Orchestrator not ready | Wait for `healthy`, refresh the model list |
| UI asks for login | Data volume already existed | `docker compose down -v` for a fresh volume (deletes history) |
| `pip` installs in the wrong place | `pip` on PATH is not the venv's | Use `python -m pip` from the venv |

---

## Known limitations

- **Groundedness is 75%, not 90%.** The model was fine-tuned closed-book (no retrieved context in training) but is served open-book. It sometimes echoes prompt instructions and can overstate the context. This is a hypothesis for the groundedness gap; a closed-book versus open-book comparison has not been run.
- **Pipeline first token is about 690 ms**, against 77 ms for the model server alone. Per-stage timings have not been profiled yet.
- **Router coverage is 63%.** Misses fall back to `general`, which still searches all sources, so retrieval quality stays high (hit@3 97%), but route-specific tuning is lost.
- **Compound questions** can be answered in part. Example: a question asking both how an attack works and how to detect it returned mitigations only.
- **Code answers.** 65% parse on the first try; failures are mostly dropped indentation. Tar extraction and TLS patterns scored poorly in the held-out set.
- **Small domain and sources.** Three public sources and a rule-based router. Answers can overstate; always check cited sources for anything important.
- **The internal-only network override is untested**, as is a full Wi-Fi-off run in your environment until you do it.
- **VRAM use while serving is 7.4 GB**, above a 6.5 GB target.

---

## Roadmap

1. **Per-stage timing** of embed, BM25, rerank, compress and prompt; parallelize BM25 and dense search.
2. **Facet-aware retrieval.** Tag ATT&CK chunks by field (detection, mitigation) and retrieve the field the question asks for.
3. **Split compound questions** into sub-queries.
4. **Per-sentence support check** after generation, with abstention when retrieval confidence is low.
5. **Closed-book versus open-book ablation** on the 76 items, then an **open-book fine-tune** (about 2 hours of training plus merge and requantization) to close the groundedness gap.
6. **Embedding-based router** alongside the keyword rules.
7. Pin the Open WebUI image to a digest; add an evaluation regression check.

---

## Licensing and data sources

- **Model:** Meta LLaMA 3.1 8B. Use of the base weights and any derivative is governed by the Llama 3.1 Community License. Review its terms before sharing weights.
- **Knowledge sources:** MITRE ATT&CK, NIST security controls and OWASP Cheat Sheets, each under its own terms. Check each source's license before redistributing derived data.
- **Code in this repository:** add a license file of your choice before publishing.
- **Not included in the repository:** model weights, the search index and training checkpoints.

---

*Author: Rishy.*
