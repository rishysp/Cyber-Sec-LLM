"""Stage 5 - shared config and helpers. Everything here is CPU-only: the GPU belongs to vLLM."""
import os, pathlib, re

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX = ROOT / "index"                       # chroma + bm25 live here (ext4, read-mostly)
CACHE = ROOT / "models" / "rag_cache"        # embedding + reranker weights, downloaded once, then offline
MODEL_DIR = pathlib.Path(os.environ.get("MODEL_DIR", ROOT / "models" / "llama31-8b-secure-awq"))

EMBED_MODEL = "BAAI/bge-small-en-v1.5"       # 384-d, ~130 MB, strong for its size
RERANK_MODEL = os.environ.get("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")   # FlashRank ONNX cross-encoder, ~34 MB
RERANK_CANDS = int(os.environ.get("RERANK_CANDS", "8"))      # candidates sent to the cross-encoder (CPU cost ~ linear)
RERANK_LEN = int(os.environ.get("RERANK_LEN", "256"))        # max tokens per candidate for the cross-encoder
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge: prefix queries only

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
LLM_NAME = os.environ.get("LLM_NAME", "secrag-llama")
LLM_REP = float(os.environ.get("LLM_REP", "1.05"))   # repetition_penalty; tune in Stage 6

SYSTEM = ("You are an offline cybersecurity assistant. Answer accurately and concisely using MITRE ATT&CK, "
          "OWASP and NIST SP 800-53 terminology. If you are unsure, say so.")
RAG_RULES = (" Use ONLY the provided context. Cite the source label in square brackets, e.g. [T1059.001]. "
             "If the context does not contain the answer, say you do not know. Answer in 2 to 4 sentences: what it is first, then how it works or how to fix it.")

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._()-][A-Za-z0-9]+)*")
_SPLIT = re.compile(r"[._()-]+")


def tok(text: str):
    """BM25 tokenizer that keeps identifiers whole ('t1059.001', 'ac-2') AND adds their parts ('t1059', '001').
    A default whitespace tokenizer would shred IDs, and IDs are exactly what security users type."""
    out = []
    for m in _TOKEN.findall(text.lower()):
        out.append(m)
        parts = [p for p in _SPLIT.split(m) if p]
        if len(parts) > 1:
            out.extend(parts)
    return out


def label(chunk: dict) -> str:
    """Short citation label: the technique/control id for ATT&CK and NIST, the document title for OWASP."""
    if chunk["source"] in ("attack", "nist"):
        return chunk["id"].split(":")[1].replace("-mit", " mitigations")
    t = chunk["title"]
    for p in ("OWASP Cheat Sheet: ", "OWASP Top 10: "):
        if t.startswith(p): t = t[len(p):]
    return t[:90]


def norm_ref(ref: str) -> str:
    """'ac-02' / 'AC-2' -> 'AC-2' (NIST sources disagree on zero padding), so IDs match however they are written."""
    return re.sub(r"(?<=[-(])0+(?=\d)", "", ref.upper())


def get_counter():
    """LLaMA-tokenizer token counter. no_truncation() matters: tokenizer.json may enable truncation at 512 tokens,
    which silently caps every count (and fakes a context-reduction number)."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tk.no_truncation()
    return lambda s: len(tk.encode(s, add_special_tokens=False).ids)


_ATTR = re.compile(r"[ \t]*\{:[^}\n]*\}?")


def clean(t: str) -> str:
    """Strip kramdown attribute residue such as '{: style="height:80px;width:80px"}' that leaked into OWASP headings."""
    return _ATTR.sub("", t)
