"""Stage 5d - LangGraph orchestration.

    classify -> retrieve -> build_prompt -> [generate -> (code intent) ast_check -> (fail, 1 retry) generate] -> END

Two compiled variants share the same nodes:
  prepare graph : stops after build_prompt (the server then STREAMS the answer itself for low perceived latency)
  full graph    : also generates and runs the AST gate (used for code questions and for offline evaluation)

The router is rule-based on purpose: it costs ~0 ms, is deterministic, and is easy to defend. An LLM classifier would
spend a slice of the 2.5 s budget (and ~60 tokens/s decode) on every request just to pick a metadata filter.
"""
import ast, re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from common import *
from context import compress
import llm as llm_mod

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
RULES = {
    "attack": r"\bT\d{4}(?:\.\d{3})?\b|mitre|att&ck|attack technique|tactic|adversar|threat actor|apt\d*|persistence|lateral movement|privilege escalation|defense evasion|credential access|c2\b|command and control",
    "nist": r"nist|800-53|\b[A-Z]{2}-\d{1,2}(?:\(\d{1,2}\))?(?!\w)|security control|control famil|compliance|fedramp|baseline",
    "owasp": r"owasp|top ?10|cheat ?sheet|\bA0?\d:20\d\d|injection|xss|cross[- ]site|csrf|ssrf|deserializ|insecure|authentication|session management|password storage|input validation",
}
CODE_ASK = r"fix (?:this|the)|remediat|secure (?:version|rewrite)|vulnerable (?:code|function)|patch (?:this|the)|rewrite.*(?:safe|secure)|make (?:this|it) secure"
SRC_FOR_CODE = ["owasp"]


VULN_HINTS = [  # (regex over the submitted code, retrieval phrase). Mirrors the 17 remediation patterns used in training.
    (r"(execute|executemany)\s*\(.*(\+|%|\bf['\"]|\.format)|SELECT\s.*['\"]\s*\+|SELECT\s.*\{", "SQL injection prevention parameterized queries"),
    (r"os\.system|os\.popen|shell\s*=\s*True", "OS command injection defense"),
    (r"(password|passwd|secret|api_?key|token)\s*=\s*['\"][^'\"]{4,}", "secrets management hard-coded credentials"),
    (r"pickle\.loads?|marshal\.loads", "deserialization insecure deserialization"),
    (r"hashlib\.(md5|sha1)", "password storage hashing algorithm"),
    (r"open\(.*\+|os\.path\.join\(.*(request|user|filename|path)", "path traversal file path"),
    (r"yaml\.load\(", "unsafe yaml deserialization"),
    (r"verify\s*=\s*False|CERT_NONE|check_hostname\s*=\s*False", "transport layer security certificate validation"),
    (r"random\.(random|randint|choice|choices)", "cryptographic storage secure random token generation"),
    (r"render_template_string|Markup\(|return\s+f?['\"]<", "cross site scripting prevention output encoding"),
    (r"\beval\(|\bexec\(", "code injection eval"),
    (r"requests\.(get|post)\(|urlopen\(", "server side request forgery prevention"),
    (r"debug\s*=\s*True", "secure configuration debug mode"),
    (r"extractall|ZipFile", "path traversal zip slip archive extraction"),
    (r"mktemp\(|/tmp/", "insecure temporary file"),
    (r"autoescape\s*=\s*False", "cross site scripting template escaping"),
    (r"set_cookie", "session management cookie secure httponly samesite"),
]


def vuln_hint(text: str) -> str:
    """Code-aware retrieval query: map risky constructs in the submitted code to the vulnerability class, because the raw
    code (variable names, boilerplate) retrieves unrelated cheat sheets (e.g. Laravel) instead of the right defense."""
    return " ".join(dict.fromkeys(h for rx, h in VULN_HINTS if re.search(rx, text, re.I | re.S)))


def prefer_for(intent):
    """How-to / remediation questions want prescriptive guidance (cheat sheets), not Top-10 risk overviews."""
    return "OWASP Cheat Sheet" if intent in ("owasp", "code") else None


class S(TypedDict, total=False):
    question: str
    intent: str
    sources: Optional[List[str]]
    chunks: list
    context: str
    messages: list
    answer: str
    attempts: int
    ast_ok: Optional[bool]
    ast_error: str
    usage: dict
    timings: dict
    ctx_tokens: int
    ctx_list: list


def classify_rules(q: str):
    if "```" in q or re.search(CODE_ASK, q, re.I) and re.search(r"\bdef \w+\(|import \w+|python", q, re.I):
        return "code", SRC_FOR_CODE
    scores = {k: len(re.findall(p, q, re.I)) for k, p in RULES.items()}
    top = sorted(scores.values(), reverse=True)
    if top[0] == 0 or top[0] == top[1]:          # no signal, or a tie between sources -> search everything
        return "general", None
    best = max(scores, key=scores.get)
    return best, [best]


_LEAD = re.compile(r"^\s*From the context[^\n]*:[ \t]*\n+", re.I)


def strip_lead(text: str) -> str:
    """Drop a leading 'From the context [...]:' line some answers start with (sources are shown in the footer anyway)."""
    return _LEAD.sub("", text, count=1)


_ECHO = re.compile(r"answer the question directly|in your own words|do not copy the context|do not repeat its headings|"
                   r"if it asks how to do or prevent|give the concrete steps or controls|start with .from the context", re.I)


def drop_echoed_instructions(text: str) -> str:
    """The 8B model sometimes parrots the prompt's instruction tail. Remove those sentences (never inside code fences)."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i, p in enumerate(parts):
        if i % 2 == 0 and _ECHO.search(p):
            parts[i] = " ".join(x for x in re.split(r"(?<=[.!?])\s+", p) if not _ECHO.search(x))
    return "".join(parts)


def tidy(text: str) -> str:
    cleaned = drop_echoed_instructions(strip_lead(text)).strip()
    return cleaned or text


def format_entries(chunks, parts):
    """One entry per chunk. OWASP entries carry the doc/section title as the label and drop the duplicated header line:
    repeating titles inside the context made the 8B model echo them back instead of answering."""
    out = []
    for c, p in zip(chunks, parts):
        head, _, body = p.partition("\n")
        out.append(f"[{label(c)}]\n{body or head}" if c["source"] == "owasp" else f"[{label(c)}] {p}")
    return out


def format_context(chunks, parts):
    return "\n\n".join(format_entries(chunks, parts))


def check_python(text: str):
    """AST gate: every python code block must parse. NOTE: proves SYNTAX only, never that the fix is secure."""
    blocks = CODE_FENCE.findall(text)
    if not blocks: return False, "no ```python code block found"
    for b in blocks:
        try: ast.parse(b)
        except SyntaxError as e: return False, f"SyntaxError line {e.lineno}: {e.msg}"
    return True, ""


def build_graph(retriever, count, prepare_only=False, ctx_budget=450, n_chunks=3):
    def classify(s: S):
        intent, sources = classify_rules(s["question"])
        return {"intent": intent, "sources": sources, "attempts": 0}

    def retrieve(s: S):
        q = s["question"]
        if s["intent"] == "code":
            q = (vuln_hint(q) + " " + CODE_FENCE.sub("", q)).strip() or q
        res, t = retriever.search(q, sources=s["sources"], n=n_chunks, prefer=prefer_for(s["intent"]))
        return {"chunks": res, "timings": t}

    def build_prompt(s: S):
        parts = compress(s["chunks"], s["question"], ctx_budget, count)
        entries = format_entries(s["chunks"], parts)
        ctx = "\n\n".join(entries)
        if s["intent"] == "code":
            tail = ("Return the corrected code in ONE ```python block, then one sentence explaining the fix. "
                    "Do not repeat the context.")
        else:
            tail = ("Answer the question directly in 2 to 4 sentences, in your own words, using the context. "
                    "If it asks how to do or prevent something, give the concrete steps or controls. "
                    "Do not copy the context, do not repeat its headings, and do not start with 'From the context'.")
        msgs = [{"role": "system", "content": SYSTEM + RAG_RULES},
                {"role": "user", "content": f"Question: {s['question']}\n\nContext:\n{ctx}\n\n{tail}"}]
        return {"context": ctx, "messages": msgs, "ctx_tokens": count(ctx), "ctx_list": entries}

    def generate(s: S):
        mt = 380 if s["intent"] == "code" else 220
        text, usage = llm_mod.chat(s["messages"], max_tokens=mt)
        return {"answer": tidy(text), "usage": usage, "attempts": s.get("attempts", 0) + 1}

    def ast_node(s: S):
        ok, err = check_python(s["answer"])
        upd = {"ast_ok": ok, "ast_error": err}
        if not ok and s["attempts"] < 2:       # one repair attempt, feeding the parser error back
            upd["messages"] = s["messages"] + [
                {"role": "assistant", "content": s["answer"]},
                {"role": "user", "content": f"That code block does not parse ({err}). "
                                            "Return only a corrected ```python block."}]
        return upd

    def after_build(s: S):
        return END if prepare_only else "generate"

    def after_generate(s: S):
        return "ast_check" if s["intent"] == "code" else END

    def after_ast(s: S):
        return "generate" if (not s["ast_ok"] and s["attempts"] < 2) else END

    g = StateGraph(S)
    for name, fn in [("classify", classify), ("retrieve", retrieve), ("build_prompt", build_prompt)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify"); g.add_edge("classify", "retrieve"); g.add_edge("retrieve", "build_prompt")
    if prepare_only:
        g.add_edge("build_prompt", END)
    else:
        g.add_node("generate", generate); g.add_node("ast_check", ast_node)
        g.add_conditional_edges("build_prompt", after_build, ["generate", END])
        g.add_conditional_edges("generate", after_generate, ["ast_check", END])
        g.add_conditional_edges("ast_check", after_ast, ["generate", END])
    return g.compile()
