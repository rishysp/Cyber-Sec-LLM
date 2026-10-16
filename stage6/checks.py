"""Stage 6 - verification of remediation snippets, beyond 'does it parse'.

Three levels, reported separately so each claim is honest:
  1. syntax   : ast.parse                      (the resume's 'AST pass rate'; proves NOTHING about security)
  2. pattern  : the vulnerable construct is gone AND the expected safe construct is present (regex rules per pattern;
                a HEURISTIC - it can be fooled, and it can reject valid but unusual fixes)
  3. behavior : for SQL injection only - run the candidate against an in-memory SQLite DB with an injection payload
                (executes model-generated code in a subprocess with a timeout; fine for this lab, never do it on a server)
"""
import ast, json, os, re, subprocess, sys, tempfile, textwrap

F = re.I | re.S
# pid -> (forbidden regex or None, [required regexes])
RULES = {
    "sqli":   (r"(SELECT|INSERT|UPDATE|DELETE)[^\n]*(['\"]\s*\+|\+\s*['\"]|['\"]\s*%\s*[\(\w]|\.format\(|\{[A-Za-z_]+\})",
               [r"execute\w*\s*\([^,)]+,\s*[\(\[\w{]"]),
    "cmdi":   (r"os\.system|os\.popen|shell\s*=\s*True", []),
    "secret": (r"(password|passwd|secret|api_?key|token)\w*\s*=\s*['\"][^'\"]{4,}['\"]", [r"environ|getenv|keyring|vault|secretmanager"]),
    "pickle": (r"pickle\.loads?\(", [r"json"]),
    "hash":   (r"hashlib\.(md5|sha1)\(", [r"bcrypt|argon2|scrypt|pbkdf2|hashlib\.(sha256|sha512)|werkzeug"]),
    "path":   (None, [r"realpath|resolve\(|commonpath|secure_filename|abspath|basename", r"startswith|commonpath|is_relative_to|relative_to|basename|secure_filename|\.parents"]),
    "yaml":   (r"yaml\.load\((?![^)]*Safe)", [r"safe_load|SafeLoader"]),
    "tls":    (r"verify\s*=\s*False|CERT_NONE|check_hostname\s*=\s*False", []),
    "random": (r"\brandom\.(random|randint|choice|choices|getrandbits)\(", [r"secrets|SystemRandom|urandom|token_"]),
    "xss":    (r"render_template_string\([^)]*['\"]\s*\+|render_template_string\(\s*f['\"]|return\s+f?['\"]<[^\n]*(\+|\{)",
               [r"escape|markupsafe|render_template\(|render_template_string\([^)]*\{\{[^)]*,\s*\w+\s*="]),
    "eval":   (r"(?<![\w.])eval\(|(?<![\w.])exec\(", []),
    "ssrf":   (None, [r"allow|urlparse|ipaddress|netloc|hostname|scheme|whitelist"]),
    "debug":  (r"debug\s*=\s*True|host\s*=\s*['\"]0\.0\.0\.0['\"]", []),
    "tar":    (None, [r"commonpath|realpath|resolve\(|startswith|filter\s*=|is_relative_to|abspath"]),
    "tmp":    (r"mktemp\(|['\"]/tmp/", [r"mkstemp|NamedTemporaryFile|TemporaryDirectory|mkdtemp"]),
    "jinja":  (r"autoescape\s*=\s*False", [r"autoescape\s*=\s*True|select_autoescape"]),
    "cookie": (None, [r"httponly\s*=\s*True", r"secure\s*=\s*True"]),
}
HELD_OUT = {"yaml", "tar", "cookie"}          # never seen in training (see 07_merge_and_split.py)
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def last_code(text: str) -> str:
    blocks = FENCE.findall(text)
    return blocks[-1] if blocks else ""


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code); return bool(code.strip())
    except SyntaxError:
        return False


def pattern_ok(pid: str, code: str) -> bool:
    forbid, require = RULES[pid]
    if not code.strip(): return False
    if forbid and re.search(forbid, code, F): return False
    return all(re.search(r, code, F) for r in require)


_HARNESS = textwrap.dedent('''
    import sqlite3, sys, json
    ns = {}
    exec(compile(open(sys.argv[1]).read(), "candidate", "exec"), ns)
    fn = ns[sys.argv[2]]
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
    conn.executemany("INSERT INTO users VALUES (?,?,?)", [(1, "alice", "a@x"), (2, "bob", "b@x"), (3, "carol", "c@x")])
    def rows(r):
        if r is None: return []
        if hasattr(r, "fetchall"): return r.fetchall()
        return r if isinstance(r, (list, tuple)) and (not r or isinstance(r[0], (list, tuple))) else [r]
    normal = len(rows(fn(conn, "alice")))
    inj = len(rows(fn(conn, "' OR '1'='1"))) + len(rows(fn(conn, "x' UNION SELECT id, name, email FROM users --")))
    print(json.dumps({"normal": normal, "inject": inj}))
''')


def sqli_behavior_ok(code: str):
    """True if the function still finds 'alice' AND returns nothing for injection payloads. None if it cannot be run."""
    m = re.search(r"def\s+(\w+)\s*\(\s*conn\s*,", code)
    if not m: return None
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "cand.py"), "w").write(code)
        open(os.path.join(d, "h.py"), "w").write(_HARNESS)
        try:
            p = subprocess.run([sys.executable, "-I", "h.py", "cand.py", m.group(1)], cwd=d, capture_output=True,
                               text=True, timeout=10)
            out = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return False
    return out["normal"] >= 1 and out["inject"] == 0
