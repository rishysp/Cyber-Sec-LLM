"""Stage 6a - build the evaluation set (fixed seed -> reproducible). Writes data/processed/eval_set.jsonl.

  hand : 30 hand-written natural questions (smoke set + extras) with gold evidence and optional answer keywords
  gen  : questions generated from the corpus metadata (technique names, control titles, cheat-sheet names). They are
         NAME-based (the id is not in the question), so they test retrieval, not id lookup.
Code-remediation items are not generated here: run_eval.py samples them from data/processed/pairs_code.jsonl (14 seen patterns + 3 held out).

HONEST LIMITS: gold evidence is the source record, answers are scored by keyword + judge, and all questions come from
the same public sources the model was trained on. This measures the system, not general security expertise.
"""
import json, os, pathlib, random
ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
P = ROOT / "data" / "processed"
rng = random.Random(11)
items = []


def add(typ, q, kind, key, terms=None, src="gen"):
    items.append({"id": f"{src}-{len(items):03d}", "type": typ, "src": src, "question": q,
                  "gold": {"kind": kind, "key": key}, "any_terms": terms or []})

HAND = [  # (type, question, kind, key, any_terms)
    ("attack", "What is MITRE ATT&CK technique T1059.001?", "id", "attack:T1059.001:", ["powershell"]),
    ("attack", "Explain T1003 OS Credential Dumping", "id", "attack:T1003:", ["credential"]),
    ("attack", "Describe the Phishing technique T1566", "id", "attack:T1566:", ["phish"]),
    ("attack", "What is Valid Accounts T1078 and how is it abused?", "id", "attack:T1078:", ["valid account", "credential"]),
    ("attack", "Explain process injection (T1055)", "id", "attack:T1055:", ["inject"]),
    ("attack", "How do adversaries dump credentials from LSASS memory?", "id", "attack:T1003.001:", ["lsass"]),
    ("attack", "Which technique covers abusing scheduled tasks for persistence?", "id", "attack:T1053.005:", ["scheduled task", "schtasks", "task scheduler"]),
    ("attack", "How does registry run key persistence work?", "id", "attack:T1547.001:", ["registry", "run key"]),
    ("nist", "What does NIST SP 800-53 AC-2 require?", "id", "nist:AC-2:", ["account"]),
    ("nist", "Which control covers boundary protection, SC-7?", "id", "nist:SC-7:", ["boundary", "managed interface"]),
    ("nist", "What does NIST IA-5 say about authenticator management?", "id", "nist:IA-5:", ["authenticator"]),
    ("nist", "Which 800-53 control requires multi-factor authentication for users?", "id", "nist:IA-2", ["multi", "factor", "authenticat"]),
    ("nist", "Which NIST control covers audit record review and analysis?", "id", "nist:AU-6:", ["audit", "review"]),
    ("nist", "Which NIST control requires flaw remediation and patching?", "id", "nist:SI-2", ["flaw", "patch", "remediat"]),
    ("nist", "Which control enforces least privilege?", "id", "nist:AC-6", ["least privilege"]),
    ("nist", "Which control covers baseline configurations of systems?", "id", "nist:CM-2", ["baseline", "configuration"]),
    ("owasp", "How do I prevent SQL injection?", "title", ["sql injection"], ["parameteri", "prepared"]),
    ("owasp", "How should passwords be stored securely?", "title", ["password storage"], ["argon2", "bcrypt", "pbkdf2", "scrypt"]),
    ("owasp", "How do I defend against cross-site scripting?", "title", ["cross site scripting", "cross-site scripting", "xss"], ["encod", "escap", "sanitiz"]),
    ("owasp", "How do I prevent CSRF attacks?", "title", ["cross-site request forgery", "csrf"], ["token"]),
    ("owasp", "How should session management be implemented securely?", "title", ["session management"], ["httponly", "secure", "expire", "timeout", "regenerat", "cookie"]),
    ("owasp", "How do I validate file uploads safely?", "title", ["file upload"], ["extension", "content", "type", "size", "store"]),
    ("owasp", "How should JSON Web Tokens be handled securely?", "title", ["json web token", "jwt"], ["signature", "algorithm", "expir"]),
    ("owasp", "How do I defend against server-side request forgery?", "title", ["server side request forgery", "ssrf", "server-side request forgery"], ["allow", "validat", "url"]),
    ("owasp", "How do I prevent XML external entity (XXE) attacks?", "title", ["xml external entity", "xxe"], ["dtd", "disable", "external"]),
    ("owasp", "How should an application log security events?", "title", ["logging"], ["log", "event"]),
    ("owasp", "What are good practices for securing a REST API?", "title", ["rest"], ["https", "authentic", "token", "validat"]),
    ("owasp", "What are Docker container security best practices?", "title", ["docker"], ["root", "image", "privilege", "secret"]),
    ("owasp", "How do I securely handle deserialization of untrusted data?", "title", ["deserialization"], ["untrusted", "integrity", "validat", "signature"]),
    ("owasp", "How do I prevent OS command injection?", "title", ["os command injection", "command injection"], ["shell", "parameter", "argument", "avoid", "validat"]),
]
for t, q, kind, key, terms in HAND:
    add(t, q, kind, key, terms, src="hand")

attack = json.load(open(P / "attack.json")); nist = json.load(open(P / "nist.json")); owasp = json.load(open(P / "owasp.json"))
techs = rng.sample(attack, 20)
for i, t in enumerate(techs):
    nm = t["name"]
    if i % 3 == 0:
        add("attack", f"What is the MITRE ATT&CK technique '{nm}' and how do adversaries use it?", "id", f"attack:{t['id']}:", [nm.lower(), t["id"].lower()])
    elif i % 3 == 1 and t["tactics"]:
        add("attack", f"Which ATT&CK tactic does the technique '{nm}' belong to?", "id", f"attack:{t['id']}:", [x.lower() for x in t["tactics"]])
    else:
        add("attack", f"How can the technique '{nm}' be mitigated or detected?", "id", [f"attack:{t['id']}:", f"attack:{t['id']}-mit:"], [nm.lower(), "mitigat", "detect", "monitor"])
for c in rng.sample([c for c in nist if len(c["statement"]) > 80], 12):
    add("nist", f"What does NIST SP 800-53 require for the control '{c['title']}'?", "id", f"nist:{c['id']}:", [c["id"].lower(), c["title"].lower()])
docs = sorted({r["doc"] for r in owasp if r["source_tag"] == "OWASP Cheat Sheet"})
for d in rng.sample(docs, 14):
    topic = d.replace(" Cheat Sheet", "")
    add("owasp", f"What are the key security recommendations for {topic}?", "title", [d.lower()], [])

with open(P / "eval_set.jsonl", "w", encoding="utf-8") as f:
    for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
import collections
print("eval items:", dict(collections.Counter((i["src"], i["type"]) for i in items)), "total", len(items), "->", P / "eval_set.jsonl")
