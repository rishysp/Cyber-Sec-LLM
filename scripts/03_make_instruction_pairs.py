"""Stage 1c - build grounded instruction pairs from the structured records (no LLM needed, no hallucinated answers).
Answers come straight from the authoritative text; the model learns domain vocabulary, ID<->name mapping and answer style.
"""
import argparse, json, os, pathlib, random, re

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
P = ROOT / "data" / "processed"

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, default=2400)
ap.add_argument("--seed", type=int, default=1337)
args = ap.parse_args()
rng = random.Random(args.seed)


def trim_sent(t, n):
    t = t.strip()
    if len(t) <= n:
        return t
    cut = t[:n]
    i = cut.rfind(". ")
    return cut[: i + 1] if i > n * 0.5 else cut.rstrip() + "..."


def trim_para(t, n):
    if len(t) <= n:
        return t
    out = ""
    for para in t.split("\n\n"):
        if len(out) + len(para) + 2 > n:
            break
        out += ("\n\n" if out else "") + para
    if out.count("```") % 2:                      # never leave a dangling code fence
        out = out[: out.rfind("```")].rstrip()
    return out


def first_sentence(t):
    m = re.match(r"(.+?[.!?])(\s|$)", t.strip(), re.S)
    return m.group(1).strip() if m else ""


def pick(opts, **kw):
    return rng.choice(opts).format(**kw)


pairs = []


def add(source, group, kind, q, a):
    if q and a:
        pairs.append({"source": source, "group": group, "kind": kind, "instruction": q, "output": a})


# ---------- ATT&CK ----------
for t in json.load(open(P / "attack.json")):
    tac, plat, g = ", ".join(t["tactics"]) or "unspecified", ", ".join(t["platforms"]) or "unspecified", t["group"]
    kw = dict(id=t["id"], name=t["name"])
    add("attack", g, "tech_def",
        pick(["What is MITRE ATT&CK technique {id} ({name})?", "Explain {name} ({id}) from MITRE ATT&CK.",
              "Describe the adversary behavior covered by {id}, {name}."], **kw),
        f"{t['name']} ({t['id']}) is an ATT&CK technique under the {tac} tactic. " + trim_sent(t["description"], 1400))
    add("attack", g, "tech_tactic",
        pick(["Which tactic does {name} ({id}) belong to, and on which platforms does it apply?",
              "Map {id} to its ATT&CK tactic(s) and platforms."], **kw),
        f"{t['name']} ({t['id']}) maps to the {tac} tactic(s) and applies to: {plat}.")
    fs = first_sentence(t["description"])
    if 40 < len(fs) < 300:
        add("attack", g, "tech_reverse", f"Which MITRE ATT&CK technique is described as follows: \"{fs}\"?",
            f"That describes {t['name']} ({t['id']}), a technique under the {tac} tactic.")
    if t["mitigations"]:
        lines = "\n".join(f"- {m['name']}: {trim_sent(m['how'], 300)}" for m in t["mitigations"][:6])
        add("attack", g, "tech_mitig",
            pick(["How can {name} ({id}) be mitigated?", "List ATT&CK mitigations for {id} ({name})."], **kw),
            f"Mitigations mapped to {t['name']} ({t['id']}):\n{lines}")
    if t["detection"]:
        add("attack", g, "tech_detect", pick(["How can {name} ({id}) be detected?", "What detection guidance exists for {id}?"], **kw),
            trim_sent(t["detection"], 1200))

# ---------- NIST SP 800-53 Rev 5 ----------
for c in json.load(open(P / "nist.json")):
    kw = dict(id=c["id"], title=c["title"], family=c["family"])
    g = c["group"]
    add("nist", g, "ctrl_family", pick(["Which NIST SP 800-53 family does {id} belong to?", "What is control {id} ({title}) and where does it sit in SP 800-53?"], **kw),
        f"{c['id']} ({c['title']}) belongs to the {c['family']} family of NIST SP 800-53 Rev 5.")
    if c["statement"]:
        add("nist", g, "ctrl_statement", pick(["What does NIST SP 800-53 control {id} ({title}) require?", "State the requirement of {id}, {title}."], **kw),
            f"{c['id']} ({c['title']}), {c['family']} family. Requirement:\n" + trim_para(c["statement"], 1500))
    if c["guidance"]:
        add("nist", g, "ctrl_guidance", pick(["Explain the purpose and implementation guidance of NIST {id} ({title}).", "Why does {id} ({title}) exist and how is it applied?"], **kw),
            trim_sent(c["guidance"], 1400))

# ---------- OWASP ----------
for r in json.load(open(P / "owasp.json")):
    body = trim_para(r["body"], 1500)
    if len(body) < 200:
        continue
    kw = dict(doc=r["doc"], head=r["heading"], tag=r["source_tag"])
    add("owasp", r["group"], "owasp_section",
        pick(["According to {tag} guidance on {doc}, what should I know about '{head}'?", "What does OWASP recommend under '{head}' in {doc}?"], **kw),
        f"From the OWASP {r['doc']} guidance, section '{r['heading']}':\n{body}")

# ---------- balanced sample ----------
SHARE = {"attack": 0.40, "nist": 0.35, "owasp": 0.25}
sel = []
for src, share in SHARE.items():
    pool = [p for p in pairs if p["source"] == src]
    rng.shuffle(pool)
    sel += pool[: int(args.target * share)]
    print(f"{src}: available={len(pool)} used={min(len(pool), int(args.target * share))}")
rng.shuffle(sel)
with open(P / "pairs_raw.jsonl", "w", encoding="utf-8") as f:
    for p in sel:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("total pairs:", len(sel))
