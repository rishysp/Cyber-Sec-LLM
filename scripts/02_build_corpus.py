"""Stage 1b - parse raw sources into clean structured records (for pair generation) and RAG chunks."""
import collections, json, os, pathlib, re

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def clean(t):
    t = re.sub(r"\(Citation:[^)]*\)", "", t or "")
    t = re.sub(r"\[([^\]]+)\]\(https?://[^)]+\)", r"\1", t)   # markdown links -> text
    t = t.replace("<code>", "`").replace("</code>", "`")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# ---------------- MITRE ATT&CK ----------------
def load_attack():
    bundle = json.load(open(RAW / "attack_enterprise.json", encoding="utf-8"))
    objs = [o for o in bundle["objects"] if not o.get("revoked") and not o.get("x_mitre_deprecated")]
    by_id = {o["id"]: o for o in objs}
    mitig = collections.defaultdict(list)
    for o in objs:
        if o["type"] == "relationship" and o["relationship_type"] == "mitigates":
            s, t = o["source_ref"], o["target_ref"]
            if s in by_id and t in by_id:
                mitig[t].append({"name": by_id[s]["name"], "how": clean(o.get("description", ""))})
    techs = []
    for o in objs:
        if o["type"] != "attack-pattern":
            continue
        ext = next((r["external_id"] for r in o.get("external_references", []) if r.get("source_name") == "mitre-attack"), None)
        if not ext:
            continue
        techs.append({
            "id": ext, "name": o["name"], "group": ext.split(".")[0],
            "tactics": [p["phase_name"].replace("-", " ").title() for p in o.get("kill_chain_phases", [])],
            "platforms": o.get("x_mitre_platforms", []),
            "description": clean(o.get("description", "")),
            "detection": clean(o.get("x_mitre_detection", "")),   # may be empty in newer ATT&CK releases
            "mitigations": mitig.get(o["id"], []),
        })
    return techs


# ---------------- NIST SP 800-53 Rev 5 (OSCAL) ----------------
def part_text(p):
    lab = next((x["value"] for x in p.get("props", []) if x["name"] == "label"), "")
    lines = []
    if p.get("prose"):
        lines.append((lab + " " if lab else "") + p["prose"])
    for s in p.get("parts", []):
        t = part_text(s)
        if t:
            lines.append(t)
    return "\n".join(lines)


def param_fix(t):
    return re.sub(r"\{\{\s*insert:\s*param,\s*[^}]+\}\}", "[organization-defined value]", t)


def load_nist():
    cat = json.load(open(RAW / "nist_800_53r5_catalog.json", encoding="utf-8"))["catalog"]
    out = []

    def walk(ctrls, fam_title):
        for c in ctrls:
            props = {p["name"]: p["value"] for p in c.get("props", [])}
            if props.get("status") == "withdrawn":
                continue
            parts = {p["name"]: p for p in c.get("parts", [])}
            out.append({
                "id": props.get("label", c["id"].upper()), "title": c["title"], "family": fam_title,
                "group": c["id"].split(".")[0],
                "statement": param_fix(part_text(parts["statement"])) if "statement" in parts else "",
                "guidance": param_fix(part_text(parts["guidance"])) if "guidance" in parts else "",
            })
            walk(c.get("controls", []), fam_title)

    for g in cat["groups"]:
        walk(g.get("controls", []), g["title"])
    return out


# ---------------- OWASP ----------------
SKIP = {"references", "related articles", "tools", "contributors", "authors and primary editors", "other cheat sheets"}


def load_owasp(max_len=8000):
    """max_len: sections with a longer body are skipped. The TRAINING json keeps 8000 (unchanged); the RAG chunks use a much
    larger cap, because the longest sections (e.g. SQL injection 'Primary Defenses') are exactly the ones users ask about."""
    recs = []
    for folder, tag in (("owasp_cheatsheets", "OWASP Cheat Sheet"), ("owasp_top10", "OWASP Top 10")):
        for md in (RAW / folder).rglob("*.md"):
            p = md.as_posix()
            if folder == "owasp_cheatsheets" and "/cheatsheets/" not in p:
                continue
            if folder == "owasp_top10" and "/docs/en/" not in p:
                continue
            text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md.read_text(encoding="utf-8", errors="ignore"))
            m = re.search(r"^#\s+(.+)$", text, re.M)
            title = (m.group(1) if m else md.stem).strip()
            for sec in re.split(r"^##\s+", text, flags=re.M)[1:]:
                head, _, body = sec.partition("\n")
                head, body = head.strip(), clean(body)
                if head.lower() in SKIP or not (250 <= len(body) <= max_len):
                    continue
                recs.append({"source_tag": tag, "doc": title, "heading": head, "body": body, "group": f"{folder}/{md.stem}"})
    return recs


# ---------------- chunking ----------------
def chunk_text(text, max_words=180):
    """Line-aware chunker: keeps code blocks intact-ish, 2-line overlap between chunks."""
    chunks, cur, n = [], [], 0
    for line in text.split("\n"):
        lw = len(line.split())
        if n + lw > max_words and cur:
            chunks.append("\n".join(cur))
            cur = cur[-2:]
            n = sum(len(l.split()) for l in cur)
        cur.append(line)
        n += lw
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def main():
    techs, ctrls, owasp = load_attack(), load_nist(), load_owasp()          # owasp: training view (<= 8000 chars)
    owasp_rag = load_owasp(max_len=60000)                                    # owasp: RAG view (long sections kept)
    print(f"owasp sections: training view={len(owasp)}  rag view={len(owasp_rag)}  (recovered {len(owasp_rag)-len(owasp)} long sections)")
    json.dump(techs, open(OUT / "attack.json", "w"), indent=1)
    json.dump(ctrls, open(OUT / "nist.json", "w"), indent=1)
    json.dump(owasp, open(OUT / "owasp.json", "w"), indent=1)

    chunks = []

    def add(src, ref, header, body):
        for i, c in enumerate(chunk_text(body)):
            chunks.append({"id": f"{src}:{ref}:{i}", "source": src, "title": header, "text": f"{header}\n{c}"})

    for t in techs:
        head = f"MITRE ATT&CK {t['id']} {t['name']} (Tactics: {', '.join(t['tactics'])}; Platforms: {', '.join(t['platforms'])})"
        add("attack", t["id"], head, t["description"])
        extra = ""
        if t["mitigations"]:
            extra += "Mitigations:\n" + "\n".join(f"- {m['name']}: {m['how']}" for m in t["mitigations"])
        if t["detection"]:
            extra += "\nDetection: " + t["detection"]
        if extra.strip():
            add("attack", t["id"] + "-mit", head, extra.strip())
    for c in ctrls:
        body = f"Statement: {c['statement']}\nGuidance: {c['guidance']}".strip()
        add("nist", c["id"], f"NIST SP 800-53 Rev 5 {c['id']} {c['title']} (Family: {c['family']})", body)
    for i, r in enumerate(owasp_rag):
        add("owasp", f"{r['group']}#{i}", f"{r['source_tag']}: {r['doc']} - {r['heading']}", r["body"])

    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    cnt = collections.Counter(c["source"] for c in chunks)
    print(f"techniques={len(techs)} controls={len(ctrls)} owasp_sections={len(owasp)}")
    print("chunks:", dict(cnt), "total", len(chunks))


if __name__ == "__main__":
    main()
