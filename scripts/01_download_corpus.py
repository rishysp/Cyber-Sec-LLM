"""Stage 1a - one-time corpus download. This is the ONLY step (besides model downloads) that needs internet."""
import datetime, hashlib, io, json, os, pathlib, zipfile
import requests

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

FILES = {
    # MITRE ATT&CK Enterprise as STIX 2.1 JSON (free to use with attribution)
    "attack_enterprise.json": "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json",
    # NIST SP 800-53 Rev 5 official OSCAL catalog (US government work, public domain)
    "nist_800_53r5_catalog.json": "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json",
}
ZIPS = {
    # OWASP Cheat Sheet Series and Top 10 (CC BY-SA licensed markdown)
    "owasp_cheatsheets": "https://github.com/OWASP/CheatSheetSeries/archive/HEAD.zip",
    "owasp_top10": "https://github.com/OWASP/Top10/archive/HEAD.zip",
}

manifest = {"downloaded_utc": datetime.datetime.utcnow().isoformat(), "items": {}}

def get(url):
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return r.content

for name, url in FILES.items():
    data = get(url)
    (RAW / name).write_bytes(data)
    manifest["items"][name] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    print(f"ok {name}: {len(data)/1e6:.1f} MB")

for name, url in ZIPS.items():
    data = get(url)
    dest = RAW / name
    dest.mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    manifest["items"][name] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    print(f"ok {name}: {len(data)/1e6:.1f} MB")

(RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("Provenance written to", RAW / "MANIFEST.json")
