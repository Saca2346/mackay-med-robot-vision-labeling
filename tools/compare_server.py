"""
Compare the laptop copy of the labelling project with the server copy (read-only on both sides).

  1. laptop, cmd:  RUN_ON_SERVER.bat langkah7_manifest        (server lists its files + hashes)
  2. laptop, cmd:  python tools\\compare_server.py server_results\\langkah7_manifest.txt

Writes server_results/COMPARE_SERVER_LAPTOP.md (summary) and .csv (one row per file that differs). Nothing is copied or
changed: the report is for deciding which copy is the master.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def local_manifest() -> dict:
    out = {}
    for p in sorted((ROOT / "label_specs_expiry").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[("EXP", p.name)] = (d.get("status", ""), str(len(d.get("expiry_label", []))), sha(p))
    for folder, pat in (("labels_final_obb", "*.txt"), ("label_specs", "*.json"), ("tools", "*.py"),
                        ("server_scripts", "*.sh"), (".", "*.md"), (".", "*.csv"), (".", "classes.txt")):
        for p in sorted((ROOT / folder).glob(pat)):
            out[("FILE", p.relative_to(ROOT).as_posix())] = ("", "", sha(p))
    return out


def server_manifest(path: pathlib.Path) -> dict:
    out, ended = {}, False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "END":
            ended = True
        parts = line.split("|")
        if len(parts) == 5 and parts[0] in ("EXP", "FILE"):
            name = parts[1][2:] if parts[1].startswith("./") else parts[1]
            out[(parts[0], name)] = (parts[2], parts[3], parts[4])
    if not ended:
        sys.exit(f"{path}: the server list is incomplete (no END line). Run RUN_ON_SERVER.bat langkah7_manifest again.")
    return out


def main() -> None:
    srv = server_manifest(pathlib.Path(sys.argv[1]))
    loc = local_manifest()
    rows = []
    for key in sorted(set(srv) | set(loc)):
        s, l = srv.get(key), loc.get(key)
        if s and l and s[2] == l[2]:
            continue
        where = "server only" if not l else "laptop only" if not s else "different content"
        rows.append({"kind": key[0], "file": key[1], "where": where,
                     "server_status": s[0] if s else "", "laptop_status": l[0] if l else "",
                     "server_labels": s[1] if s else "", "laptop_labels": l[1] if l else ""})
    out = pathlib.Path(sys.argv[1]).parent
    with open(out / "COMPARE_SERVER_LAPTOP.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["kind", "file", "where"])
        w.writeheader()
        w.writerows(rows)
    exp_s = [v[0] for k, v in srv.items() if k[0] == "EXP"]
    exp_l = [v[0] for k, v in loc.items() if k[0] == "EXP"]
    both_exp = [r for r in rows if r["kind"] == "EXP" and r["where"] == "different content"]
    md = ["# Server vs laptop (labelling project)", "",
          f"Files compared: server {len(srv)}, laptop {len(loc)}; identical {len(set(srv) & set(loc)) - sum(1 for r in rows if r['where'] == 'different content')}.", "",
          "| | server | laptop |", "| --- | --- | --- |",
          f"| expiry specs `selesai` (done) | {exp_s.count('selesai')} | {exp_l.count('selesai')} |",
          f"| expiry specs `auto` | {exp_s.count('auto')} | {exp_l.count('auto')} |", "",
          "| Difference | Count |", "| --- | --- |"]
    for where in ("different content", "server only", "laptop only"):
        for kind in ("EXP", "FILE"):
            n = sum(1 for r in rows if r["where"] == where and r["kind"] == kind)
            if n:
                md.append(f"| {where}, {'expiry spec' if kind == 'EXP' else 'other file'} | {n} |")
    md += ["", "Expiry specs whose status differs:", ""]
    pairs = {}
    for r in both_exp:
        pairs[(r["server_status"], r["laptop_status"])] = pairs.get((r["server_status"], r["laptop_status"]), 0) + 1
    md += [f"- server `{a}` / laptop `{b}`: {n}" for (a, b), n in sorted(pairs.items())] or ["- none"]
    md += ["", "Other files that differ (first 40):", ""]
    md += [f"- `{r['file']}` ({r['where']})" for r in rows if r["kind"] == "FILE"][:40] or ["- none"]
    (out / "COMPARE_SERVER_LAPTOP.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten: {out / 'COMPARE_SERVER_LAPTOP.md'} and .csv")


if __name__ == "__main__":
    main()
