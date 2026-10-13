#!/bin/bash
# READ-ONLY: list every label file on the server with a content hash, so the laptop copy can be compared
# (tools/compare_server.py). Nothing on the server is changed.
cd ~/Downloads/prelabel_size_label_full || { echo "folder not found"; exit 1; }
python3 - <<'EOF'
import hashlib, json, pathlib

def sha(p):
    # line endings normalised: a Windows (CRLF) and a Linux (LF) copy of the same text count as equal
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]

for p in sorted(pathlib.Path("label_specs_expiry").glob("*.json")):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"EXP|{p.name}|{d.get('status', '')}|{len(d.get('expiry_label', []))}|{sha(p)}")
    except Exception as e:
        print(f"EXP|{p.name}|unreadable {type(e).__name__}||{sha(p)}")
for folder, pat in (("labels_final_obb", "*.txt"), ("label_specs", "*.json"), ("tools", "*.py"), ("server_scripts", "*.sh"),
                    (".", "*.md"), (".", "*.csv"), (".", "classes.txt")):
    for p in sorted(pathlib.Path(folder).glob(pat)):
        print(f"FILE|{p.as_posix()}|||{sha(p)}")
print("END")
EOF
