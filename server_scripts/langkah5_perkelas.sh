#!/usr/bin/env bash
# Per-class cross-validation numbers of the 3-class run (cv_results.json) + the final model summary.
cd ~/Downloads/prelabel_size_label_full || exit 1
source ~/trse_venv/bin/activate
python - <<'PY'
import json, statistics as st
r = json.load(open("logs/cv_results.json"))
print("photos", r.get("photos"), "| folds", r.get("folds"), "| imgsz", r.get("imgsz"), "| epochs", r.get("epochs"))
per = {}
for key, run in r["runs"].items():
    for part in ("clean", "degraded"):
        for cls, v in run[part]["per_class"].items():
            per.setdefault((part, cls), []).append(v)
for (part, cls), vs in sorted(per.items()):
    keys = [k for k in vs[0] if isinstance(vs[0][k], (int, float))]
    print(f"{part:9s} {cls:13s} " + "  ".join(f"{k} {st.mean(v[k] for v in vs):.3f}+-{st.pstdev(v[k] for v in vs):.3f}" for k in keys) + f"   (n folds {len(vs)})")
b = json.load(open("logs/best_model.json"))
print("model:", b.get("model"), "| final weights:", b.get("final_weights"))
print("thresholds:", json.dumps(b.get("thresholds")))
PY
ls -la runs/best_final/ | head; sha256sum runs/best_final/box_best.pt
