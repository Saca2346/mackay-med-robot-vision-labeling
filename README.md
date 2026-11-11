# Labelling project: box / size_label / expiry_label (Mackay medicine-box robot)

Photos, labels and tools that build the training data for the vision model (code repo: `medicine_shelf_vision`, after the
folder move `mackay_vision\code`). The lab server (hucenrotia-ai) keeps the master copy of photos, labels and models.

## Content

| Path | What it is |
| --- | --- |
| `raw/` | the labelled photos (196: 23 Sep, 28 Sep, 1 Oct) |
| `label_specs/`, `label_specs_expiry/` | label specifications per photo (boxes + size labels; expiry labels with status `auto` / `selesai`) |
| `labels_final_obb/` | final YOLO-OBB labels: `cls x1 y1 x2 y2 x3 y3 x4 y4` (normalised), classes in `classes.txt` |
| `tools/` | labelling, checking, export, training and packing scripts |
| `server_scripts/` + `RUN_ON_SERVER.bat` | server steps run from the laptop over ssh; outputs in `server_results/` |
| `New image/` | 87 phone photos of 28 Sep; 33 of them (16h) are now reviewed in the dataset |
| `archive/` | old packages and results (`MOVED_FILES_20261014.csv` lists what was moved there) |
| `docs/` | English guides; `docs/id/` Indonesian originals |

Classes (never add one, never use product names as class names):
`0 box` (package spine), `1 size_label` (product name + size cluster), `2 expiry_label` (printed expiry date).

Reviewed server master on 14 Oct (Codex): 196 photos, 1,859 boxes, 2,306 size panels and 1,489 expiry labels; all 196 expiry specs are `selesai`, none remain `auto`. The 33 close-ups have been converted to specs. Geometry/readiness checks passed. Three-class training completed in 70 minutes; the v6 accuracy comparison failed and out-of-fold safety evaluation passed (0 wrong MATCH). The demo model is retained. See `../code/docs/CVAT_EXPIRY_REVIEW_20261014.md`. Laptop labels were synchronised and checked with server SHA-256 hashes; the prior laptop snapshot is retained in `audit_local/cvat_20261014/laptop_before_reviewed_master.tgz`.

## Guides

| Topic | File |
| --- | --- |
| Correcting labels in CVAT / X-AnyLabeling / Label Studio, label rules | `docs/LABELING_TOOLS_GUIDE.md` |
| 3-class training on the server | `docs/TRAINING_3_CLASSES.md` (Indonesian: `PANDUAN_TRAINING_3KELAS.md`) |
| Data plan for the next round | code repo `docs/DATA_COLLECTION_PLAN.md` |
| Server vs laptop differences | `RUN_ON_SERVER.bat langkah7_manifest`, then `python tools\compare_server.py server_results\langkah7_manifest.txt` |

The old README (first prelabel package, 104 photos) is in `docs/id/README_prelabel_104_old.md`.
