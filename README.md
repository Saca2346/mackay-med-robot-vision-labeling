# Labelling project: box / size_label / expiry_label (Mackay medicine-box robot)

Photos, labels and tools that build the training data for the vision model (code repo: `medicine_shelf_vision`, after the
folder move `mackay_vision\code`). The lab server (hucenrotia-ai) keeps the master copy of photos, labels and models.

## Content

| Path | What it is |
| --- | --- |
| `raw/` | the labelled photos (163: 23 Sep, 28 Sep, 1 Oct) |
| `label_specs/`, `label_specs_expiry/` | label specifications per photo (boxes + size labels; expiry labels with status `auto` / `selesai`) |
| `labels_final_obb/` | final YOLO-OBB labels: `cls x1 y1 x2 y2 x3 y3 x4 y4` (normalised), classes in `classes.txt` |
| `tools/` | labelling, checking, export, training and packing scripts |
| `server_scripts/` + `RUN_ON_SERVER.bat` | server steps run from the laptop over ssh; outputs in `server_results/` |
| `New image/` | 87 phone photos of 28 Sep; 33 of them (16h) are not in the dataset yet |
| `archive/` | old packages and results (`MOVED_FILES_20261014.csv` lists what was moved there) |
| `docs/` | English guides; `docs/id/` Indonesian originals |

Classes (never add one, never use product names as class names):
`0 box` (package spine), `1 size_label` (product name + size cluster), `2 expiry_label` (printed expiry date).

Counts on 14 Oct: 1,826 box, 2,277 size_label. Expiry labels after the server merge (14 Oct): 1,444 in all 163 photos,
443 of them in the 51 reviewed (`selesai`) photos; 112 photos still `auto`. The server tightened the date labels and reopened 26
photos with missed dates on 12 Oct (`docs/id/PANDUAN_LANJUTAN_3KELAS.md`); review them with `python tools/review_tanggal.py`.
Laptop and server labels are identical since 14 Oct; the laptop copy before the merge is in `archive/laptop_before_server_merge_20261014/`.

## Guides

| Topic | File |
| --- | --- |
| Correcting labels in CVAT / X-AnyLabeling / Label Studio, label rules | `docs/LABELING_TOOLS_GUIDE.md` |
| 3-class training on the server | `docs/TRAINING_3_CLASSES.md` (Indonesian: `PANDUAN_TRAINING_3KELAS.md`) |
| Data plan for the next round | code repo `docs/DATA_COLLECTION_PLAN.md` |
| Server vs laptop differences | `RUN_ON_SERVER.bat langkah7_manifest`, then `python tools\compare_server.py server_results\langkah7_manifest.txt` |

The old README (first prelabel package, 104 photos) is in `docs/id/README_prelabel_104_old.md`.
