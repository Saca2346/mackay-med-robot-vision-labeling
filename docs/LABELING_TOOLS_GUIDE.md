# Correcting labels on the Ubuntu server (hucenrotia-ai): CVAT, X-AnyLabeling, Label Studio

All labelling work is done on **Ubuntu (hucenrotia-ai)**, and the project folder there is the **master**.
(Indonesian original: `docs/id/PANDUAN_CVAT_XANYLABELING_LABELSTUDIO.md`.)

In this guide:
- **PROJECT** = `/home/Zen/Downloads/prelabel_size_label_full`, or `~/Downloads/prelabel_size_label_full`.
- Every `python ...` command runs from the PROJECT folder with the venv active:

```
cd ~/Downloads/prelabel_size_label_full
source ~/trse_venv/bin/activate
```

## Remote access

A terminal is easiest through **VS Code Remote-SSH** from the laptop (copy / paste works; see the code repo,
`docs/REMOTE_SERVER_CLAUDE_CODEX.md`). The desktop tools below (X-AnyLabeling) still need the Ubuntu desktop (Remmina).

**Remmina settings.** For labelling through Remmina choose **Quality: Best** and **Color depth: 32 bpp**. At lower quality the
image is compressed and package edges get blurry.

**CVAT and Label Studio are web based.** You can:
- open Firefox / Chrome **on the Ubuntu desktop** through Remmina, or
- use the **laptop browser** through an SSH tunnel (sharper and more responsive), when the laptop can SSH to Ubuntu:
  ```
  ssh -L 8080:localhost:8080 -L 8081:localhost:8081 Zen@hucenrotia-ai
  ```

Important rule: **upload photos and download exports only from the browser on Ubuntu.** That way the files land directly on
Ubuntu (the master). The labelling itself may be done from any browser, because the data is stored on the server.

## Overview of the cycle

1. Prepare a **batch** folder:
   - `batches/cek_manual`: 7 seed photos that need checking, made with `python tools/batch.py cek_manual --review`.
   - `batches/al_round1`: made by `al_select.py`.
2. Open the batch in **one** tool and correct the labels.
3. Export the corrected result from the tool.
4. `python tools/import_reviewed.py <export>` moves the result to `labels_reviewed_obb/`.
5. `python tools/export_datasets.py` builds a new seed dataset.
6. Retrain on the GPU, then `python tools/al_select.py ...` chooses the next batch. Back to step 2.

Every batch folder contains:

| File / folder | Used by |
|---|---|
| `images/` | the batch photos (every tool) |
| `labels/` + `classes.txt` | X-AnyLabeling |
| `cvat_annotations.xml` | CVAT |
| `label_studio_tasks.json` | Label Studio |
| `list.txt` | list of photo names |

Batches are **not copied** from the laptop. Rebuild them on Ubuntu:

```
python tools/batch.py cek_manual --review
```

This batch holds photos 3, 12, 13, 29, 30, 31, 48. The left group that is blurred in photos 29-31 needs the most attention.

**Important rules:**
- One batch is worked on in **one tool only**.
- Class names must be exactly `box` and `size_label` (and `expiry_label` for dates). Their order in the tool may differ, because
  `import_reviewed.py` matches classes by **name**.

---

## A. CVAT (Docker)

### A1. Installation (once)

1. Install Docker Engine + the compose plugin (if missing), then **log out and in again**:
   ```
   sudo apt install docker.io docker-compose-v2
   sudo usermod -aG docker $USER
   ```
2. Start CVAT:
   ```
   git clone https://github.com/cvat-ai/cvat ~/cvat
   cd ~/cvat
   git checkout v2.76.0
   docker compose up -d
   ```
   The first download is several GB. Wait until `docker compose ps` shows every container `running` or `healthy`.
3. Create an admin account:
   ```
   docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
   ```
4. Open `http://localhost:8080`.

**Stop and start** (from `~/cvat`): `docker compose stop` / `docker compose start`. **Never** run `docker compose down -v`,
because `-v` deletes all CVAT data.

### A2. Create the project (once)

1. Click **Projects**, then **+**, then **Create a new project**.
2. Name: `TRSE box size_label`.
3. Add two labels with **Add label**, type **Rectangle**:
   - `box`
   - `size_label`
4. Click **Submit & Open**.

### A3. Create a task for one batch (from the browser on Ubuntu)

1. Inside the project click **+**, then **Create a new task**. Name = the batch name.
2. **Select files -> My computer**, then choose **all** photos from `PROJECT/batches/<batch>/images`.
3. Open **Advanced configuration** and set **Image quality** to `95` or more.
4. Click **Submit & Open**.

### A4. Load the initial labels

1. On the task page click **Actions**, then **Upload annotations**.
2. Import format: **CVAT 1.1**.
3. Choose the file `PROJECT/batches/<batch>/cvat_annotations.xml`, then click **OK**.

### A5. Correcting

**Basic controls:**
- Next / previous photo: **F** / **D**.
- Save: **Ctrl+S** (often).
- Undo: **Ctrl+Z**.

**Changing a box:**
- Click the box, then drag a corner or a side.
- To **rotate**, drag the **rotation handle** (small circle above the box).

**Adding a box:**
1. Choose the **Draw new rectangle** tool.
2. Choose the label, then draw the box.
3. Rotate it with the rotation handle until it is parallel to the package spine.

Press **N** to repeat the last drawing tool.

**Deleting a box:** select the box, then press **Delete**.

### A6. Export (from the browser on Ubuntu)

1. **Actions -> Export task dataset**.
2. Format: **Ultralytics YOLO Oriented Bounding Boxes 1.0**.
3. Turn **Save images** off, then click **OK**.
4. Download the zip from the **Requests** menu. The file is saved in `~/Downloads/`.
5. Bring the result back into the PROJECT:
   ```
   python tools/import_reviewed.py ~/Downloads/<file-name>.zip --dry
   python tools/import_reviewed.py ~/Downloads/<file-name>.zip
   python tools/export_datasets.py
   ```

Do not use "YOLO 1.1" or "Ultralytics YOLO Detection": both drop the rotation. `import_reviewed.py` refuses exports with
upright boxes.

---

## B. X-AnyLabeling (desktop application, through Remmina)

### B1. Installation (once)

X-AnyLabeling needs **Python 3.11-3.13**. Check with `python3 --version` first:
- Ubuntu 24.04 uses Python 3.12, so it works directly.
- Ubuntu 22.04 uses Python 3.10. Use the Linux build from **Releases** (`https://github.com/CVHub520/X-AnyLabeling/releases`),
  or install Python 3.12 through `uv` / conda.

Use a venv separate from `trse_venv`. Run it in a terminal **inside the Remmina desktop**, not through SSH, because the
application needs a screen:

```
python3 -m venv ~/xal-env
source ~/xal-env/bin/activate
pip install -U pip
pip install "x-anylabeling-cvhub[cpu]"
xanylabeling
```

The `[cpu]` version is enough for corrections, because the drafts are already made by our model. `[gpu]` is only needed for B5
(auto-labelling inside X-AnyLabeling).

### B2. Open the batch and the initial labels

1. **File -> Open Dir** (**Ctrl+U**), then choose `PROJECT/batches/<batch>/images`. The window title must show **[1/N]** with
   N = the number of photos in the batch.
2. Make sure **File -> Auto Save** is on.
3. **Upload -> Upload YOLO-Obb Annotations** (not YOLO-Hbb, those are upright boxes):
   - Class file: `PROJECT/batches/<batch>/labels/classes.txt`.
   - Label folder: `PROJECT/batches/<batch>/labels`.
4. One `.json` per photo appears in `images/`; that is X-AnyLabeling's working file and is normal.

### B3. Correcting

- New rotated box: press **O**.
- Rotate: select the box (edit mode), then press **Z / X / C / V**.
- Next / previous photo: **D** / **A**.
- Delete: select the box, then press **Delete**.
- Press **Ctrl+S** on every photo. The time of that photo's `.json` must change (`ls -l images/*.json`).
- **Do not** use **Tools -> Save Visualization Image** while the batch is open: X-AnyLabeling also reads subfolders, so the
  `images_visualizations` folder makes the visualisation pictures count as photos (e.g. [20/20]). If it already happened, move
  that folder out of `images/`.

### B4. Export

1. **Export -> Export YOLO-Obb Annotations**, same class file.
2. The export goes to `PROJECT/batches/<batch>/labels` (it overwrites the drafts; the drafts stay in the `.json` files and
   `labels_auto/`).
3. Bring the result back into the PROJECT:
   ```
   python tools/import_reviewed.py batches/<batch>/labels --dry
   python tools/import_reviewed.py batches/<batch>/labels
   python tools/export_datasets.py
   ```
   `import_reviewed` refuses photos that are **exactly the model draft** (not corrected), and shows per photo how many objects
   are the same / changed / deleted compared with the draft.

### B5. (Optional) Auto-label directly with the bootstrap model on the GPU

1. Convert the model to ONNX:
   ```
   yolo export model=runs/bootstrap_v1/weights/best.pt format=onnx imgsz=1280
   ```
2. Create the file `trse_obb.yaml`:
   ```yaml
   type: yolo11_obb          # yolov8_obb if the model is YOLOv8-OBB
   name: trse_obb
   display_name: TRSE box/size_label
   model_path: /home/Zen/Downloads/prelabel_size_label_full/runs/bootstrap_v1/weights/best.onnx
   conf_threshold: 0.35
   iou_threshold: 0.7
   classes:
     - box
     - size_label
   ```
3. In X-AnyLabeling open the model dropdown, choose **...Load Custom Model**, then choose `trse_obb.yaml`.
4. Press **Ctrl+A** (auto-label every photo), then correct as usual.

---

## C. Label Studio

### C1. Installation (once)

Use a separate venv:

```
python3 -m venv ~/ls-env
source ~/ls-env/bin/activate
pip install label-studio
```

### C2. Starting Label Studio

Every time:

```
source ~/ls-env/bin/activate
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$HOME/Downloads/prelabel_size_label_full
label-studio start --port 8081
```

Port 8081 avoids a clash with CVAT on 8080. Open `http://localhost:8081`, then create a local account.

### C3. Create the project (once)

1. **Create Project**, name it `TRSE box size_label`.
2. **Labeling Setup -> Custom template -> Code**, then paste the content of `PROJECT/export/label_studio/label_config.xml`.
   Click **Save**.
3. Connect the photo folder:
   1. **Settings -> Cloud Storage -> Add Source Storage**.
   2. Storage type: **Local files**.
   3. Absolute local path: `/home/Zen/Downloads/prelabel_size_label_full/raw`.
   4. Click **Add Storage**.
   5. **Do not click "Sync Storage"**: it would create one empty task per photo.

### C4. Load a batch

1. In the project click **Import -> Upload Files**, then choose `PROJECT/batches/<batch>/label_studio_tasks.json`.
2. The result depends on the batch type:
   - **`cek_manual`**: labels arrive as *annotations*. Correct them, then click **Update**.
   - **`al_round`**: model labels arrive as *predictions*. Turn on **Settings -> Annotation -> Use predictions to prelabel tasks**
     (older versions: **Settings -> Machine Learning**). Then correct, and click **Submit**.

### C5. Correcting

- Hotkey **1** = `box`, **2** = `size_label`.
- Drag a corner to resize, the rotation handle to rotate.
- **Backspace** or **Delete** to delete.

### C6. Export (from the browser on Ubuntu)

1. **Export**, choose the **JSON** format (not JSON-MIN, not YOLO), then click **Export**.
2. Bring the result back into the PROJECT:
   ```
   python tools/import_reviewed.py ~/Downloads/project-1-at-....json --dry
   python tools/import_reviewed.py ~/Downloads/project-1-at-....json
   python tools/export_datasets.py
   ```

Label Studio's "YOLOv8 OBB" export has reported problems. JSON keeps the full rotation and is converted correctly by
`import_reviewed.py` (difference under 0.5 px).

---

## D. After correcting

1. **Always run `--dry` first.** The result is one of three:
   - **ERROR:** nothing is written. Fix it in the tool, then export again.
   - **WARNING**, e.g. a `size_label` outside a box or a photo without a box: check that photo again.
   - **Valid:** run without `--dry`. Old files are backed up automatically in `labels_reviewed_obb/_backup/`.
2. `python tools/export_datasets.py`. Reviewed photos move from the pool to the seed.
3. Next round: `bash tools/run_bootstrap.sh 2`. This trains `runs/bootstrap_v2` with the larger seed and builds `batches/al_round2`.

---

## Label rules (the same in every tool)

**`box`**
- Only the package spine, as a rotated box parallel to the spine.
- End faces, side faces, gaps, hands and fingers are not included.
- The box must not go outside the target package.
- A partly hidden package: label only the visible part.
- Thin slivers (less than about 1 % of the photo width) are not labelled.

**`size_label`**
- The cluster of product name + size (diameter / length), always inside its `box`.
- Terumo (Nagomi, Accuforce): 2 `size_label`, i.e. name + size and the TERUMO logo block.
- iVascular: 1 `size_label`, i.e. the coloured panel.
- Sled Bag: Boston Scientific logo + the text "Permanent Sled Bag".
- REF, LOT, date or barcode alone get **no** `size_label`.

**`expiry_label`** (third class, since 11-10)
- The printed expiry date next to the hourglass icon, inside its `box`.

**Never:**
- Create a new class.
- Write text or a product name as a class name.
