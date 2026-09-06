# Hendricks Retail Video Analytic

Each task is its own pipeline. Annotate once with the GUIs, then run one pipeline or `run_all.py`.

See [`pipelines/README.md`](pipelines/README.md) for the approach behind each pipeline (models used, design/tradeoff reasoning, thresholds, and why VLM was dropped in favor of ReID/rule-based scoring) — and each pipeline's own `README.md` (e.g. [`pipelines/interest/README.md`](pipelines/interest/README.md), [`pipelines/shelf_vector_interest/README.md`](pipelines/shelf_vector_interest/README.md), [`pipelines/staff_interaction/README.md`](pipelines/staff_interaction/README.md)) for full detail on that pipeline specifically.

```bash
pip install -r requirements.txt
```

Videos:
- `raw_videos/entrance.mp4` — walkway interest + staff–customer interaction
- `raw_videos/interior.mp4` — per-shelf interest (`shelf_vector_pipeline`)

---

## Annotate (GUIs)

Run these **once per video**. Click **Save** in each GUI.

| GUI | What you mark | Needed for |
|---|---|---|
| Store boundary | outside walkway, inside shop, optional entrance line | interest (entrance) |
| Staff | click each staff member once (ReID gallery) | staff_interaction |
| Shelf faces | edge (2 clicks) → outward normal → customer zone | shelf_vector_interest |

```bash
# entrance.mp4
python3 pipelines/boundary/boundary_gui.py --video raw_videos/entrance.mp4
python3 pipelines/staff_interaction/staff_gui.py --video raw_videos/entrance.mp4

# interior.mp4
python3 pipelines/shelf_vector_interest/shelf_face_gui.py --video raw_videos/interior.mp4
```

Configs land in `configs/`.

---

## Run everything

Interest and staff-interaction are drawn on **one** entrance video. Shelf-vector runs after that on the interior clip.

```bash
python3 run_all.py
python3 run_all.py --preview
python3 run_all.py --skip-shelf          # combined entrance video only
python3 run_all.py --skip-entrance       # shelf-vector only
```

Outputs (project-root `outputs/`):

```
outputs/
  entrance_annotated.mp4      <- exterior: interest + staff on the same video
  interior_annotated.mp4      <- shelf-vector interest
  csv/
    interest/
    staff_interaction/
    shelf_vector_interest/
```

Yes — the exterior/entrance annotated video is `outputs/entrance_annotated.mp4`.

---

## Run one pipeline

```bash
# Walkway interest (entrance)
python3 pipelines/interest/interest_pipeline.py --video raw_videos/entrance.mp4 --preview

# Staff–customer sessions (entrance)
python3 pipelines/staff_interaction/staff_interaction_pipeline.py --video raw_videos/entrance.mp4 --preview

# Per-shelf interest (interior)
python3 pipelines/shelf_vector_interest/shelf_vector_pipeline.py --video raw_videos/interior.mp4 --preview --show-person-vector
```

Drop `--preview` to write files only. More detail is in `pipelines/README.md` and each pipeline's own README.

---

## Docker (CPU only)

The image installs CPU-only PyTorch. GPU is not used even if you pass `--gpus`.

```bash
docker build -t hendricks-retail .

mkdir -p outputs

docker run --rm \
  -v "$(pwd)/raw_videos:/app/raw_videos:ro" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  hendricks-retail
```

Pass `run_all.py` flags after the image name:

```bash
docker run --rm \
  -v "$(pwd)/raw_videos:/app/raw_videos:ro" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  hendricks-retail \
  python3 run_all.py --skip-shelf
```

`raw_videos/` must contain `entrance.mp4` and `interior.mp4`. Mount `models/` read-write so YOLO can download `yolo11x-pose.pt` on first run if it is missing. Annotated videos and CSVs are written to `./outputs` on the host. Scene configs in `configs/` are already copied into the image.
