# Hendricks Retail Video Analytic

Each task is its own pipeline. Annotate once with the GUIs, then run one pipeline or `run_all.py`.

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

Configs land in `pipelines/configs/`.

---

## Run everything

Interest and staff-interaction are drawn on **one** entrance video. Shelf-vector runs after that on the interior clip.

```bash
python3 run_all.py
python3 run_all.py --preview
python3 run_all.py --skip-shelf          # combined entrance video only
python3 run_all.py --skip-entrance       # shelf-vector only
```

Outputs:
- `pipelines/outputs/combined/entrance_annotated.mp4` — interest + staff on the same frames
- `pipelines/outputs/interest/` — interest CSVs
- `pipelines/outputs/staff_interaction/` — staff CSVs
- `pipelines/outputs/shelf_vector_interest/` — shelf video + CSVs

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
