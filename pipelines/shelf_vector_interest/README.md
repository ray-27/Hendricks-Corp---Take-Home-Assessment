# Shelf-vector interest pipeline

A second, independent take on the per-shelf customer-interest task, using
**explicit geometry** instead of a per-frame distance heuristic. Does not
modify or import from `pipelines/shelf_interest`; the two pipelines can be
run and compared side by side.

## The idea

`pipelines/shelf_interest` decides "is this person looking at the shelf" by
finding the nearest point on the shelf polygon to the person's current foot
position, every frame. That target point slides as the person walks, so
the facing-angle check inherits some noise from wherever their feet happen
to be.

This pipeline instead asks you to mark, **once per shelf**, the geometry a
human reviewer would actually use:

- **`normal`** — a fixed unit vector, perpendicular to the shelf's front
  edge, pointing outward into the area customers stand in. This is the
  shelf's "line of sight" and never moves.
- **`zone`** — a polygon: the area a customer must be standing in to be
  eligible for this shelf's interest at all. A hard point-in-polygon gate,
  not a soft distance cue.

A person **engages** a shelf face when both are true, every frame:

1. their foot point is inside `zone`, and
2. their facing vector is within `--face-deg` of `-normal` (they're facing
   back toward the shelf).

If a customer is standing where two shelves' zones overlap — exactly the
"between two shelves" case the task brief calls out — whichever face's
normal is most directly opposed to the customer's own facing vector wins.
That is the same evidence a human would use to decide, and it falls out of
the geometry for free instead of needing a separate rule.

## Anti-double-counting

Same three-knob approach as `shelf_interest`, tuned via CLI:

- `--engage-s` (default 1.2s): consecutive engaged frames required before
  an event **opens**.
- `--gap-close-s` (default 1.0s): after opening, a temporary drop in
  engagement (half-turn, one step back) is tolerated for this long before
  the event **closes**. Keeps one continuous visit as one event.
- `--min-event-s` (default 0.8s): a close only counts as an event if it
  lasted at least this long — filters out threshold jitter.
- `--cooldown-s` (default 5.0s): after a (person, shelf) event closes, that
  pair can't open a new event again until this much time has passed —
  distinguishes "still lingering" from "left and came back later".

## Person direction vector: model choice

Uses the same `PoseDetector` (Ultralytics YOLO-pose) as the rest of this
repo, swappable via `--pose-weights`:

- **Default / recommended now**: `yolo11x-pose.pt` (shared `PoseDetector` default)
  for keypoint accuracy at oblique CCTV angles. Nano (`yolov8n-pose.pt`) is
  still a drop-in if you need speed: `--pose-weights yolov8n-pose.pt`. Same
  Ultralytics API; raise `--pose-imgsz` if the camera is wide-angle.
- **If keypoint quality is still the bottleneck**: **RTMPose** (OpenMMLab
  MMPose) is the strongest option for occluded/oblique top-down views, but
  it's a different framework (two-stage: a person detector + a separate
  top-down pose model, via `mmpose`/`mmdet`), not a drop-in weights swap —
  budget for a small adapter that produces the same keypoint layout
  (`det.kpts`, `NOSE`/`L_EAR`/`R_EAR` indices) this pipeline expects.

Direction vector construction (`_facing_vector` in
`shelf_vector_pipeline.py`): head vector (nose − ear midpoint) blended 70/30
with torso orientation, falling back to torso-only when the head keypoints
aren't visible. Torso alone is steadier but lags a quick head turn; the
blend favors head direction (what someone is actually looking at) while
staying stable when they glance rather than fully turn.

## Setup

1. Mark shelf faces (edge → normal → customer zone):
   ```bash
   python3 pipelines/shelf_vector_interest/shelf_face_gui.py --video raw_videos/interior.mp4
   ```
   For each shelf: click the two edge points, click once on the customer
   side to set the outward normal (click again to flip it), then click 3+
   points for the interest zone. Add more faces with **+ Add face**. Save
   writes `pipelines/configs/shelf_faces.json`, independent of every other
   config file in this repo.

2. Run the pipeline:
   ```bash
   python3 pipelines/shelf_vector_interest/shelf_vector_pipeline.py \
       --video raw_videos/interior.mp4 --preview
   ```

## Outputs (`outputs/`)

- `outputs/interior_annotated.mp4` — per-shelf edge/normal/zone overlay, boxes
  colored by assigned shelf, live interaction duration, running per-shelf
  and total event counts, and a line connecting an engaged customer to the
  shelf they're engaging.
- `outputs/csv/shelf_vector_interest/shelf_vector_summary.csv` — `shelf_id, shelf_name, interest_events`.
- `outputs/csv/shelf_vector_interest/shelf_vector_events.csv` — one row per counted event: `track_id,
  shelf_id, shelf_name, start_frame, end_frame, duration_s`.

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--show-person-vector` | off | draw each person's facing vector (green if facing OK, yellow if not) and the expected shelf-facing direction (magenta) |
| `--face-deg` | 55 | max angle between facing vector and `-normal` to count as "facing" |
| `--engage-s` | 1.2 | sustained engaged time to open an event |
| `--gap-close-s` | 1.0 | tolerated disengaged gap before closing |
| `--min-event-s` | 0.8 | minimum duration for a close to be counted |
| `--cooldown-s` | 5.0 | per (person, shelf) cooldown after a close |
| `--pose-weights` / `--pose-imgsz` | ultralytics default / 1280 | pose model + inference size |
| `--reid-model`, `--reid-threshold`, `--reid-relink`, `--reid-weight`, `--max-missed`, `--retired-ttl-frames` | see `--help` | identity persistence tuning (own tracker in `tracker.py`, mirrors `shelf_interest`'s approach) |
| `--no-reid` | off | IoU-only tracking, no appearance embeddings |

## How this compares to `shelf_interest`

| | `shelf_interest` | `shelf_vector_interest` (this pipeline) |
|---|---|---|
| Shelf geometry | polygon + optional front-line | fixed edge + outward normal + explicit customer zone |
| "Close enough" | bounding-box-height-relative distance threshold | hard zone-polygon membership |
| "Facing" | angle to nearest point on shelf polygon (moves with the person) | angle to a fixed per-shelf normal vector |
| Between two shelves | resolved by distance + orientation score | resolved by whichever normal is most opposed to the person's facing vector |
| Config file | `shelf_zones.json` | `shelf_faces.json` |

Both are independent and can be run on the same video to compare event
counts; neither imports from the other.
