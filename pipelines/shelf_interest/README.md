# Shelf Interest Pipeline

Per-shelf customer interest using **YOLO pose + shelf geometry + optional ReID continuity**, no VLM.

## Why this approach

For this task, a VLM is optional and usually heavier than needed. The brief asks
for consistent, explainable criteria and clean event counting. This pipeline
implements that directly:

- attention direction from pose (`attention_vector`)
- nearest shelf assignment from drawn shelf polygons/front-lines
- ReID-assisted identity continuity across brief occlusions / crossing
- sustained scoring gates
- continuous interaction merged into one event
- return after a gap counted as a new event

## Setup (once per video)

Draw store boundary first:

```bash
python3 pipelines/boundary/boundary_gui.py --video raw_videos/interior.mp4
```

Then draw shelves:

```bash
python3 pipelines/boundary/shelf_gui.py --video raw_videos/interior.mp4
```

In the shelf GUI:

- add each shelf polygon (3+ points)
- optional: add shelf front-line (2 points) to define customer-facing edge

Saved into:
- `pipelines/configs/store_boundary_zones.json` (store boundary)
- `pipelines/configs/shelf_zones.json` (shelf polygons/front-lines)

## Run

```bash
python3 pipelines/shelf_interest/shelf_interest_pipeline.py --video raw_videos/interior.mp4 --preview
```

Useful flags:

```bash
--stride 2
--max-frames 0
--no-video
--pose-weights /path/to/pose.pt
--pose-imgsz 1280
--no-reid
--reid-model /path/to/reid.onnx
--reid-threshold 0.52
--reid-relink 0.58
--max-missed 90
--retired-ttl-frames 450
```

## Interest definition used

For each tracked person inside the interior zone:

1. Evaluate each shelf:
   - `orient`: is the person oriented toward that shelf
   - `proximity`: how close they are to that shelf boundary/front
   - `dist_bh`: hard body-height-normalized distance to the shelf
   - `slow`: lower movement speed in body-heights/s
   - `turn`: are they turning toward that same shelf
2. Assign one shelf with hysteresis (prevents flicker between adjacent shelves).
3. Apply hard distance constraints (`max_assign_dist_bh`, `max_event_dist_bh`)
   so distant people are not counted as shelf interaction.
4. Open an event once evidence is sustained (`sustain_s`), with a faster
   near-shelf trigger for clear close interactions.
5. Keep the event open while evidence remains; close only after `episode_gap_s`
   of weak/missing evidence.
6. Track only detections strictly inside the shop interior (foot-point and
   bbox center both must be inside).
7. Event counters increase only when a valid-duration event closes; this avoids
   duplicate counting from brief score dips.

That gives the required behavior:

- continuous interaction = one event
- leaving and later returning = new event

## Outputs

Written to `pipelines/outputs/shelf_interest/`:

- `shelf_interest_summary.csv` (`shelf_id,shelf_name,interest_events`)
- `shelf_interest_events.csv` (per event with start/end/duration)
- `shelf_interest_track_log.csv` (per track summary)
- `shelf_interest_annotated.mp4` (unless `--no-video`)
