# Interest Pipeline

Outside-walkway store-interest pipeline using only YOLO pose + SNN motion.

## What It Does

- Detects people with `YOLOv8n-pose` (`src/analytics/pose.py`).
- Tracks them with IoU tracking (`tracker.py`), no ReID dependency.
- Computes per-person motion cues with a small LIF SNN (`snn_motion.py`).
- Scores "interest" from 4 cues (`scoring.py`):
  - looking toward storefront (`orient`)
  - turning toward storefront (`turn`)
  - slowing down (`slow`)
  - approaching entrance (`approach`)
- Marks conversion (`entered`) when the track dwells inside the shop zone.

## Inputs

- Video file, e.g. `raw_videos/entrance.mp4`
- Boundary config from `pipelines/boundary/boundary_gui.py`:
  - `pipelines/configs/store_boundary_zones.json`

## Run

```bash
python3 pipelines/interest/interest_pipeline.py --video raw_videos/entrance.mp4 --preview
```

Useful flags:

```bash
--stride 2
--dump-cues
--no-video
--snn-grid 14 14
--snn-diff-thresh 10
--snn-vth 0.55
--snn-tau 3.0
```

## Current Decision Thresholds

From `scoring.py` (`InterestParams`):

- `attend_deg=68`
- `turn_deg=9`, `turn_window_s=0.45`
- `walk_bh=1.55`, `slow_bh=0.55`
- `motion_trend_ref=0.035`
- `approach_ref_bh=0.25`
- weights: `orient=0.32`, `turn=0.18`, `slow=0.20`, `approach=0.30`
- `score_threshold=0.47`, `sustain_s=0.45`, `ema=0.45`
- `entered_dwell_s=0.25`, `entered_backfill_score=0.38`
- `min_hits=4`

## Outputs

Written to `pipelines/outputs/interest/`:

- `interest_summary.csv`
- `interest_track_log.csv`
- `interest_annotated.mp4` (unless `--no-video`)
- `interest_cues.csv` (with `--dump-cues`)

## Notes

- New tracks are spawned only in the outside/walkway polygon.
- People who already entered are not re-counted as new interested passers-by when leaving.
- `PoseDetector` auto-uses best device: CUDA -> MPS -> CPU.


NOTE
The entered intrested people are little too much than actual. (try to fix this)