# Pipelines

Modular, independently-runnable pipelines. Each task in the brief gets its
own pipeline (own detector/tracker choices, own outputs) rather than one
monolithic script that tries to do everything at once. Pipelines don't call
into each other and don't share Python state — they hand data off through
files, so any of them can be run, re-run, or swapped out on its own. They
are only ever combined afterwards, by joining their output CSVs (e.g. on
`track_id` / timestamp) or by watching their annotated videos side by side.

```
pipelines/
    configs/
    store_boundary_zones.json <- written by boundary_gui.py (store outside/inside/entrance)
    shelf_zones.json          <- written by shelf_gui.py (named shelf polygons/front-lines)
    shelf_faces.json          <- written by shelf_face_gui.py (shelf edge + outward normal + customer zone)
    staff_marks.json          <- written by staff_gui.py: a small ReID embedding gallery for
                                  the default (non-VLM) staff role classifier -- see section 5
  boundary/
    boundary_store.py         <- store boundary dataclass + load/save
    boundary_gui.py           <- GUI for outside/inside/entrance
    shelf_store.py            <- shelf layout dataclass + load/save
    shelf_gui.py              <- GUI for shelf polygons/front-lines
  interest/
    tracker.py                <- plain IoU tracker (no ReID — see below)
    snn_motion.py             <- per-person spiking (LIF) motion sensor
    scoring.py                <- the interest criteria + weighted score
    interest_pipeline.py      <- runnable: YOLO-pose + SNN motion -> interest CSV/video
  shelf_interest/
    tracker.py                <- IoU tracker for interior shelf interactions
    scoring.py                <- per-shelf assignment + sustained episode logic
    shelf_interest_pipeline.py <- runnable: per-shelf interest CSV/video
  shelf_vector_interest/
    tracker.py                <- own IoU+ReID tracker (self-contained, not shared with shelf_interest)
    scoring.py                <- zone-gate + fixed-normal facing check + open/close state machine
    shelf_face_store.py       <- shelf-face dataclass (edge/normal/zone) + load/save
    shelf_face_gui.py         <- GUI for edge -> normal -> customer zone, per shelf face
    shelf_vector_pipeline.py  <- runnable: per-shelf-face interest CSV/video
  staff_interaction/
    tracker.py                <- own IoU+ReID tracker (self-contained, not shared with other pipelines)
    scoring.py                <- role (reid default / vlm opt-in) + interaction (rule default / vlm opt-in)
                                  + shared open/gap/cooldown session state machine
    staff_store.py             <- ReID gallery dataclass + load/save (pipelines/configs/staff_marks.json)
    staff_gui.py               <- GUI: click each staff member once to enrol their ReID embedding
    vlm_judge.py               <- Qwen2-VL-2B-Instruct wrapper, only loaded if --role-method/--interaction-method vlm
    staff_interaction_pipeline.py <- runnable: per-staff interaction sessions CSV/video
  outputs/
    interest/                 <- this pipeline's own outputs, namespaced
    shelf_interest/           <- per-shelf interest outputs
    shelf_vector_interest/    <- shelf-vector pipeline's own outputs
    staff_interaction/        <- staff-customer interaction outputs
```

## 1. Mark the boundary (run once per video)

```
python3 pipelines/boundary/boundary_gui.py --video raw_videos/entrance.mp4
```

Click:
- **outside** — the public walkway / walking area in front of the shop.
  Interest is only ever judged for a person while they are in this zone,
  because the brief's "passers-by" are people out on the walkway, not
  people already inside.
- **inside** — the interior of the shop. Staying inside this polygon for
  `entered_dwell_s` is what marks a person as "entered" rather than merely
  "interested".
- **entrance line** (optional but recommended) — 2 points across the
  doorway/threshold. This becomes the concrete point every pipeline treats
  as "the storefront" when computing look-direction and approach distance.
  If skipped, pipelines fall back to the centroid of `inside`.
This writes `pipelines/configs/store_boundary_zones.json`, keyed by video
filename, e.g.:

```json
{
  "entrance.mp4": {
    "outside": [[x, y], ...],
    "inside": [[x, y], ...],
    "entrance_line": [[x1, y1], [x2, y2]]
  }
}
```

Every other pipeline that needs store geometry reads this file. Re-running
the GUI and re-saving simply overwrites the entry for
that video; nothing downstream needs to change.

### Shelf geometry (separate file)

```
python3 pipelines/boundary/shelf_gui.py --video raw_videos/interior.mp4
```

Click one polygon per shelf (3+ points), and optionally draw a 2-point
customer-facing front-line per shelf.

This writes `pipelines/configs/shelf_zones.json`, keyed by video filename.

## 2. Interest pipeline (SNN motion + YOLO posture, no ReID)

```
python3 pipelines/interest/interest_pipeline.py --video raw_videos/entrance.mp4 --preview
```

Per the task instructions, this pipeline is scoped to exactly two model
families: **movement** (a small spiking neural network) and **YOLO pose**
(posture/orientation). It has no dependency on the ReID embedder or gallery
used elsewhere in this repo — a plain IoU tracker is enough for the
short-lived tracks on the walkway, so this pipeline can be run completely
on its own, on any clip, once its boundary is drawn.

Pipeline stages:

```
frame -> YOLOv8n-pose (per-person box + 17 keypoints)
      -> IoUTracker    (position/box history per track_id, no appearance model)
      -> SNNMotionTracker (per-track LIF neuron pool over the person's crop:
                            DVS-style ON/OFF events -> leaky integrate-and-fire
                            -> motion_energy, motion_trend)
      -> scoring.score_track (4 cues -> weighted score -> EMA + sustain)
      -> scoring.update_track_state (interested / entered decision)
```

### Why an SNN for the movement/speed cue

`snn_motion.py` reuses the same DVS-simulation + LIF-neuron idea already
prototyped at the project root in `video_snn.py`, but scopes it from "one
neuron per pixel of the whole frame" down to a small (10x10) neuron grid
over *one tracked person's crop*, so every track gets its own independent
membrane state:

- The leaky membrane (`V[t] = V[t-1]*k + s(t)`, fire+reset at threshold)
  makes it a short-term integrator of gait/limb motion: a stride's stance
  phase doesn't immediately read as "stopped", but a sustained drop in
  motion energy does — which is the actual "slowing down" signal, read from
  body motion rather than only from bounding-box centroid displacement.
- It naturally ignores sub-threshold jitter from a slightly noisy
  detection box, the same way a real DVS pixel ignores sub-threshold
  brightness noise.
- It's cheap: one frame-difference + one threshold compare per person per
  frame.

The plain foot-point displacement (from pose keypoints) is still used for
direction (which way is this person walking / are they closing on the
entrance) since that needs an actual vector, not just an energy scalar. The
SNN's `motion_trend` and the foot-point speed are blended for the
"slowing down" cue (see `scoring.py` docstring for the exact reasoning and
weights).

### Interest criteria (see `scoring.py` for the full reasoning)

Four independent, observable cues, combined with fixed weights, each capped
to `[0, 1]`:

| cue        | source                              | brief language it captures                  |
|------------|-------------------------------------|----------------------------------------------|
| `orient`   | YOLO-pose torso + head vector vs. entrance | "looking toward the storefront"        |
| `turn`     | change in orient-angle over ~0.6s   | "turning their head or body toward it"       |
| `slow`     | SNN motion-energy trend + foot speed| "slowing down"                                |
| `approach` | closing distance to the entrance    | "approaching the entrance"                    |

A track is marked **interested** only once the combined, EMA-smoothed score
clears a threshold *and holds* for `sustain_s` (0.8s) — deliberately not
decided by any single frame, and deliberately not decided by "stopped"
alone (stopping isn't even one of the weighted cues; a stopped-but-facing-
away person scores low, a slowing-and-turning-toward person scores high
even if they never fully stop).

### All thresholds/parameters, in one place (`scoring.py::InterestParams`)

There is no single "shoulder tilt" threshold — orientation comes from the
torso-perpendicular + head-yaw *attention vector* in `src/analytics/pose.py`
(`PoseDet.facing_vector()` / `attention_vector()`), gated by a keypoint
confidence cutoff, and is only turned into a cue by comparing its angle to
the entrance direction against `attend_deg` below. There's one place that
looks at raw speed magnitude (`walk_bh`/`slow_bh`), one for the SNN's
motion-decay signal, and one for closing speed. All distances/speeds are in
**body-heights/second** (bbox height as the unit), not pixels or m/s, since
this pipeline has no floor-plane calibration.

| parameter | value | meaning |
|---|---|---|
| `KP_CONF` (`pose.py`) | 0.30 | below this a keypoint (shoulder/hip/eye/etc.) is treated as missing, not wrong |
| `attend_deg` | 55° | attention-vector-to-entrance angle cone that counts as "looking at the shop"; cue saturates at 0° |
| `turn_deg` | 12° | angle swing *toward* the shop over `turn_window_s` that saturates the "turning toward it" cue |
| `turn_window_s` | 0.6s | window the turning cue compares "now" against |
| `walk_bh` | 1.40 bh/s | normal/unremarkable walking pace — at or above this the speed-based half of `slow` is 0 |
| `slow_bh` | 0.45 bh/s | at/below this the speed-based half of `slow` is fully saturated |
| `motion_trend_ref` | 0.05 | SNN motion-energy drop that fully saturates the trend-based half of `slow` |
| `speed_window_s` | 0.5s | window for computing foot-point speed / approach |
| `approach_ref_bh` | 0.35 bh/s | closing speed toward the entrance that saturates the `approach` cue |
| `w_orient / w_turn / w_slow / w_approach` | 0.35 / 0.15 / 0.25 / 0.25 | cue weights (sum to 1.0) |
| `score_threshold` | 0.55 | EMA score must clear this |
| `sustain_s` | 0.8s | ...and hold above threshold for this long, continuously |
| `ema` | 0.60 | smoothing factor on the interest score (higher = smoother/slower to react) |
| `entered_dwell_s` | 0.8s | continuous time inside the shop polygon before a track is marked "entered" |
| `min_hits` | 5 | tracks with fewer detections than this are treated as flicker, not counted |

Tune these in `InterestParams` (or pass your own instance into
`interest_pipeline.py`) the same way the constants in the original
`src/analytics/events.py` were tuned against labelled clips.

### Fixes: outside-only tracking, and not re-flagging a leaving customer

Two issues from the first pass are fixed:

1. **People already inside the shop showed up as tracked boxes.** YOLO-pose
   still runs on the full frame (it has to, so a person can be followed as
   they cross into the shop), but `IoUTracker` now takes a
   `spawn_predicate`: a **brand-new** track id is only allowed to start on a
   detection inside the `outside` polygon. Someone who only ever appears
   inside (staff, existing shoppers) never gets tracked or drawn by this
   pipeline at all. Boxes are also only *drawn* for a track while it is
   currently in the outside zone — once someone is inside, their box
   disappears from the preview/annotated video (they're still tracked
   internally, just not rendered, since this pipeline's job is the walkway).
2. **A person leaving the shop must not be counted as newly interested.**
   Because a track that starts outside keeps its id as it walks inside
   (`spawn_predicate` doesn't touch already-existing tracks — they keep
   matching frame-to-frame via IoU, and `max_missed` was raised to 45 frames
   so the id survives the walk to the doorway and back into camera view),
   `scoring.update_track_state` now only evaluates/updates the interest
   score `while not track.entered`. Once a track is marked `entered=True`
   (dwelled inside for `entered_dwell_s`), its interest score is frozen —
   if they walk back out and slow down/look around near the doorway on
   their way out, that no longer flips them into "interested". In the
   annotated video such a track is labelled `LEFT STORE` (not `INTERESTED`)
   when it reappears outside.

Caveat: this still relies on the tracker never losing the identity while
the person is inside (there's no ReID fallback in this pipeline by design —
see the note above). If someone is out of camera view or occluded inside
for longer than `max_missed` frames, they'll reappear as a new, un-entered
track and could be re-scored. Raise `max_missed` further, or add a short
entrance-line-proximity re-linking heuristic, if that turns out to matter
for your footage.

### Running on the best available accelerator

`src/analytics/pose.py::PoseDetector` now auto-selects
CUDA → MPS (Apple Silicon) → CPU via `best_device()`, and prints which one
it picked (e.g. `PoseDetector device: mps`). Pass `device="cpu"` explicitly
to `PoseDetector()` to override.

### Outputs

Written to `pipelines/outputs/interest/`:
- `interest_summary.csv` — Total Interested / Interested Entered / Interested Passed By
- `interest_track_log.csv` — per-track_id interest/entered decisions
- `interest_annotated.mp4` — boxes, interest bar, motion energy/trend, HUD
- `interest_cues.csv` (with `--dump-cues`) — per-frame cue breakdown for tuning

## 3. Shelf-interest pipeline (interior, per-shelf episodes)

```
python3 pipelines/shelf_interest/shelf_interest_pipeline.py --video raw_videos/interior.mp4 --preview
```

This pipeline uses YOLO pose + shelf geometry (from `shelf_gui.py`) with
optional ReID continuity, not a VLM. It assigns each interior customer to the
most likely shelf using orientation + proximity, applies hysteresis to avoid
rapid shelf flips, and opens/closes interaction episodes using sustain/gap
timing so a continuous interaction is counted once and a later return is
counted as a new event.

Written to `pipelines/outputs/shelf_interest/`:
- `shelf_interest_summary.csv` — cumulative events per shelf
- `shelf_interest_events.csv` — start/end/duration for each shelf event
- `shelf_interest_track_log.csv` — per-track compact event counts
- `shelf_interest_annotated.mp4` — shelf association, event duration, shelf totals

## 4. Shelf-vector interest pipeline (interior, explicit shelf-face geometry)

```
python3 pipelines/shelf_vector_interest/shelf_face_gui.py --video raw_videos/interior.mp4
python3 pipelines/shelf_vector_interest/shelf_vector_pipeline.py --video raw_videos/interior.mp4 --preview
```

A second, independent take on the same per-shelf interest task, using
explicit geometry instead of a per-frame distance heuristic: each shelf
face is marked once with a fixed outward **normal** vector and a **zone**
polygon (via `shelf_face_gui.py`, into `pipelines/configs/shelf_faces.json`).
A customer engages a face iff their foot point is inside its zone *and*
their YOLO-pose-derived facing vector points back toward the shelf (within
`--face-deg` of `-normal`). When someone stands where two shelves' zones
overlap, whichever face's normal is most opposed to their facing vector
wins — resolving "which shelf are they looking at" directly from the
geometry, without a separate distance-based tie-break.

Self-contained: has its own `tracker.py` (IoU + optional ReID) and does not
import from `shelf_interest`. See `pipelines/shelf_vector_interest/README.md`
for the full design rationale, anti-double-counting rules, and pose-model
recommendations (YOLOv8x-pose/YOLO11-pose as a drop-in upgrade, RTMPose as
a higher-effort option for oblique/occluded views).

Written to `pipelines/outputs/shelf_vector_interest/`:
- `shelf_vector_summary.csv` — cumulative events per shelf face
- `shelf_vector_events.csv` — start/end/duration for each shelf-face event
- `shelf_vector_annotated.mp4` — face edge/normal/zone overlay, live duration, per-shelf totals

## 5. Staff-customer interaction pipeline (Task 3, entrance.mp4)

```
# one-time setup: click each staff member once (a couple of clicks per
# person, on different frames, gives a more robust gallery)
python3 pipelines/staff_interaction/staff_gui.py --video raw_videos/entrance.mp4

python3 pipelines/staff_interaction/staff_interaction_pipeline.py --video raw_videos/entrance.mp4 --preview
```

Average number of customer interaction sessions per staff member, using
YOLO pose + IoU/ReID tracking for identity. Two independent judgments,
each with a default (non-VLM) and an opt-in VLM implementation -- see
`scoring.py`'s module docstring for the full history of why the defaults
ended up here:

  - `--role-method reid` (default): is this track staff? Cosine similarity
    between the track's running ReID embedding (already computed for
    tracking) and a small gallery enrolled once via `staff_gui.py`.
  - `--role-method vlm` (opt-in, `pipelines/staff_interaction/vlm_judge.py`):
    a VLM asked "is this a staff apron?" per track crop. No setup step,
    but slower and, on this footage, not more reliable than the enrolled
    gallery -- prompt-tuning it to stop matching customers' bags/jackets
    tended to also start rejecting real staff, and vice versa.
  - `--interaction-method rule` (default): a weighted proximity +
    mutual-facing score computed directly from the pose keypoints already
    extracted for tracking -- no model call. See `rule_pair_cues` in
    `scoring.py`.
  - `--interaction-method vlm` (opt-in): a VLM asked "are these two people
    interacting?" per pair, throttled by a cooldown while they stay close.

Both role methods drive the same sticky majority-vote latch (a track that
votes staff stays staff for as long as it remains in view, matching the
brief's "same staff instance for as long as they remain within the camera
view"). Both interaction methods drive the same open/gap/cooldown session
state machine: a pair opens a session after sustained engagement, closes
after a gap (or an immediate force-close once they physically separate
beyond `--near-bh`/`--far-bh`), `--min-event-s` drops sessions that opened
and immediately closed, and `--cooldown-s` after a close is what makes
"customer leaves and later returns to the same staff member" count as a
*separate* session per the brief.

**Why ReID-gallery + rule-based scoring are the defaults, not the VLM.**
Two rounds of VLM prompt engineering on the role question each fixed one
failure mode by making the other worse: a stricter "is this an apron"
prompt that stopped matching customers' bags/jackets also started
rejecting real staff whose uniform didn't look exactly like the
description, and a looser prompt did the reverse -- there wasn't a single
wording that was precise for *this specific footage's* uniform without
either false-positive or false-negative failures. The interaction VLM had
a mirror problem: a real exchange (e.g. staff bent down talking to a
seated customer) sometimes read as "not interacting" because the framing
didn't match the VLM's idea of "actively interacting" in general. Both
were also the slowest part of the pipeline by a wide margin.

Trading a few seconds of one-time manual enrolment (`staff_gui.py`) for a
deterministic classifier sidesteps this: role becomes "does this track's
appearance match one of *these* specific people" (a narrower, better
question than "does this look like an apron in general" or "is this pixel
apron-coloured"), and interaction becomes arithmetic on keypoints already
computed for tracking, tunable via named thresholds (`--near-bh`,
`--face-deg`, `--score-threshold`, ...) instead of English prompt wording.
Neither approach is strictly better in every case, which is why both stay
in the codebase, selectable per run.

Written to `pipelines/outputs/staff_interaction/`:
- `staff_interaction_summary.csv` — per staff instance: sessions, first/last
  frame, plus `staff_count` / `total_sessions` / `average_sessions_per_staff`
  (every detected staff instance is included, even at 0 sessions)
- `staff_interaction_sessions.csv` — one row per counted session
- `staff_interaction_annotated.mp4` — role-colored boxes, pair engagement
  lines/scores, live session duration

## Adding more pipelines

Each new pipeline (e.g. dwell-time, ReID-based re-entry) should follow the
same shape: its own subfolder under `pipelines/`, reading geometry configs
(`store_boundary_zones.json`, `shelf_zones.json`, `shelf_faces.json`) as
needed, writing to its own `pipelines/outputs/<name>/`, and staying free of
imports from sibling pipelines. Combine results after the fact.
