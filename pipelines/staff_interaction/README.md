# Staff-customer interaction pipeline (Task 3)

Average number of customer interaction sessions per staff member, using
**YOLO pose + IoU/ReID tracking for identity**, and two independent
judgments that each have a default (non-VLM) and an opt-in VLM
implementation:

| Judgment | Default | Opt-in |
|---|---|---|
| staff vs customer, per track | `--role-method reid` — cosine similarity against a small ReID gallery enrolled once via `staff_gui.py` | `--role-method vlm` — Qwen2-VL-2B asked "is this a staff apron?" per crop |
| is this pair actively interacting | `--interaction-method rule` — weighted proximity + mutual-facing score from pose keypoints | `--interaction-method vlm` — Qwen2-VL-2B asked "are these two people interacting?" per pair |

Both role implementations feed the same sticky majority-vote latch; both
interaction implementations feed the same open/gap/cooldown session state
machine (`InteractionSession`). Self-contained: its own tracker, own
scoring, no imports from any sibling pipeline.

## Why the defaults are reid/rule, not VLM

Both VLM implementations were tried first and kept in the codebase
(`--role-method vlm`, `--interaction-method vlm`), but neither is the
default, because of two rounds of the same pattern on real footage:

- **Role.** A VLM prompt asking generically "is this an apron?" matched
  customers' bags, jackets, and patterned clothing (false positives). A
  prompt tightened to rule those out then rejected real staff whose
  uniform didn't match the description closely enough (false negatives).
  Because the role vote is *sticky* (a track that latches "staff" stays
  staff for the rest of the clip, per the brief), a false positive isn't
  a one-frame mistake — it's a phantom row in the summary table for the
  rest of the video, and it steals credit for sessions it was never part
  of. Neither prompt direction eliminated both failure modes at once.
- **Interaction.** The same shape of problem one level up: a real
  exchange (staff bent down talking to a seated customer, an unusual
  camera angle) sometimes read as "not interacting" to the VLM, while a
  customer and staff member merely standing near each other sometimes
  read as "interacting."
- Both VLM calls are also the slowest part of the pipeline by a wide
  margin (roughly 0.5-2s per call on Apple Silicon MPS, vs ~10ms for a
  pose+IoU frame), which matters when tuning thresholds means re-running
  the whole clip repeatedly.

The ReID-gallery + rule-based defaults trade a few seconds of one-time
manual enrolment (`staff_gui.py`) for classifiers that are:

- **Deterministic and narrower.** Role becomes "does this track's
  appearance match one of *these* specific people I was shown are staff"
  rather than "does this look like an apron in general" — a much
  better-conditioned question, and the brief explicitly allows this
  ("the solution may use visual appearance ... or any other reasonable
  approach").
- **Essentially free at inference time.** The tracker already computes a
  running ReID embedding for every track (for identity continuity itself)
  and pose keypoints (for tracking), so role is one cosine similarity
  against a handful of gallery vectors, and interaction is arithmetic on
  keypoints already in memory — no extra model call per frame/pair.
- **Tunable via named thresholds** (`--role-sim-threshold`, `--near-bh`,
  `--face-deg`, `--score-threshold`, ...) instead of English prompt
  wording, which is faster to iterate on and easier to reason about /
  explain than "the model said no."

Neither approach dominates the other in every case, which is why both
stay selectable per run rather than one replacing the other outright.

## Setup

**Role = reid (default): one-time enrolment.**

```bash
python3 pipelines/staff_interaction/staff_gui.py --video raw_videos/entrance.mp4
```

Scrub to a frame, click each staff member's body once (a plain click, not
a drag) to enrol a ReID embedding. Click the same person again on a
different frame/pose for a more robust gallery entry. Saves to
`pipelines/configs/staff_marks.json`, keyed by video filename.

**Role = vlm, or interaction = vlm: no setup step**, but first run
downloads `Qwen/Qwen2-VL-2B-Instruct` (~4-5GB) via `transformers`;
subsequent runs load from the local HF cache.

```bash
python3 pipelines/staff_interaction/staff_interaction_pipeline.py --video raw_videos/entrance.mp4 --preview
```

Useful flags:

```bash
--role-method {reid,vlm}          # default reid
--interaction-method {rule,vlm}   # default rule

--stride 2
--max-frames 0
--no-video
--pose-weights /path/to/pose.pt
--pose-imgsz 1280

# role: reid (default)
--role-sim-threshold 0.62   # cosine similarity to latch staff
--role-reid-min-votes 3
--role-reid-min-ratio 0.55
--role-reid-min-checks 5

# role: vlm (opt-in)
--vlm-model Qwen/Qwen2-VL-2B-Instruct
--vlm-max-side 448
--staff-description ""      # e.g. "a black apron with a small round logo on the chest"
--role-vlm-min-votes 2
--role-vlm-min-ratio 0.60
--role-vlm-max-checks 4
--role-vlm-query-cooldown-s 1.0
--role-vlm-recheck-cooldown-s 6.0

# interaction: rule (default)
--near-bh 1.45          # hard proximity gate, body-heights
--very-near-bh 0.75     # this close, skip the facing requirement
--face-deg 75.0         # facing-cone half-angle
--w-prox 0.55
--w-face 0.45
--score-threshold 0.50
--open-s 1.00
--gap-close-s 1.50

# interaction: vlm (opt-in)
--far-bh 2.60
--query-cooldown-s 2.50
--retry-cooldown-s 1.00
--open-confirmations 1
--close-confirmations 2

# interaction: shared
--min-event-s 1.20
--cooldown-s 3.00
--min-hits 4

--no-reid                  # tracker identity continuity only -- incompatible with --role-method reid
--reid-model /path/to/reid.onnx
--max-missed 45
--retired-ttl-frames 90
```

## Design

### Stage 1 — staff vs customer, per track

Every live track is scored on its own, before any pairing happens, so
this pipeline only ever scores (staff, customer) pairs in stage 2 — never
customer-customer or staff-staff pairs.

**`reid` (default, `scoring.py::update_role_reid`):** every frame, cosine
similarity between the track's running appearance embedding (already
maintained by `tracker.PersonTracker`) and the closest vector in the
enrolled gallery. A frame above `--role-sim-threshold` is a "vote"; a
track latches `staff` once it has at least `--role-reid-min-checks`
checks, `--role-reid-min-votes` of them a vote, and the vote ratio clears
`--role-reid-min-ratio`. No cooldown needed (it's not a model call), so
noise from a single bad-angle frame is naturally outvoted by the many
other frames a track is visible for.

**`vlm` (opt-in, `scoring.py::update_role`):** the VLM is shown a
single-person *torso* crop (shoulders-to-hips, from pose keypoints, with a
fallback to the full box if the pose looks degenerate) and asked once per
`--role-vlm-query-cooldown-s`, up to `--role-vlm-max-checks` times per
*window*. A track that exhausts a window without latching staff gets a
fresh window after `--role-vlm-recheck-cooldown-s` rather than being
locked out as "customer" forever from one bad window.

Both methods use the same sticky majority-vote latch: a track only flips
to `staff` once and never flips back, matching the brief's *"a staff
member should be treated as the same staff instance for as long as they
remain within the camera view."* If they leave view and re-enter, the
tracker (own IoU + optional-ReID, `retired_ttl_frames` kept deliberately
short) assigns a new track ID and role classification starts fresh — the
brief explicitly allows this: *"re-identification across separate
appearances is optional."*

### Stage 2 — is a (staff, customer) pair actively interacting?

**`rule` (default, `scoring.py::rule_pair_cues`):** a per-frame score from
pose keypoints already extracted for tracking, no model call:

- **proximity** — foot-point distance, normalised by average body height
  (roughly scale-invariant near/far in frame), 1.0 at zero distance down
  to 0.0 at `--near-bh`. Beyond `--near-bh` the pair can't be "engaged" at
  all, full stop.
- **facing** — how directly each person's `attention_vector()` (torso
  direction refined by head yaw) points at the other's foot point,
  averaged over whichever of the two has a usable vector; neutral (0.5)
  if *neither* does, so a missing pose cue doesn't by itself veto an
  otherwise very close, sustained pair. Skipped entirely (forced to 1.0)
  inside `--very-near-bh` — at that range (handing something over,
  looking at the same small object together) requiring a clean facing
  angle produced false negatives on real exchanges.
- blended `--w-prox` / `--w-face`; a pair counts as "engaged" this frame
  if the blend clears `--score-threshold`.

**`vlm` (opt-in, `scoring.py::InteractionTracker`):** a pair becomes
query-eligible once within `--near-bh`, re-queried at most every
`--query-cooldown-s` while close; the VLM is shown the union crop of both
boxes and asked "are these two people actively interacting?"

### Session state machine — same shape either way, and same shape as every other pipeline in this repo

- **Open** after sustained engagement (`--open-s` of continuous rule
  "engaged" frames, or `--open-confirmations` consecutive VLM "yes"
  answers).
- **Close** after a disengaged gap (`--gap-close-s`, or
  `--close-confirmations` consecutive VLM "no" answers) — absorbs a brief
  bad read without ending a real, ongoing conversation.
- **Force-close immediately** on physical separation (rule: leaving
  `--near-bh`; vlm: exceeding `--far-bh`) — a cheap distance check, no
  model call needed. The session's end time is taken as *now* in this
  case (not the last confirmed engaged tick, which could be stale) since
  separation is directly observed evidence of the true end.
- **`--min-event-s`** filters out a session that opened and then
  immediately force-closed from being counted as a real session.
- **`--cooldown-s`** after a close: that exact (staff, customer) pair
  cannot open a new session again until this elapses. This is what makes
  a customer who leaves and later returns to the *same* staff member count
  as a **separate session**, per the brief: *"If the interaction ends and
  the customer later returns to interact with the same staff member
  again, the later interaction should be counted as a separate session."*

Sessions are keyed on `(staff_track_id, customer_track_id)`, and the
required metric counts **sessions**, not unique customers — repeated
engagements with different customers, or the same customer at different
times, each add to that staff member's session count, matching the brief.

### Zero-interaction staff are still counted

`register_staff()` is called the moment a track is classified as staff,
independent of whether any interaction ever opens — so a staff member who
never engages a customer in the clip still appears in the summary at 0
sessions and is included in the average, per *"All detected staff
instances should be included when calculating the average number of
interactions, including staff members with zero detected interactions."*

## Outputs (`pipelines/outputs/staff_interaction/`)

- `staff_interaction_summary.csv` — one row per staff instance
  (`staff_instance, track_id, interaction_sessions, first_frame,
  last_frame`), plus `staff_count`, `total_sessions`, and
  `average_sessions_per_staff` at the bottom.
- `staff_interaction_sessions.csv` — one row per counted session
  (`staff_id, customer_id, start_frame, end_frame, duration_s, reason`).
  `reason` is one of `gap`/`no-longer-engaged` (engagement lapsed),
  `separated` (physical separation), `lost` (a track left view),
  `end` (video ended while open).
- `staff_interaction_annotated.mp4` — boxes colored by role (green =
  staff, gray = customer) with the running role-vote ratio under customer
  boxes, a line between any candidate pair (bright + thicker while a
  session is open; the live rule score or VLM answer printed at the
  midpoint), live session duration, and a running staff/session count in
  the HUD.

## Known limitations / what to check first if counts look wrong

- **Gallery quality dominates the `reid` role method.** If a track is
  getting misclassified either way, first check `pipelines/configs/staff_marks.json`
  for that video — a stray click on the wrong person, or too few
  embeddings for someone in an unusual pose, is a far more likely cause
  than the thresholds. Re-run `staff_gui.py`, click a couple more times
  per staff member across different frames, and re-save.
- `--role-sim-threshold` is the main dial: too low pulls in customers with
  similar clothing colours; too high makes a real staff member need a
  near-perfect angle to be recognised at all.
- `--near-bh` / `--very-near-bh` / `--face-deg` are the main dials for the
  rule interaction method — if real interactions are being missed, check
  the annotated video for the printed score at the pair's midpoint (a
  score just under `--score-threshold` on approach means loosen one of
  these; no line drawn at all means they never even got within
  `--near-bh`).
- If a staff member is briefly fully occluded by a customer during a
  conversation, `--max-missed` needs to comfortably exceed the occlusion
  length or the pair's session will be force-closed early via
  `close_all_for` when the track is dropped.
- Switching `--role-method`/`--interaction-method` to `vlm` is a useful
  cross-check on a clip where the rule/reid defaults look wrong, even
  though it isn't the default — the two approaches fail differently, so
  agreement between them on a given session is a good sign either way.
