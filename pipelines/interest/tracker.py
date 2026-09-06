"""Lightweight IoU tracker for the interest pipeline.

This pipeline intentionally does not depend on the ReID model or embedding
pipeline elsewhere in this repo (`src/reid`). The brief's interest cues
(orientation, turning, slowdown, approach) only need each person's position
and pose over consecutive frames -- they don't need appearance-based
re-identification across a long occlusion. A constant-velocity IoU tracker
is enough for a person crossing the walking area, and keeps this pipeline's
only model dependency to YOLOv8-pose, per the brief for this layer.

`spawn_predicate`
------------------
YOLO-pose runs on the whole frame, which also contains staff and other
people already inside the shop. This pipeline only cares about the walking
area outside, so brand-new tracks are only allowed to start on a detection
that is currently in the "outside" boundary polygon (see
`interest_pipeline.py`, which passes `boundary.in_outside` here). A track
that *started* outside and later walks inside is still matched frame to
frame as normal -- the gate only blocks spawning fresh identities for
people the camera only ever saw inside, e.g. staff or shoppers already in
the store. Keeping the same track id across the outside/inside boundary is
what lets scoring.py freeze interest once a track is marked "entered", so a
customer walking back out to leave is not re-scored as a new interested
passer-by.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import INTEREST_MAX_MISSED, INTEREST_TRACK_MIN_HITS, TRACK_MATCH_IOU  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-6)


@dataclass
class Track:
    track_id: int
    det: object  # PoseDet
    box: tuple[int, int, int, int]
    foot_px: np.ndarray
    first_frame: int
    last_frame: int
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hits: int = 1
    missed: int = 0
    # (frame_idx, foot_px, box_height) -- everything scoring.py needs for speed
    history: deque = field(default_factory=lambda: deque(maxlen=150))
    # (frame_idx, angle_to_shop_deg) -- for the "turning toward" cue
    angle_hist: deque = field(default_factory=lambda: deque(maxlen=150))

    # --- interest state (mirrors the definition in scoring.py) --------------
    interest_ema: float = 0.0
    interest_run: int = 0
    interested: bool = False
    interest_frame: int | None = None
    best_interest: float = 0.0
    inside_run: int = 0
    entered: bool = False
    seen_outside: bool = False

    def predicted_box(self) -> tuple[int, int, int, int]:
        dx, dy = float(self.vel[0]) * self.missed, float(self.vel[1]) * self.missed
        x1, y1, x2, y2 = self.box
        return int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)


class IoUTracker:
    def __init__(
        self,
        match_iou: float = TRACK_MATCH_IOU,
        max_missed: int = INTEREST_MAX_MISSED,
        min_hits: int = INTEREST_TRACK_MIN_HITS,
        spawn_predicate: Callable[[object], bool] | None = None,
    ) -> None:
        self.match_iou = match_iou
        # Generous on purpose: a track that walks into the shop must survive
        # the frames it takes to cross the doorway (or the pipeline loses the
        # identity and can no longer tell "returning customer" from "new
        # interested passer-by" when they walk back out).
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.spawn_predicate = spawn_predicate
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, frame_idx: int, dets: list) -> tuple[list[Track], list[Track]]:
        active = list(self.tracks.values())
        assigned: set[int] = set()
        used_dets: set[int] = set()

        if active and dets:
            cost = np.zeros((len(dets), len(active)), np.float32)
            for i, det in enumerate(dets):
                for j, tr in enumerate(active):
                    cost[i, j] = 1.0 - iou(det.box, tr.predicted_box())
            order = sorted(
                ((cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1]))
            )
            max_cost = 1.0 - self.match_iou
            for c, i, j in order:
                if c > max_cost:
                    break
                tr = active[j]
                if i in used_dets or tr.track_id in assigned:
                    continue
                self._attach(tr, frame_idx, dets[i])
                used_dets.add(i)
                assigned.add(tr.track_id)

        for i, det in enumerate(dets):
            if i in used_dets:
                continue
            if self.spawn_predicate is not None and not self.spawn_predicate(det):
                continue
            tid = self.next_id
            self.next_id += 1
            tr = Track(
                track_id=tid,
                det=det,
                box=det.box,
                foot_px=det.foot_point(),
                first_frame=frame_idx,
                last_frame=frame_idx,
            )
            tr.history.append((frame_idx, tr.foot_px.copy(), float(det.height)))
            self.tracks[tid] = tr
            assigned.add(tid)

        retired: list[Track] = []
        for tr in list(self.tracks.values()):
            if tr.track_id in assigned:
                continue
            tr.missed += 1
            if tr.missed > self.max_missed:
                retired.append(self.tracks.pop(tr.track_id))

        live = [self.tracks[t] for t in assigned if t in self.tracks]
        live.sort(key=lambda t: t.track_id)
        return live, retired

    def _attach(self, tr: Track, frame_idx: int, det) -> None:
        prev_center = np.array(
            [(tr.box[0] + tr.box[2]) * 0.5, (tr.box[1] + tr.box[3]) * 0.5], np.float32
        )
        new_center = np.array(
            [(det.box[0] + det.box[2]) * 0.5, (det.box[1] + det.box[3]) * 0.5], np.float32
        )
        gap = max(1, frame_idx - tr.last_frame)
        tr.vel = (new_center - prev_center) / gap
        tr.det = det
        tr.box = det.box
        tr.foot_px = det.foot_point()
        tr.hits += 1
        tr.missed = 0
        tr.last_frame = frame_idx
        tr.history.append((frame_idx, tr.foot_px.copy(), float(det.height)))
