"""Tracking for the entrance clip.

Association is IoU-first with ReID appearance as the tie-breaker and as the way
to recover an identity after an occlusion. The bias is deliberately toward
merging rather than splitting: Task 1 reports a count of unique interested
people, and a track that splits in two would inflate that count.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .pose import PoseDet


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
    det: PoseDet
    emb: np.ndarray
    first_frame: int
    last_frame: int
    box: tuple[int, int, int, int]
    foot_px: np.ndarray
    world: np.ndarray | None = None
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hits: int = 1
    missed: int = 0
    sim: float = 1.0
    thumb: np.ndarray | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=90))

    # --- Task 1 state -------------------------------------------------------
    interest_ema: float = 0.0
    interest_run: int = 0
    interested: bool = False
    interest_frame: int | None = None
    best_interest: float = 0.0
    inside_run: int = 0
    entered: bool = False
    seen_exterior: bool = False

    # --- Task 3 state -------------------------------------------------------
    apron_votes: int = 0
    apron_checks: int = 0
    staff_sim: float = 0.0
    role: str = "customer"

    @property
    def is_staff(self) -> bool:
        return self.role == "staff"

    def predicted_box(self) -> tuple[int, int, int, int]:
        """Constant-velocity guess, so fast walkers still overlap their track."""
        dx, dy = float(self.vel[0]) * self.missed, float(self.vel[1]) * self.missed
        x1, y1, x2, y2 = self.box
        return int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)


class Tracker:
    def __init__(
        self,
        match_threshold: float = 0.45,
        max_missed: int = 30,
        min_hits: int = 5,
        w_iou: float = 0.6,
        ema: float = 0.9,
    ) -> None:
        # 0.45 cosine is permissive on purpose; splitting one shopper into two
        # identities costs more than briefly merging two.
        self.match_threshold = match_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.w_iou = w_iou
        self.ema = ema
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(
        self,
        frame_idx: int,
        dets: list[PoseDet],
        embs: np.ndarray,
        thumbs: list[np.ndarray] | None = None,
    ) -> tuple[list[Track], list[Track]]:
        """Returns (live tracks updated this frame, tracks retired this frame)."""
        active = list(self.tracks.values())
        assigned: set[int] = set()
        used_dets: set[int] = set()

        if active and dets:
            cost = np.zeros((len(dets), len(active)), np.float32)
            for i, det in enumerate(dets):
                for j, tr in enumerate(active):
                    overlap = iou(det.box, tr.predicted_box())
                    appear = float(embs[i] @ tr.emb) if embs.size else 0.0
                    cost[i, j] = 1.0 - (self.w_iou * overlap + (1.0 - self.w_iou) * appear)
            order = sorted(
                ((cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1]))
            )
            max_cost = 1.0 - self.match_threshold
            for c, i, j in order:
                if c > max_cost:
                    break
                tr = active[j]
                if i in used_dets or tr.track_id in assigned:
                    continue
                self._attach(tr, frame_idx, dets[i], embs[i], float(1.0 - c))
                if thumbs is not None:
                    tr.thumb = thumbs[i]
                used_dets.add(i)
                assigned.add(tr.track_id)

        for i, det in enumerate(dets):
            if i in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            tr = Track(
                track_id=tid,
                det=det,
                emb=embs[i] if embs.size else np.zeros(256, np.float32),
                first_frame=frame_idx,
                last_frame=frame_idx,
                box=det.box,
                foot_px=det.foot_point(),
                thumb=thumbs[i] if thumbs is not None else None,
            )
            tr.history.append((frame_idx, tr.foot_px.copy(), None, det.height))
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

    def _attach(self, tr: Track, frame_idx: int, det: PoseDet, emb: np.ndarray, sim: float) -> None:
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
        tr.sim = sim
        tr.hits += 1
        tr.missed = 0
        tr.last_frame = frame_idx
        if emb.size:
            tr.emb = self.ema * tr.emb + (1.0 - self.ema) * emb
            tr.emb = tr.emb / max(float(np.linalg.norm(tr.emb)), 1e-12)
        tr.history.append((frame_idx, tr.foot_px.copy(), None, det.height))

    def all_tracks(self) -> list[Track]:
        return list(self.tracks.values())
