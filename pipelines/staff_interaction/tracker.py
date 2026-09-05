"""Generic person tracker (IoU + ReID) for the staff-interaction pipeline.

Near-duplicate of `pipelines/shelf_vector_interest/tracker.py`, kept local so
this pipeline stays free of imports from a sibling pipeline (see
`pipelines/README.md`). The only addition over that copy is the per-track
role bookkeeping this pipeline needs (`role`, `role_votes`, `role_checks`,
`last_role_query_frame`) -- tracking mechanics are otherwise identical.

Matching is two-staged:
  1. IoU continuity first -- keeps the same ID through pose/orientation
     changes and appearance shifts (turning, bending, lighting) as long as
     the box still overlaps its predicted position.
  2. ReID rescue for anything IoU couldn't match -- recovers identity
     through brief occlusion or crossing paths.
A short-lived "retired" bank lets a track that was lost for a while
(camera drop, long occlusion) reclaim its old ID by appearance rather than
starting a fresh one. Per the brief, re-identification across separate
appearances is optional for staff -- this reactivation window is kept short
(`retired_ttl_frames`) so it only bridges brief occlusions, not a staff
member leaving and re-entering camera view later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


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


def _center(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if an < 1e-6 or bn < 1e-6:
        return 0.0
    return float(np.clip((a @ b) / (an * bn), -1.0, 1.0))


@dataclass
class Track:
    track_id: int
    det: object
    box: tuple[int, int, int, int]
    foot_px: np.ndarray
    first_frame: int
    last_frame: int
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    embedding: np.ndarray | None = None
    similarity: float = 1.0
    hits: int = 1
    missed: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=180))

    # --- staff-role state, sticky for as long as the track lives -----------
    # Votes/checks are generic (not apron-specific) since role is decided by
    # `VLMJudge.ask_role` per crop, not by a colour threshold -- see scoring.py.
    role: str = "customer"
    role_votes: int = 0
    role_checks: int = 0
    last_role_query_frame: int = -(10**9)

    @property
    def is_staff(self) -> bool:
        return self.role == "staff"

    def predicted_box(self) -> tuple[int, int, int, int]:
        dx, dy = float(self.vel[0]) * self.missed, float(self.vel[1]) * self.missed
        x1, y1, x2, y2 = self.box
        return int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)


class PersonTracker:
    def __init__(
        self,
        match_iou: float = 0.25,
        max_missed: int = 45,
        reid_weight: float = 0.65,
        reid_threshold: float = 0.52,
        emb_ema: float = 0.85,
        relink_similarity: float = 0.58,
        relink_max_dist_bh: float = 6.0,
        retired_ttl_frames: int = 90,
    ) -> None:
        self.match_iou = match_iou
        self.max_missed = max_missed
        self.reid_weight = float(np.clip(reid_weight, 0.0, 1.0))
        self.reid_threshold = reid_threshold
        self.emb_ema = float(np.clip(emb_ema, 0.0, 0.99))
        self.relink_similarity = relink_similarity
        self.relink_max_dist_bh = relink_max_dist_bh
        self.retired_ttl_frames = retired_ttl_frames
        self.tracks: dict[int, Track] = {}
        self.retired_bank: dict[int, tuple[Track, int]] = {}
        self.next_id = 1

    def update(
        self, frame_idx: int, dets: list, embeddings: np.ndarray | None = None
    ) -> tuple[list[Track], list[Track]]:
        self._prune_retired_bank(frame_idx)
        active = list(self.tracks.values())
        assigned: set[int] = set()
        used_dets: set[int] = set()
        use_reid = (
            embeddings is not None
            and len(dets) > 0
            and isinstance(embeddings, np.ndarray)
            and embeddings.shape[0] == len(dets)
            and embeddings.shape[1] > 0
        )

        if active and dets:
            iou_cost = np.ones((len(dets), len(active)), np.float32)
            for i, det in enumerate(dets):
                for j, tr in enumerate(active):
                    iou_cost[i, j] = 1.0 - iou(det.box, tr.predicted_box())
            iou_order = sorted(
                ((iou_cost[i, j], i, j) for i in range(iou_cost.shape[0]) for j in range(iou_cost.shape[1]))
            )
            iou_max_cost = 1.0 - self.match_iou
            for c, i, j in iou_order:
                if c > iou_max_cost:
                    break
                tr = active[j]
                if i in used_dets or tr.track_id in assigned:
                    continue
                iou_sim = 1.0 - c
                self._attach(tr, frame_idx, dets[i], emb=(embeddings[i] if use_reid else None), sim=iou_sim)
                used_dets.add(i)
                assigned.add(tr.track_id)

            if use_reid:
                rem_det_idx = [i for i in range(len(dets)) if i not in used_dets]
                rem_track = [tr for tr in active if tr.track_id not in assigned]
                if rem_det_idx and rem_track:
                    fuse_cost = np.ones((len(rem_det_idx), len(rem_track)), np.float32)
                    for di, i in enumerate(rem_det_idx):
                        for tj, tr in enumerate(rem_track):
                            iou_score = iou(dets[i].box, tr.predicted_box())
                            sim = _cos(embeddings[i], tr.embedding) if tr.embedding is not None else 0.0
                            fuse = self.reid_weight * sim + (1.0 - self.reid_weight) * iou_score
                            fuse_cost[di, tj] = 1.0 - fuse
                    fuse_order = sorted(
                        (
                            (fuse_cost[di, tj], di, tj)
                            for di in range(fuse_cost.shape[0])
                            for tj in range(fuse_cost.shape[1])
                        )
                    )
                    fuse_max_cost = 1.0 - self.reid_threshold
                    for c, di, tj in fuse_order:
                        if c > fuse_max_cost:
                            break
                        i = rem_det_idx[di]
                        tr = rem_track[tj]
                        if i in used_dets or tr.track_id in assigned:
                            continue
                        sim = 1.0 - c
                        self._attach(tr, frame_idx, dets[i], emb=embeddings[i], sim=sim)
                        used_dets.add(i)
                        assigned.add(tr.track_id)

        for i, det in enumerate(dets):
            if i in used_dets:
                continue
            if use_reid:
                relinked = self._try_relink(frame_idx, det, embeddings[i], assigned)
                if relinked is not None:
                    used_dets.add(i)
                    assigned.add(relinked.track_id)
                    continue
                reactivated = self._try_reactivate_retired(frame_idx, det, embeddings[i], assigned)
                if reactivated is not None:
                    used_dets.add(i)
                    assigned.add(reactivated.track_id)
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
                embedding=(embeddings[i].copy() if use_reid else None),
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
                rt = self.tracks.pop(tr.track_id)
                retired.append(rt)
                self.retired_bank[rt.track_id] = (rt, frame_idx)

        live = [self.tracks[t] for t in assigned if t in self.tracks]
        live.sort(key=lambda t: t.track_id)
        return live, retired

    def _attach(self, tr: Track, frame_idx: int, det, emb: np.ndarray | None = None, sim: float = 1.0) -> None:
        prev_center = _center(tr.box)
        new_center = _center(det.box)
        gap = max(1, frame_idx - tr.last_frame)
        tr.vel = (new_center - prev_center) / gap
        tr.det = det
        tr.box = det.box
        tr.foot_px = det.foot_point()
        tr.similarity = float(sim)
        if emb is not None:
            emb = emb.astype(np.float32, copy=False)
            if tr.embedding is None:
                tr.embedding = emb.copy()
            else:
                tr.embedding = self.emb_ema * tr.embedding + (1.0 - self.emb_ema) * emb
                n = float(np.linalg.norm(tr.embedding))
                if n > 1e-6:
                    tr.embedding = tr.embedding / n
        tr.hits += 1
        tr.missed = 0
        tr.last_frame = frame_idx
        tr.history.append((frame_idx, tr.foot_px.copy(), float(det.height)))

    def _try_relink(self, frame_idx: int, det, emb: np.ndarray, assigned: set[int]) -> Track | None:
        best, best_score = None, -1.0
        det_center = _center(det.box)
        for tr in self.tracks.values():
            if tr.track_id in assigned or tr.embedding is None:
                continue
            if tr.missed <= 0 or tr.missed > self.max_missed:
                continue
            sim = _cos(emb, tr.embedding)
            if sim < self.relink_similarity:
                continue
            tr_center = _center(tr.box)
            dist_px = float(np.linalg.norm(det_center - tr_center))
            h = max(1.0, float(tr.box[3] - tr.box[1]))
            if dist_px / h > self.relink_max_dist_bh:
                continue
            if sim > best_score:
                best_score, best = sim, tr
        if best is None:
            return None
        self._attach(best, frame_idx, det, emb=emb, sim=best_score)
        return best

    def _try_reactivate_retired(self, frame_idx: int, det, emb: np.ndarray, assigned: set[int]) -> Track | None:
        best, best_score = None, -1.0
        det_center = _center(det.box)
        for tid, (tr, retired_at) in self.retired_bank.items():
            if tid in assigned or frame_idx - retired_at > self.retired_ttl_frames or tr.embedding is None:
                continue
            sim = _cos(emb, tr.embedding)
            if sim < self.relink_similarity:
                continue
            tr_center = _center(tr.box)
            dist_px = float(np.linalg.norm(det_center - tr_center))
            h = max(1.0, float(tr.box[3] - tr.box[1]))
            if dist_px / h > self.relink_max_dist_bh:
                continue
            if sim > best_score:
                best_score, best = sim, tr
        if best is None:
            return None
        self.retired_bank.pop(best.track_id, None)
        self.tracks[best.track_id] = best
        self._attach(best, frame_idx, det, emb=emb, sim=best_score)
        return best

    def _prune_retired_bank(self, frame_idx: int) -> None:
        stale = [
            tid
            for tid, (_tr, retired_at) in self.retired_bank.items()
            if frame_idx - retired_at > self.retired_ttl_frames
        ]
        for tid in stale:
            self.retired_bank.pop(tid, None)
