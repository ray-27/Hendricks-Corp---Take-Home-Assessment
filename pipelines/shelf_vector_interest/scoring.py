"""Vector-based shelf engagement: association + anti-double-count state machine.

Association (per frame, per track)
-----------------------------------
A track is "engaged" with a shelf face iff:
  1. its foot point lies inside the face's `zone` polygon (hard gate), and
  2. its facing vector is within `face_deg` of `-normal` (pointing back at
     the shelf).

If a track's foot point is inside more than one face's zone (the "customer
between two shelves" case from the task brief), the face whose normal is
most directly opposed to the customer's facing vector wins -- i.e. we pick
the shelf they are actually looking at, not merely the nearest one.

Anti-double-count state machine (per track)
--------------------------------------------
- `candidate_run`: consecutive frames engaged with the same face while not
  yet "open". Once it reaches `engage_s` seconds, the event opens.
- Once open, brief drops in engagement (glancing away, a half-step) are
  tolerated for up to `gap_close_s` seconds before the event is closed --
  this prevents one continuous visit from being split into many events.
- A closed event is only counted (returned) if it lasted >= `min_event_s`
  -- this is what stops one-frame score jitter around the threshold from
  registering as an event.
- After a face's event closes, that (track, face) pair is on `cooldown_s`
  seconds of cooldown: a return within that window continues to be
  considered part of leaving/re-approaching and will not itself open a new
  event until cooldown has elapsed and `engage_s` sustain is re-earned --
  this is the required "same shelf, later, separate visit" rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from shelf_vector_interest.shelf_face_store import ShelfFace


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, np.float32)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    an, bn = _unit(a), _unit(b)
    if not an.any() or not bn.any():
        return 180.0
    c = float(np.clip(an @ bn, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


@dataclass
class VectorParams:
    face_deg: float = 55.0  # max angle between person's facing vec and -normal
    engage_s: float = 1.2  # sustained engagement required to open an event
    gap_close_s: float = 1.0  # tolerated disengaged gap before closing
    cooldown_s: float = 5.0  # per (track, shelf) cooldown after a close
    min_event_s: float = 0.8  # minimum duration for a close to count
    min_hits: int = 4  # ignore very fresh tracks (a couple of frames old)


@dataclass
class FaceAssoc:
    shelf_id: str | None = None
    shelf_name: str = ""
    in_zone: bool = False
    angle_deg: float = 180.0
    facing_ok: bool = False
    engaged: bool = False


def associate(foot_pt: np.ndarray, facing_vec: np.ndarray, faces: list[ShelfFace], p: VectorParams) -> FaceAssoc:
    best = FaceAssoc()
    for f in faces:
        if not f.ready() or not f.in_zone(foot_pt):
            continue
        ang = _angle_deg(facing_vec, -f.normal_vec())
        facing_ok = ang <= p.face_deg
        if best.shelf_id is None or ang < best.angle_deg:
            best = FaceAssoc(
                shelf_id=f.shelf_id,
                shelf_name=f.label(),
                in_zone=True,
                angle_deg=ang,
                facing_ok=facing_ok,
                engaged=facing_ok,
            )
    return best


@dataclass
class _TrackState:
    open: bool = False
    shelf_id: str | None = None
    shelf_name: str = ""
    start_frame: int = 0
    last_engaged_frame: int = 0
    gap_run: int = 0
    candidate_shelf_id: str | None = None
    candidate_name: str = ""
    candidate_run: int = 0
    event_counts: dict = field(default_factory=lambda: defaultdict(int))


@dataclass
class EngagementEvent:
    track_id: int
    shelf_id: str
    shelf_name: str
    start_frame: int
    end_frame: int
    duration_s: float
    reason: str


class EngagementTracker:
    """Per-track open/close state machine, driven by per-frame `FaceAssoc`.

    `fps` is fixed for the lifetime of a video, so it is set once here
    rather than threaded through every call.
    """

    def __init__(self, params: VectorParams, fps: float) -> None:
        self.p = params
        self.fps = max(float(fps), 1e-6)
        self.state: dict[int, _TrackState] = {}
        self.last_exit: dict[int, dict] = defaultdict(dict)

    def _st(self, track_id: int) -> _TrackState:
        st = self.state.get(track_id)
        if st is None:
            st = _TrackState()
            self.state[track_id] = st
        return st

    def event_count(self, track_id: int, shelf_id: str) -> int:
        st = self.state.get(track_id)
        return st.event_counts.get(shelf_id, 0) if st else 0

    def total_counts(self) -> dict:
        totals: dict = defaultdict(int)
        for st in self.state.values():
            for sid, n in st.event_counts.items():
                totals[sid] += n
        return totals

    def is_open(self, track_id: int) -> tuple[bool, str | None, str]:
        st = self.state.get(track_id)
        if st is None or not st.open:
            return False, None, ""
        return True, st.shelf_id, st.shelf_name

    def current_duration_s(self, track_id: int, frame_idx: int) -> float:
        st = self.state.get(track_id)
        if st is None or not st.open:
            return 0.0
        end = max(st.start_frame, frame_idx)
        return (end - st.start_frame + 1) / self.fps

    def update(self, track_id: int, hits: int, assoc: FaceAssoc, frame_idx: int) -> EngagementEvent | None:
        p = self.p
        st = self._st(track_id)
        sustain_frames = max(1, round(p.engage_s * self.fps))
        gap_frames = max(1, round(p.gap_close_s * self.fps))
        cooldown_frames = max(0, round(p.cooldown_s * self.fps))

        if st.open:
            if assoc.engaged and assoc.shelf_id == st.shelf_id:
                st.last_engaged_frame = frame_idx
                st.gap_run = 0
                return None
            st.gap_run += 1
            if st.gap_run > gap_frames:
                return self._finalize(track_id, st, frame_idx, "gap")
            return None

        if hits < p.min_hits:
            return None

        if assoc.engaged:
            if st.candidate_shelf_id == assoc.shelf_id:
                st.candidate_run += 1
            else:
                st.candidate_shelf_id = assoc.shelf_id
                st.candidate_name = assoc.shelf_name
                st.candidate_run = 1
            if st.candidate_run >= sustain_frames:
                sid = st.candidate_shelf_id
                last = self.last_exit[track_id].get(sid, -10**9)
                if frame_idx - last >= cooldown_frames:
                    st.open = True
                    st.shelf_id = sid
                    st.shelf_name = st.candidate_name
                    st.start_frame = frame_idx - st.candidate_run + 1
                    st.last_engaged_frame = frame_idx
                    st.gap_run = 0
                    st.candidate_run = 0
                    st.candidate_shelf_id = None
                # else: still cooling down on this shelf; keep accumulating candidate_run
                # so it opens the instant cooldown elapses, without losing the sustain streak
        else:
            st.candidate_run = 0
            st.candidate_shelf_id = None
        return None

    def close(self, track_id: int, frame_idx: int, reason: str = "end") -> EngagementEvent | None:
        st = self.state.get(track_id)
        if st is None or not st.open:
            return None
        return self._finalize(track_id, st, frame_idx, reason)

    def _finalize(self, track_id: int, st: _TrackState, frame_idx: int, reason: str) -> EngagementEvent | None:
        start = st.start_frame
        end = max(start, st.last_engaged_frame)
        sid, name = st.shelf_id, st.shelf_name
        self.last_exit[track_id][sid] = frame_idx
        st.open = False
        st.shelf_id = None
        st.shelf_name = ""
        st.gap_run = 0
        st.candidate_run = 0
        st.candidate_shelf_id = None

        duration_s = (end - start + 1) / self.fps
        if duration_s < self.p.min_event_s:
            return None  # too short to count -- avoids counting threshold jitter as an event

        st.event_counts[sid] += 1
        return EngagementEvent(
            track_id=track_id,
            shelf_id=sid,
            shelf_name=name,
            start_frame=start,
            end_frame=end,
            duration_s=duration_s,
            reason=reason,
        )
