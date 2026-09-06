"""Staff-role classification + staff-customer interaction scoring.

Brief, quoted for reference:

    Staff members can be visually distinguished from customers by the
    aprons they wear. ... For this assessment, a staff member should be
    treated as the same staff instance for as long as they remain within
    the camera view. ... An interaction should involve evidence that a
    staff member and customer are actively engaging with one another.
    Candidates should define reasonable criteria and temporal rules for
    determining when an interaction begins and ends...

Two judgments, each rule/model-free at inference time:

  role            `RoleReIDParams` / `update_role_reid` -- cosine similarity
                  against a small, manually-enrolled ReID gallery
                  (`staff_gui.py`).
  interaction     `RuleInteractionParams` / `RuleInteractionTracker` -- a
                  weighted proximity + mutual-facing score from pose
                  keypoints, no model call.

Why rule/reid instead of a VLM
------------------------------
A VLM-based version of each judgment was tried first (asking "is this a
staff apron?" / "are these two people interacting?" per crop) and dropped
entirely: two rounds of prompt engineering on the role question each fixed
one failure mode by making the other worse (a stricter "is this an apron"
prompt that stopped matching customers' bags/jackets also started
rejecting real staff whose uniform did not look exactly like the
description, and a looser prompt did the reverse), the interaction VLM had
the mirror problem (a real staff-customer exchange, e.g. one bent down
talking to a seated customer, was sometimes answered "no" because the
framing did not read as the VLM's idea of "actively interacting"), and
both were the slowest part of the pipeline by a wide margin (0.5-2s per
call on Apple Silicon MPS).

The rule/reid approach trades a small one-time manual step (enrolling
staff once via `staff_gui.py`) for classifiers that are deterministic,
fast (no model call at inference time -- role is a cosine similarity
against a gallery the tracker's ReID embedder is already computing;
interaction is arithmetic on pose keypoints already extracted for
tracking), and tunable by adjusting a handful of named thresholds instead
of English prompt wording. See `pipelines/staff_interaction/README.md` for
the full before/after reasoning.

Session semantics
------------------
  - A session opens once a pair has been continuously "engaged" for
    `open_s` and closes after a gap of `gap_close_s` without engagement --
    this is what turns a noisy per-frame signal into one session per
    continuous conversation instead of many.
  - `min_event_s` filters out a session that opened and immediately closed
    from being counted as a real session (threshold jitter, not a real
    exchange).
  - `cooldown_s` after a close: the same (staff, customer) pair cannot open
    a new session again until this elapses, which is what makes "customer
    leaves and later returns to the same staff member" count as a
    *separate* session, per the brief.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import (  # noqa: E402
    STAFF_COOLDOWN_S,
    STAFF_FACE_DEG,
    STAFF_GAP_CLOSE_S,
    STAFF_MIN_EVENT_S,
    STAFF_MIN_HITS,
    STAFF_NEAR_BH,
    STAFF_OPEN_S,
    STAFF_ROLE_MIN_CHECKS,
    STAFF_ROLE_MIN_RATIO,
    STAFF_ROLE_MIN_VOTES,
    STAFF_ROLE_SIM_THRESHOLD,
    STAFF_SCORE_THRESHOLD,
    STAFF_VERY_NEAR_BH,
    STAFF_W_FACE,
    STAFF_W_PROX,
)


# --------------------------------------------------------------------- role


@dataclass
class RoleReIDParams:
    similarity_threshold: float = STAFF_ROLE_SIM_THRESHOLD
    min_votes: int = STAFF_ROLE_MIN_VOTES
    min_ratio: float = STAFF_ROLE_MIN_RATIO
    min_checks: int = STAFF_ROLE_MIN_CHECKS


def update_role_reid(track, gallery: np.ndarray | None, p: RoleReIDParams) -> None:
    """Mutates `track.role` in place, from cosine similarity between the
    track's running appearance embedding (already maintained by
    `tracker.PersonTracker` for ReID-assisted tracking itself -- this reuses
    it rather than computing anything new) and a small gallery of staff
    embeddings enrolled once via `staff_gui.py`.

    Essentially free per frame (one small matrix-vector product), so unlike
    the VLM version there is no cooldown/budget here -- every frame with a
    usable embedding is a vote, and majority-of-recent-frames naturally
    smooths out the odd bad-angle frame without a separate retry/recheck
    mechanism.
    """
    if track.is_staff:
        return
    if gallery is None or gallery.shape[0] == 0 or track.embedding is None:
        return

    sim = float(np.max(gallery @ track.embedding))
    track.role_checks += 1
    if sim >= p.similarity_threshold:
        track.role_votes += 1
    if (
        track.role_checks >= p.min_checks
        and track.role_votes >= p.min_votes
        and track.role_votes / track.role_checks >= p.min_ratio
    ):
        track.role = "staff"


# ------------------------------------------------------------- interaction


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
class RuleInteractionParams:
    near_bh: float = STAFF_NEAR_BH
    very_near_bh: float = STAFF_VERY_NEAR_BH
    face_deg: float = STAFF_FACE_DEG
    w_prox: float = STAFF_W_PROX
    w_face: float = STAFF_W_FACE
    score_threshold: float = STAFF_SCORE_THRESHOLD

    open_s: float = STAFF_OPEN_S
    gap_close_s: float = STAFF_GAP_CLOSE_S
    cooldown_s: float = STAFF_COOLDOWN_S
    min_event_s: float = STAFF_MIN_EVENT_S
    min_hits: int = STAFF_MIN_HITS


@dataclass
class RuleCues:
    dist_bh: float = 999.0
    prox: float = 0.0
    face: float = 0.0
    score: float = 0.0
    engaged: bool = False


def rule_pair_cues(staff, cust, p: RuleInteractionParams) -> RuleCues:
    """Per-frame, model-free "are they engaging" score from pose keypoints
    alone -- geometry the tracker already has, no extra work per pair.

    Two cues, blended rather than AND-ed together so a track with a
    partially unreliable pose (occluded, bent over, back turned) still
    gets partial credit instead of failing a hard gate outright:

      prox: 1.0 when foot points coincide, 0.0 at `near_bh` body-heights
            apart. Distance is normalised by body height (not pixels) so it
            is roughly scale-invariant across near/far positions in frame.
      face: how directly each person's `attention_vector()` (torso
            direction refined by head yaw, from `analytics.pose.PoseDet`)
            points at the other person's foot point, averaged over
            whichever of the two has a usable vector. If *neither* has a
            usable vector (both back-on or keypoints missing) this is
            neutral (0.5) rather than 0 -- a missing cue should not by
            itself veto an otherwise very close, sustained pair.
      Below `very_near_bh` the facing cue is skipped entirely (forced to
            1.0): at that range (handing something over, looking at the
            same small object) orientation stops being a meaningful signal
            and requiring it produces false negatives (see README).

    Distances beyond `near_bh` short-circuit to a zero score rather than
    computing a facing cue at all -- prevents a coincidentally
    well-oriented pair on opposite sides of the store from scoring above
    zero.
    """
    bh = max(1.0, (float(staff.det.height) + float(cust.det.height)) * 0.5)
    dist_bh = float(np.linalg.norm(staff.foot_px.astype(np.float32) - cust.foot_px.astype(np.float32))) / bh
    if dist_bh > p.near_bh:
        return RuleCues(dist_bh=dist_bh)

    prox = float(np.clip(1.0 - dist_bh / p.near_bh, 0.0, 1.0))
    if dist_bh <= p.very_near_bh:
        face = 1.0
    else:
        vec_to_cust = _unit((cust.foot_px - staff.foot_px).astype(np.float32))
        vec_to_staff = -vec_to_cust
        angs = []
        a_s = staff.det.attention_vector()
        if a_s.any():
            angs.append(_angle_deg(a_s, vec_to_cust))
        a_c = cust.det.attention_vector()
        if a_c.any():
            angs.append(_angle_deg(a_c, vec_to_staff))
        face = float(np.clip(1.0 - (sum(angs) / len(angs)) / p.face_deg, 0.0, 1.0)) if angs else 0.5

    score = p.w_prox * prox + p.w_face * face
    return RuleCues(dist_bh=dist_bh, prox=prox, face=face, score=score, engaged=score >= p.score_threshold)


@dataclass
class _RulePairState:
    open: bool = False
    start_frame: int = 0
    last_engaged_frame: int = 0
    gap_run: int = 0
    candidate_run: int = 0


class RuleInteractionTracker:
    """Per-(staff, customer)-pair open/close state machine, advanced by a
    per-frame weighted score instead of throttled VLM answers. Same
    open/gap/cooldown/min-event shape as `InteractionTracker` below (and as
    `shelf_vector_interest.scoring.EngagementTracker`) -- only the per-frame
    signal differs, so `staff_interaction_pipeline.py` can swap between the
    two with the same call sites.
    """

    def __init__(self, params: RuleInteractionParams, fps: float) -> None:
        self.p = params
        self.fps = max(float(fps), 1e-6)
        self.pairs: dict[tuple[int, int], _RulePairState] = {}
        self.last_close: dict[tuple[int, int], int] = {}
        self.session_counts: dict[int, int] = defaultdict(int)

    def _st(self, key: tuple[int, int]) -> _RulePairState:
        st = self.pairs.get(key)
        if st is None:
            st = _RulePairState()
            self.pairs[key] = st
        return st

    def register_staff(self, staff_id: int) -> None:
        if staff_id not in self.session_counts:
            self.session_counts[staff_id] = 0

    def is_open(self, staff_id: int, customer_id: int) -> bool:
        st = self.pairs.get((staff_id, customer_id))
        return bool(st and st.open)

    def current_duration_s(self, staff_id: int, customer_id: int, frame_idx: int) -> float:
        st = self.pairs.get((staff_id, customer_id))
        if st is None or not st.open:
            return 0.0
        end = max(st.start_frame, frame_idx)
        return (end - st.start_frame + 1) / self.fps

    def update(self, staff, cust, cues: RuleCues, frame_idx: int) -> "InteractionSession | None":
        p = self.p
        key = (staff.track_id, cust.track_id)
        st = self._st(key)
        sustain_frames = max(1, int(round(p.open_s * self.fps)))
        gap_frames = max(1, int(round(p.gap_close_s * self.fps)))
        cooldown_frames = max(0, int(round(p.cooldown_s * self.fps)))

        if st.open:
            if cues.engaged:
                st.last_engaged_frame = frame_idx
                st.gap_run = 0
                return None
            st.gap_run += 1
            if st.gap_run > gap_frames:
                return self._finalize(key, st, frame_idx, "gap")
            return None

        if staff.hits < p.min_hits or cust.hits < p.min_hits:
            return None

        if cues.engaged:
            st.candidate_run += 1
            if st.candidate_run >= sustain_frames:
                last = self.last_close.get(key, -(10**9))
                if frame_idx - last >= cooldown_frames:
                    st.open = True
                    st.start_frame = frame_idx - st.candidate_run + 1
                    st.last_engaged_frame = frame_idx
                    st.gap_run = 0
                    st.candidate_run = 0
                # else: still cooling down on this pair -- keep accumulating
                # candidate_run so it opens the instant cooldown elapses.
        else:
            st.candidate_run = 0
        return None

    def close_all_for(self, track_id: int, frame_idx: int) -> list:
        out = []
        for key, st in list(self.pairs.items()):
            if not st.open:
                continue
            if key[0] == track_id or key[1] == track_id:
                ev = self._finalize(key, st, frame_idx, "separated", end_frame_override=frame_idx)
                if ev is not None:
                    out.append(ev)
        return out

    def close_end(self, frame_idx: int) -> list:
        out = []
        for key, st in list(self.pairs.items()):
            if st.open:
                ev = self._finalize(key, st, frame_idx, "end")
                if ev is not None:
                    out.append(ev)
        return out

    def _finalize(
        self,
        key: tuple[int, int],
        st: _RulePairState,
        frame_idx: int,
        reason: str,
        end_frame_override: int | None = None,
    ) -> "InteractionSession | None":
        start = st.start_frame
        end = max(start, end_frame_override if end_frame_override is not None else st.last_engaged_frame)
        self.last_close[key] = frame_idx
        st.open = False
        st.gap_run = 0
        st.candidate_run = 0

        duration_s = (end - start + 1) / self.fps
        if duration_s < self.p.min_event_s:
            return None

        self.session_counts[key[0]] += 1
        return InteractionSession(
            staff_id=key[0],
            customer_id=key[1],
            start_frame=start,
            end_frame=end,
            duration_s=duration_s,
            reason=reason,
        )

    def summary(self) -> dict[int, int]:
        return dict(self.session_counts)


@dataclass
class InteractionSession:
    staff_id: int
    customer_id: int
    start_frame: int
    end_frame: int
    duration_s: float
    reason: str


