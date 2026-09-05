"""Staff-role classification + staff-customer interaction scoring.

Brief, quoted for reference:

    Staff members can be visually distinguished from customers by the
    aprons they wear. ... For this assessment, a staff member should be
    treated as the same staff instance for as long as they remain within
    the camera view. ... An interaction should involve evidence that a
    staff member and customer are actively engaging with one another.
    Candidates should define reasonable criteria and temporal rules for
    determining when an interaction begins and ends...

This module now has two independent implementations of each judgment, and
the pipeline picks one per judgment via `--role-method` / `--interaction-method`:

  role            reid (default)   `RoleReIDParams` / `update_role_reid` --
                                    cosine similarity against a small,
                                    manually-enrolled ReID gallery
                                    (`staff_gui.py`).
                  vlm              `RoleParams` / `update_role` -- a VLM
                                    (`vlm_judge.VLMJudge`) asked "is this a
                                    staff apron?" per track.
  interaction     rule (default)   `RuleInteractionParams` / `RuleInteractionTracker`
                                    -- a weighted proximity + mutual-facing
                                    score from pose keypoints, no model
                                    call.
                  vlm              `InteractionParams` / `InteractionTracker`
                                    -- a VLM asked "are these two people
                                    interacting?" per pair.

Why the defaults changed from VLM to rule/reid
------------------------------------------------
Two rounds of prompt engineering on the VLM path each fixed one failure
mode by making the other one worse: a stricter "is this an apron" prompt
that stopped matching customers' bags/jackets also started rejecting real
staff whose uniform did not look exactly like the description, and a
looser prompt did the reverse. The interaction VLM had the mirror problem
-- a real staff-customer exchange (e.g. one bent down talking to a seated
customer) was sometimes answered "no" because the framing did not read as
the VLM's idea of "actively interacting". Both are also the slowest part
of the pipeline by a wide margin (0.5-2s per call on Apple Silicon MPS).

The rule/reid defaults trade a small one-time manual step (enrolling staff
once via `staff_gui.py`) for classifiers that are deterministic, fast (no
model call at inference time -- role is a cosine similarity against a
gallery the tracker's ReID embedder is already computing; interaction is
arithmetic on pose keypoints already extracted for tracking), and tunable
by adjusting a handful of named thresholds instead of English prompt
wording. Both VLM implementations are kept in this file and in
`vlm_judge.py` as an opt-in alternative (`--role-method vlm`,
`--interaction-method vlm`) since neither approach is strictly better in
every case -- see `pipelines/staff_interaction/README.md` for the full
before/after reasoning on each.

Session semantics (shared by both interaction implementations)
------------------------------------------------------------------------
  - A session opens once a pair has been continuously "engaged" for
    `open_s`/`open_confirmations` and closes after a gap of `gap_close_s`/
    `close_confirmations` without engagement -- this is what turns a
    noisy per-frame or per-query signal into one session per continuous
    conversation instead of many.
  - Moving beyond the near-distance threshold force-closes an open session
    immediately (cheap distance check) -- if they've physically separated,
    the interaction is over regardless of the last score/answer.
  - `min_event_s` filters out a session that opened and immediately closed
    from being counted as a real session (threshold jitter, not a real
    exchange).
  - `cooldown_s` after a close: the same (staff, customer) pair cannot open
    a new session again until this elapses, which is what makes "customer
    leaves and later returns to the same staff member" count as a
    *separate* session, per the brief.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------- role
# ----------------------------------------------------------- role: reid (default)


@dataclass
class RoleReIDParams:
    similarity_threshold: float = 0.62  # cosine similarity against the closest gallery vector
    min_votes: int = 3  # frames above threshold required to latch staff
    min_ratio: float = 0.55  # ...as a fraction of frames checked
    min_checks: int = 5  # don't decide off just one or two noisy frames


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


# ------------------------------------------------------------- role: vlm (opt-in)


@dataclass
class RoleParams:
    # This vote is sticky once it latches "staff" (see module docstring and
    # the brief's "same staff instance for as long as they remain in view")
    # and a false *positive* there is costly -- a phantom staff row, and
    # sessions attributed to someone who was never staff. So locking in
    # "staff" still requires most of a window to agree, not a bare
    # majority. But a false *negative* is just as real a failure mode (a
    # genuine staff member permanently written off as a customer, missing
    # from the summary table entirely) and, unlike the positive case, is
    # NOT self-correcting if we only ever give one window's worth of
    # checks -- one bad crop (bent over, back turned, odd lighting) during
    # that window and they're a "customer" forever. So instead of a single
    # one-shot budget, a track that exhausts a window without latching
    # staff gets a fresh window every `recheck_cooldown_s` for as long as
    # it's in view, at low cost since it's still gated by the per-track
    # cooldown either way.
    min_votes: int = 2  # "yes" answers required to latch staff
    min_ratio: float = 0.60  # ...as a fraction of checks made in the current window
    max_checks: int = 4  # checks per window before giving up on *this* window
    role_query_cooldown_s: float = 1.0  # spacing between role queries for the same track
    recheck_cooldown_s: float = 6.0  # after exhausting a window, wait this long then open a new one


def _torso_crop(track, frame_bgr, pad_frac: float = 0.20):
    """Crop just the torso (shoulders-to-hips, from pose keypoints) instead
    of the full person box, padded generously.

    The full-body box also includes legs, shoes, and anything the person is
    carrying below the waist, none of which is relevant to "is this an
    apron" and can distract a small VLM. Falls back to the full person box
    when torso keypoints aren't confidently visible, *or* when the
    estimated torso region is implausibly small relative to the full box
    (e.g. someone bent over rummaging in a bag, where the shoulder/hip
    keypoints can collapse toward each other) -- in both cases a wider,
    safer crop beats a tight but wrong one.
    """
    quad = track.det.torso_quad()
    box = track.det.box
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    if quad is None:
        return track.det.crop(frame_bgr)
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = int(quad[:, 0].min()), int(quad[:, 1].min())
    x2, y2 = int(quad[:, 0].max()), int(quad[:, 1].max())
    w, h = x2 - x1, y2 - y1
    if box_w > 0 and box_h > 0 and (w * h) < 0.15 * (box_w * box_h):
        return track.det.crop(frame_bgr)
    px, py = int(w * pad_frac), int(h * pad_frac)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(fw, x2 + px), min(fh, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return track.det.crop(frame_bgr)
    return frame_bgr[y1:y2, x1:x2]


def update_role(track, frame_bgr, judge, p: RoleParams, frame_idx: int, fps: float) -> None:
    """Mutates `track.role` in place. No-op once a track is already staff.
    Otherwise queries up to `max_checks` times per window; if a window is
    exhausted without latching staff, the counters reset and a new window
    opens after `recheck_cooldown_s` -- see `RoleParams` docstring for why
    this isn't a one-shot lockout."""
    if track.is_staff:
        return

    cooldown_frames = max(1, int(round(p.role_query_cooldown_s * fps)))
    if frame_idx - track.last_role_query_frame < cooldown_frames:
        return

    if track.role_checks >= p.max_checks:
        recheck_frames = max(1, int(round(p.recheck_cooldown_s * fps)))
        if frame_idx - track.last_role_query_frame < recheck_frames:
            return
        track.role_checks = 0
        track.role_votes = 0

    track.last_role_query_frame = frame_idx
    crop = _torso_crop(track, frame_bgr)
    answer = judge.ask_role(crop)
    if answer is None:
        return  # unparseable -- retried on the next query tick, doesn't count as a check
    track.role_checks += 1
    if answer:
        track.role_votes += 1
    if track.role_votes >= p.min_votes and track.role_votes / track.role_checks >= p.min_ratio:
        track.role = "staff"


# ------------------------------------------------------------- interaction
# ------------------------------------------------------ interaction: rule (default)


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
    near_bh: float = 1.45  # hard proximity gate, body-heights -- beyond this the pair can't be "engaged" at all
    very_near_bh: float = 0.75  # this close, skip the facing requirement entirely (side-by-side, handing something over)
    face_deg: float = 75.0  # facing-cone half-angle that still counts as "oriented toward the other person"
    w_prox: float = 0.55  # weight of the proximity cue in the blended score
    w_face: float = 0.45  # weight of the facing cue
    score_threshold: float = 0.50  # blended score above this counts as "engaged" this frame

    open_s: float = 1.00  # sustained engagement required to open a session
    gap_close_s: float = 1.50  # tolerated disengaged gap before closing an open session
    cooldown_s: float = 3.00  # per (staff, customer) cooldown after a close
    min_event_s: float = 1.20  # minimum duration for a close to count
    min_hits: int = 4  # ignore very fresh tracks


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


# -------------------------------------------------------- interaction: vlm (opt-in)


@dataclass
class InteractionParams:
    near_bh: float = 1.45  # conversational distance -- inside this, the pair becomes query-eligible
    far_bh: float = 2.60  # beyond this, force-close any open session (cheap, no VLM call)

    query_cooldown_s: float = 2.50  # spacing between interaction queries for the same pair
    retry_cooldown_s: float = 1.00  # shorter retry spacing after an unparseable answer

    open_confirmations: int = 1  # consecutive "yes" answers required to open a session
    close_confirmations: int = 2  # consecutive "no" answers required to close an open session

    min_event_s: float = 1.50  # minimum duration for a close to count (filters open-then-immediately-gone)
    cooldown_s: float = 3.00  # per (staff, customer) cooldown after a close -> later return is a new session
    min_hits: int = 4  # ignore very fresh tracks


@dataclass
class PairCues:
    dist_bh: float = 99.0
    eligible: bool = False  # within near_bh -- a candidate for querying at all
    queried: bool = False  # a VLM call was actually made this frame
    answer: bool | None = None  # this tick's parsed answer, if queried


@dataclass
class InteractionSession:
    staff_id: int
    customer_id: int
    start_frame: int
    end_frame: int
    duration_s: float
    reason: str


@dataclass
class _PairState:
    open: bool = False
    start_frame: int = 0
    last_engaged_frame: int = 0
    last_query_frame: int = -(10**9)
    consec_yes: int = 0
    consec_no: int = 0


class InteractionTracker:
    """Per-(staff, customer)-pair open/close state machine, advanced by
    throttled VLM answers instead of a per-frame weighted score.

    `fps` is fixed for the lifetime of a video, so it is set once here
    rather than threaded through every call (same pattern as
    `shelf_vector_interest.scoring.EngagementTracker`).
    """

    def __init__(self, params: InteractionParams, fps: float) -> None:
        self.p = params
        self.fps = max(float(fps), 1e-6)
        self.pairs: dict[tuple[int, int], _PairState] = {}
        self.last_close: dict[tuple[int, int], int] = {}
        self.session_counts: dict[int, int] = defaultdict(int)

    def _st(self, key: tuple[int, int]) -> _PairState:
        st = self.pairs.get(key)
        if st is None:
            st = _PairState()
            self.pairs[key] = st
        return st

    def register_staff(self, staff_id: int) -> None:
        """Ensures a staff instance appears in the summary even at 0 sessions."""
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

    # --------------------------------------------------------- cheap gate

    def eligible(self, staff, cust) -> PairCues:
        """Distance-only pre-check -- no VLM call. Call this every frame for
        every (staff, customer) pair; only call `update()` (which may query
        the VLM) for pairs this returns `eligible=True` for, or that are
        already open (so a force-close beyond `far_bh` can happen)."""
        h = max(1.0, (float(staff.det.height) + float(cust.det.height)) * 0.5)
        dist_bh = float(np.linalg.norm(staff.foot_px - cust.foot_px)) / h
        key = (staff.track_id, cust.track_id)
        is_open = self.is_open(*key)
        return PairCues(dist_bh=dist_bh, eligible=(dist_bh <= self.p.near_bh) or is_open)

    # ------------------------------------------------------------- update

    def update(self, staff, cust, cues: PairCues, frame_idx: int, frame_bgr, judge) -> InteractionSession | None:
        p = self.p
        key = (staff.track_id, cust.track_id)
        st = self._st(key)

        if st.open and cues.dist_bh > p.far_bh:
            # Physical separation is observed directly (cheap distance check),
            # unlike a "no longer engaged" close which relies on the last
            # confirmed VLM tick -- so the true end is *now*, not whenever
            # the pair was last re-confirmed as engaged (which could be up
            # to one `query_cooldown_s` stale and would otherwise risk
            # under-counting, or even dropping, a real short session).
            return self._finalize(key, st, frame_idx, "separated", end_frame_override=frame_idx)

        if staff.hits < p.min_hits or cust.hits < p.min_hits:
            return None

        cooldown_s = p.retry_cooldown_s if st.last_query_frame < 0 else p.query_cooldown_s
        cooldown_frames = max(1, int(round(cooldown_s * self.fps)))
        if frame_idx - st.last_query_frame < cooldown_frames:
            return None  # not due for a query yet -- reuse existing open/closed state as-is

        if not st.open:
            last = self.last_close.get(key, -(10**9))
            cooldown_after_close = max(0, int(round(p.cooldown_s * self.fps)))
            if frame_idx - last < cooldown_after_close:
                return None  # still cooling down after the previous session with this pair

        crop = _pair_crop(staff, cust, frame_bgr)
        st.last_query_frame = frame_idx
        cues.queried = True
        answer = judge.ask_interaction(crop)
        cues.answer = answer
        if answer is None:
            st.last_query_frame = frame_idx - int(round((p.query_cooldown_s - p.retry_cooldown_s) * self.fps))
            return None  # unparseable -- retried sooner, doesn't move either streak

        if answer:
            st.consec_yes += 1
            st.consec_no = 0
            st.last_engaged_frame = frame_idx
            if not st.open and st.consec_yes >= p.open_confirmations:
                st.open = True
                st.start_frame = frame_idx
                st.last_engaged_frame = frame_idx
        else:
            st.consec_no += 1
            st.consec_yes = 0
            if st.open and st.consec_no >= p.close_confirmations:
                return self._finalize(key, st, frame_idx, "no-longer-engaged")
        return None

    def close_all_for(self, track_id: int, frame_idx: int) -> list[InteractionSession]:
        """Force-closes any open pairs involving a track that just left view
        (staff or customer) -- otherwise a session could stay open forever."""
        out = []
        for key, st in list(self.pairs.items()):
            if not st.open:
                continue
            if key[0] == track_id or key[1] == track_id:
                ev = self._finalize(key, st, frame_idx, "lost")
                if ev is not None:
                    out.append(ev)
        return out

    def close_end(self, frame_idx: int) -> list[InteractionSession]:
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
        st: _PairState,
        frame_idx: int,
        reason: str,
        end_frame_override: int | None = None,
    ) -> InteractionSession | None:
        start = st.start_frame
        end = max(start, end_frame_override if end_frame_override is not None else st.last_engaged_frame)
        self.last_close[key] = frame_idx
        st.open = False
        st.consec_yes = 0
        st.consec_no = 0

        duration_s = (end - start + 1) / self.fps
        if duration_s < self.p.min_event_s:
            return None  # opened and ended too quickly to count as a real session

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


def _pair_crop(staff, cust, frame_bgr, pad_frac: float = 0.15):
    """Union bounding box of both people's boxes, padded and clipped to the
    frame, then actually cropped -- this is the image handed to the VLM for
    the interaction question. Built fresh per query rather than cached on
    the tracks, since it needs the source frame, which the tracker does not
    hold onto.
    """
    fh, fw = frame_bgr.shape[:2]
    ax1, ay1, ax2, ay2 = staff.box
    bx1, by1, bx2, by2 = cust.box
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(fw, x2 + px), min(fh, y2 + py)
    return frame_bgr[y1:y2, x1:x2]
