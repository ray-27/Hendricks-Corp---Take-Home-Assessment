"""Store-interest scoring for the walking area.

Brief, quoted for reference (`Minimum definition of store interest`):

    A person should be considered interested when there is visual evidence
    that they direct their attention toward the store. This should not be
    determined solely by whether the person stops in front of the store;
    other observable behaviors, such as looking toward the storefront,
    turning their head or body toward it, slowing down, or approaching the
    entrance, may also indicate interest.

That maps directly onto four independent, observable cues, each normalised
to [0, 1] so no single one can decide the outcome on its own:

  1. orient    -- looking toward the storefront right now (YOLO-pose torso +
                  head/attention vector vs. the direction to the entrance).
  2. turn      -- turning their head or body toward it: the angle-to-shop is
                  *decreasing* over a short window, i.e. they are rotating
                  toward the store even if not square-on to it yet. This is
                  the literal "turning toward it" cue from the brief and is
                  deliberately separate from cue 1, which only looks at the
                  current instant.
  3. slow      -- slowing down. Sourced primarily from the SNN motion-energy
                  trend (gait/limb motion decaying = decelerating), with the
                  plain foot-point speed blended in so someone who has been
                  standing still the whole time (no trend to fall from)
                  still scores as "slow" rather than neutral.
  4. approach  -- approaching the entrance: net closing distance to the
                  entrance/storefront target over the same short window.

Cues are combined with fixed weights into one score. A person only counts
as interested once the score clears `score_threshold` *and holds* for
`sustain_s` seconds -- a single noisy frame cannot flip the decision, and
"stopping" alone is not one of the weighted cues, matching the brief's
explicit instruction not to decide interest by stopping alone.

All distances/speeds are expressed in body-heights (person bbox height),
not pixels or metres: this pipeline has no floor homography of its own (by
design -- see `pipelines/README.md`), and body-heights survive perspective
scale changes across the frame far better than raw pixels do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InterestParams:
    speed_window_s: float = 0.45  # window for foot-point speed / approach
    turn_window_s: float = 0.45  # window for the "turning toward" delta-angle cue

    attend_deg: float = 68.0  # attention cone that counts as "looking at the shop"
    turn_deg: float = 9.0  # angle swing toward the shop, over turn_window_s, that saturates cue 2

    walk_bh: float = 1.55  # unremarkable walking pace, body-heights/s -> slowdown cue = 0
    slow_bh: float = 0.55  # at/below this the speed-based half of the slowdown cue saturates
    motion_trend_ref: float = 0.035  # SNN energy drop (per ~0.5s) that saturates the trend half

    approach_ref_bh: float = 0.25  # closing speed (body-heights/s) that saturates cue 4

    w_orient: float = 0.32
    w_turn: float = 0.18
    w_slow: float = 0.20
    w_approach: float = 0.30

    score_threshold: float = 0.47
    sustain_s: float = 0.45
    ema: float = 0.45

    entered_dwell_s: float = 0.25  # continuous time inside the shop polygon = "entered"
    entered_backfill_score: float = 0.38  # entered + meaningful outside signal => mark interested
    min_hits: int = 4  # ignore flicker detections when counting people


@dataclass
class TrackCues:
    orient: float = 0.0
    turn: float = 0.0
    slow: float = 0.0
    approach: float = 0.0
    score: float = 0.0
    speed_bh: float = 0.0
    motion_energy: float = 0.0
    motion_trend: float = 0.0
    in_outside: bool = False
    in_inside: bool = False


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float | None:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return None
    cos = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _foot_speed_and_approach(track, boundary, fps: float, p: InterestParams) -> tuple[float, float]:
    """Body-height-normalised speed and closing speed toward the entrance."""
    window = max(2, int(round(p.speed_window_s * fps)))
    hist = list(track.history)
    if len(hist) < 2:
        return 0.0, 0.0
    now = hist[-1]
    past = hist[max(0, len(hist) - 1 - window)]
    dt = (now[0] - past[0]) / fps
    if dt <= 1e-6:
        return 0.0, 0.0

    scale = max(float(np.mean([h[2] for h in hist[-window:]])), 1.0)
    speed = float(np.linalg.norm(now[1] - past[1])) / scale / dt

    tgt_now = boundary.entrance_target(now[1])
    tgt_past = boundary.entrance_target(past[1])
    d_now = float(np.linalg.norm(now[1] - tgt_now)) / scale
    d_past = float(np.linalg.norm(past[1] - tgt_past)) / scale
    approach = (d_past - d_now) / dt
    return speed, approach


def _turn_cue(track, ang: float | None, fps: float, p: InterestParams) -> float:
    """Positive when the angle-to-shop has been shrinking (turning toward it)."""
    window = max(1, int(round(p.turn_window_s * fps)))
    hist = list(track.angle_hist)
    prev_ang = None
    if hist:
        now_frame = hist[-1][0]
        for f, a in reversed(hist):
            if a is not None and now_frame - f >= window:
                prev_ang = a
                break
        if prev_ang is None:
            for f, a in hist:
                if a is not None:
                    prev_ang = a
                    break
    track.angle_hist.append((track.last_frame, ang))
    if ang is None or prev_ang is None:
        return 0.0
    delta = prev_ang - ang  # positive => turned toward the shop since `window` frames ago
    return float(np.clip(delta / p.turn_deg, 0.0, 1.0))


def score_track(track, det, boundary, snn_out: dict, fps: float, p: InterestParams) -> TrackCues:
    c = TrackCues()
    foot = track.foot_px
    c.in_outside = boundary.in_outside(foot)
    c.in_inside = boundary.in_inside(foot)

    target = boundary.entrance_target(foot)
    to_shop = np.asarray(target, np.float32) - np.asarray(foot, np.float32)

    torso = det.facing_vector()
    head = det.attention_vector()
    if not torso.any():
        # No confident shoulders (common for distant walkers): people walk
        # forwards, so recent heading is a usable stand-in for body orientation.
        hist = list(track.history)
        if len(hist) >= 2:
            back = hist[max(0, len(hist) - 6)]
            step = (hist[-1][1] - back[1]).astype(np.float32)
            n = float(np.linalg.norm(step))
            torso = step / n if n > 2.0 else np.zeros(2, np.float32)
    if not head.any():
        head = torso

    ang = _angle_deg(head, to_shop)
    if ang is not None:
        c.orient = float(np.clip(1.0 - ang / p.attend_deg, 0.0, 1.0))

    c.turn = _turn_cue(track, ang, fps, p)

    speed_bh, approach_bh = _foot_speed_and_approach(track, boundary, fps, p)
    c.speed_bh = speed_bh
    c.motion_energy = float(snn_out.get("motion_energy", 0.0))
    c.motion_trend = float(snn_out.get("motion_trend", 0.0))

    slow_from_trend = float(np.clip(-c.motion_trend / p.motion_trend_ref, 0.0, 1.0))
    slow_from_speed = float(
        np.clip((p.walk_bh - speed_bh) / max(p.walk_bh - p.slow_bh, 1e-6), 0.0, 1.0)
    )
    c.slow = float(np.clip(0.6 * slow_from_trend + 0.4 * slow_from_speed, 0.0, 1.0))

    c.approach = float(np.clip(approach_bh / p.approach_ref_bh, 0.0, 1.0))

    c.score = (
        p.w_orient * c.orient
        + p.w_turn * c.turn
        + p.w_slow * c.slow
        + p.w_approach * c.approach
    )
    return c


def update_track_state(track, cues: TrackCues, fps: float, p: InterestParams) -> None:
    if cues.in_outside:
        track.seen_outside = True

    # Interest is only judged out on the walkway -- the zone the brief
    # describes as passers-by in front of the store -- and only *before*
    # the person has entered. Once `track.entered` is True the person is a
    # customer, not a passer-by: if they later walk back out (to leave),
    # any slowing down / looking around at the doorway on the way out must
    # not be re-scored as a fresh "interested" passer-by. This is why the
    # tracker is asked to keep the same track id across the outside/inside
    # boundary (see tracker.py's `spawn_predicate`) -- without that, a
    # returning customer would look like a brand new, un-entered track.
    if cues.in_outside and not track.interested and not track.entered:
        track.interest_ema = p.ema * track.interest_ema + (1.0 - p.ema) * cues.score
        track.best_interest = max(track.best_interest, track.interest_ema)
        if track.interest_ema >= p.score_threshold:
            track.interest_run += 1
        else:
            track.interest_run = 0
        if track.interest_run >= max(1, int(round(p.sustain_s * fps))):
            track.interested = True
            track.interest_frame = track.last_frame

    if cues.in_inside:
        track.inside_run += 1
        if track.inside_run >= max(1, int(round(p.entered_dwell_s * fps))):
            track.entered = True
            # Backfill: if the person quickly entered before clearing the
            # sustain gate, but we saw a meaningful interest score outside,
            # count them as interested_entered instead of dropping them.
            if (
                not track.interested
                and track.seen_outside
                and track.best_interest >= p.entered_backfill_score
            ):
                track.interested = True
                track.interest_frame = track.last_frame
    else:
        track.inside_run = 0
