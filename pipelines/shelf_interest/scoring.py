"""Per-shelf interest scoring and episode accounting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShelfInterestParams:
    speed_window_s: float = 0.50
    turn_window_s: float = 0.60
    near_shelf_bh: float = 0.95
    max_assign_dist_bh: float = 0.60
    max_event_dist_bh: float = 0.60
    attend_deg: float = 60.0
    turn_deg: float = 12.0
    walk_bh: float = 1.20
    still_bh: float = 0.35
    w_orient: float = 0.40
    w_proximity: float = 0.30
    w_slow: float = 0.20
    w_turn: float = 0.10
    score_threshold: float = 0.50
    sustain_s: float = 0.45
    fast_near_sustain_s: float = 0.20
    fast_proximity: float = 0.72
    fast_orient: float = 0.18
    min_orient_for_event: float = 0.18
    min_proximity_for_event: float = 0.30
    episode_gap_s: float = 1.20
    return_cooldown_s: float = 1.60
    min_event_duration_s: float = 0.90
    hold_dist_bh: float = 1.35
    hold_slow_min: float = 0.25
    switch_margin: float = 0.08
    min_hits: int = 4


@dataclass
class ShelfCues:
    shelf_id: str | None = None
    shelf_name: str = ""
    assign_score: float = 0.0
    orient: float = 0.0
    proximity: float = 0.0
    slow: float = 0.0
    turn: float = 0.0
    score: float = 0.0
    speed_bh: float = 0.0
    dist_bh: float = 99.0


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float | None:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return None
    cos = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _speed_bh(track, fps: float, p: ShelfInterestParams) -> float:
    window = max(2, int(round(p.speed_window_s * fps)))
    hist = list(track.history)
    if len(hist) < 2:
        return 0.0
    now = hist[-1]
    past = hist[max(0, len(hist) - 1 - window)]
    dt = (now[0] - past[0]) / max(fps, 1e-6)
    if dt <= 1e-6:
        return 0.0
    scale = max(float(np.mean([h[2] for h in hist[-window:]])), 1.0)
    return float(np.linalg.norm(now[1] - past[1])) / scale / dt


def _turn_cue(track, shelf_id: str | None, angle: float | None, fps: float, p: ShelfInterestParams) -> float:
    window = max(1, int(round(p.turn_window_s * fps)))
    hist = list(track.angle_hist)
    prev_ang = None
    if hist and shelf_id is not None:
        now_frame = hist[-1][0]
        for f, sid, a in reversed(hist):
            if sid != shelf_id or a is None:
                continue
            if now_frame - f >= window:
                prev_ang = a
                break
        if prev_ang is None:
            for _f, sid, a in hist:
                if sid == shelf_id and a is not None:
                    prev_ang = a
                    break
    track.angle_hist.append((track.last_frame, shelf_id, angle))
    if angle is None or prev_ang is None:
        return 0.0
    delta = prev_ang - angle
    return float(np.clip(delta / max(p.turn_deg, 1e-6), 0.0, 1.0))


def _score_shelf(track, det, shelf, fps: float, p: ShelfInterestParams) -> tuple[float, ShelfCues]:
    foot = np.asarray(track.foot_px, np.float32)
    h = max(1.0, float(det.height))
    target = shelf.look_target(foot)
    to_shelf = target - foot

    facing = det.attention_vector()
    if not facing.any():
        facing = det.facing_vector()
    if not facing.any():
        hist = list(track.history)
        if len(hist) >= 2:
            back = hist[max(0, len(hist) - 6)]
            step = (hist[-1][1] - back[1]).astype(np.float32)
            n = float(np.linalg.norm(step))
            facing = step / n if n > 2.0 else np.zeros(2, np.float32)

    ang = _angle_deg(facing, to_shelf)
    orient = float(np.clip(1.0 - (ang or 180.0) / max(p.attend_deg, 1e-6), 0.0, 1.0))

    dist_signed_px = shelf.signed_dist(foot)
    dist_px = max(0.0, dist_signed_px)
    dist_bh = dist_px / h
    proximity = float(np.clip(1.0 - dist_bh / max(p.near_shelf_bh, 1e-6), 0.0, 1.0))

    speed_bh = _speed_bh(track, fps, p)
    slow = float(np.clip((p.walk_bh - speed_bh) / max(p.walk_bh - p.still_bh, 1e-6), 0.0, 1.0))

    assign_score = 0.55 * orient + 0.45 * proximity
    cues = ShelfCues(
        shelf_id=shelf.shelf_id,
        shelf_name=shelf.label(),
        assign_score=assign_score,
        orient=orient,
        proximity=proximity,
        slow=slow,
        speed_bh=speed_bh,
        dist_bh=dist_bh,
    )
    return ang, cues


def score_track(track, det, boundary, fps: float, p: ShelfInterestParams) -> ShelfCues:
    shelves = boundary.ready_shelves()
    if not shelves:
        return ShelfCues()

    per = []
    angle_by_shelf = {}
    for shelf in shelves:
        ang, cues = _score_shelf(track, det, shelf, fps, p)
        per.append(cues)
        angle_by_shelf[shelf.shelf_id] = ang

    best = max(per, key=lambda x: x.assign_score)
    if track.shelf_id is not None:
        cur = next((c for c in per if c.shelf_id == track.shelf_id), None)
        if cur is not None and (best.assign_score - cur.assign_score) < p.switch_margin:
            best = cur

    # Hard assignment gate: if the person is too far from every shelf, do not
    # assign a shelf even if orientation points toward one.
    if best.dist_bh > p.max_assign_dist_bh:
        track.shelf_id = None
        return ShelfCues(speed_bh=best.speed_bh, dist_bh=best.dist_bh)

    track.shelf_id = best.shelf_id

    best.turn = _turn_cue(track, best.shelf_id, angle_by_shelf.get(best.shelf_id), fps, p)
    best.score = (
        p.w_orient * best.orient
        + p.w_proximity * best.proximity
        + p.w_slow * best.slow
        + p.w_turn * best.turn
    )
    return best


def close_event(
    track, frame_idx: int, fps: float, reason: str = "", min_duration_s: float = 0.0
) -> dict | None:
    if not track.event_open or track.event_shelf_id is None or track.event_start_frame is None:
        return None
    end_frame = int(track.event_last_strong_frame or frame_idx)
    start_frame = int(track.event_start_frame)
    if end_frame < start_frame:
        end_frame = start_frame
    duration_s = (end_frame - start_frame + 1) / max(fps, 1e-6)
    ev = {
        "track_id": track.track_id,
        "shelf_id": track.event_shelf_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_s": duration_s,
        "reason": reason,
    }
    # Count only validated events to avoid double-counting split jitter.
    if duration_s >= max(0.0, float(min_duration_s)):
        track.event_counts[track.event_shelf_id] += 1
    track.events.append(ev)
    track.event_open = False
    track.event_shelf_id = None
    track.event_start_frame = None
    track.event_last_strong_frame = None
    track.low_run = 0
    return ev


def update_track_state(track, cues: ShelfCues, fps: float, p: ShelfInterestParams) -> dict | None:
    sustain_frames = max(1, int(round(p.sustain_s * fps)))
    fast_sustain_frames = max(1, int(round(p.fast_near_sustain_s * fps)))
    gap_frames = max(1, int(round(p.episode_gap_s * fps)))
    return_cooldown_frames = max(1, int(round(p.return_cooldown_s * fps)))

    raw_score = cues.score if cues.shelf_id is not None else 0.0
    track.interest_ema = 0.55 * track.interest_ema + 0.45 * raw_score
    track.best_interest = max(track.best_interest, track.interest_ema)
    quality_ok = (
        cues.orient >= p.min_orient_for_event
        and cues.proximity >= p.min_proximity_for_event
    )
    strong = (
        cues.shelf_id is not None
        and cues.dist_bh <= p.max_event_dist_bh
        and quality_ok
        and track.interest_ema >= p.score_threshold
    )
    strong_fast = (
        cues.shelf_id is not None
        and cues.dist_bh <= p.max_event_dist_bh
        and cues.proximity >= p.fast_proximity
        and cues.orient >= p.fast_orient
    )

    if track.event_open:
        same_shelf = cues.shelf_id == track.event_shelf_id
        strong_same = strong and same_shelf
        holdable = (
            same_shelf
            and cues.dist_bh <= p.hold_dist_bh
            and cues.slow >= p.hold_slow_min
            and quality_ok
        )
        if strong_same or holdable:
            # Only grow counted duration when strict "strong" is true.
            if strong_same:
                track.event_last_strong_frame = track.last_frame
            track.low_run = 0
            return None
        track.low_run += 1
        if track.low_run > gap_frames:
            ev = close_event(
                track,
                track.last_frame,
                fps,
                reason="gap",
                min_duration_s=p.min_event_duration_s,
            )
            track.last_closed_shelf_id = ev["shelf_id"] if ev is not None else track.last_closed_shelf_id
            track.last_closed_frame = track.last_frame
            return ev
        return None

    # No open event: require sustained strong evidence before opening one.
    if strong or strong_fast:
        if track.candidate_shelf_id == cues.shelf_id:
            track.candidate_run += 1
        else:
            track.candidate_shelf_id = cues.shelf_id
            track.candidate_run = 1
        required = fast_sustain_frames if strong_fast else sustain_frames
        if track.candidate_run >= required and track.hits >= p.min_hits:
            if (
                track.last_closed_shelf_id == cues.shelf_id
                and track.last_closed_frame is not None
                and (track.last_frame - track.last_closed_frame) < return_cooldown_frames
            ):
                # Suppress immediate re-open jitter on the same shelf.
                return None
            track.event_open = True
            track.event_shelf_id = cues.shelf_id
            track.event_start_frame = track.last_frame - track.candidate_run + 1
            track.event_last_strong_frame = track.last_frame
            track.low_run = 0
    else:
        track.candidate_run = 0
        track.candidate_shelf_id = None
    return None
