"""Event logic for Task 1 (store interest / conversion) and Task 3 (staff).

Every threshold below is expressed in seconds or metres rather than frames or
pixels. Seconds survive a change of frame stride; metres survive perspective,
because the floor homography from the setup step converts image points to the
floor plane. Where no homography was drawn, the fallback normalises pixel
distances by bounding-box height, i.e. "body heights" instead of metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pose import PoseDet
from .tracker import Track
from .zones import Scene


@dataclass
class InterestParams:
    # Interest is scored from four cues so that it is never decided by stopping
    # alone, which the brief explicitly rules out.
    speed_window_s: float = 0.5
    attend_deg: float = 55.0  # torso/head cone that counts as facing the shop
    walk_mps: float = 1.20  # unremarkable walking pace -> slowdown cue = 0
    slow_mps: float = 0.45  # at or below this the slowdown cue saturates
    walk_bh: float = 1.40  # same two, in body-heights/s, when no homography
    slow_bh: float = 0.45
    approach_ref_mps: float = 0.50  # closing speed that saturates the approach cue
    head_bonus_deg: float = 30.0  # head turned this much past the torso saturates
    w_orient: float = 0.40
    w_slow: float = 0.25
    w_approach: float = 0.20
    w_head: float = 0.15
    score_threshold: float = 0.55
    sustain_s: float = 0.80  # must hold, so a single noisy frame cannot latch
    ema: float = 0.60
    entered_dwell_s: float = 0.80  # inside the interior zone = "continued in"
    min_hits: int = 5  # ignore flicker detections when counting people


@dataclass
class StaffParams:
    apron_fraction: float = 0.25  # torso pixels inside the sampled apron colour
    min_votes: int = 4
    min_ratio: float = 0.40
    max_checks: int = 40  # stop testing once a track is clearly one or the other
    reid_similarity: float = 0.60  # cosine against the enrolled staff gallery


@dataclass
class InteractionParams:
    near_m: float = 1.50  # conversational distance
    very_near_m: float = 1.00  # this close, skip the orientation requirement
    near_body_heights: float = 1.20  # fallback when no homography
    face_dot: float = 0.30  # attention vectors roughly opposing
    open_s: float = 2.00  # sustained before a session is counted
    close_s: float = 1.50  # separation before the session is considered over


@dataclass
class TrackCues:
    orient: float = 0.0
    slow: float = 0.0
    approach: float = 0.0
    head: float = 0.0
    score: float = 0.0
    speed: float = 0.0
    speed_unit: str = "m/s"
    in_exterior: bool = False
    in_interior: bool = False


@dataclass
class FrameInfo:
    frame_idx: int
    cues: dict[int, TrackCues] = field(default_factory=dict)
    active_sessions: list[tuple[int, int, float]] = field(default_factory=list)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float | None:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return None
    cos = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


class EntranceAnalytics:
    def __init__(
        self,
        scene: Scene,
        fps: float,
        interest: InterestParams | None = None,
        staff: StaffParams | None = None,
        interaction: InteractionParams | None = None,
    ) -> None:
        self.scene = scene
        self.fps = max(fps, 1e-3)
        self.p = interest or InterestParams()
        self.sp = staff or StaffParams()
        self.ip = interaction or InteractionParams()

        self.records: dict[int, dict] = {}
        self.staff_instances: dict[int, dict] = {}
        self.pairs: dict[tuple[int, int], dict] = {}
        self.sessions_log: list[dict] = []

    # ------------------------------------------------------------------ helpers

    def _world_or_none(self, pt) -> np.ndarray | None:
        return self.scene.to_world(pt)

    def _motion_direction(self, tr: Track) -> np.ndarray:
        window = max(2, int(round(self.p.speed_window_s * self.fps)))
        hist = list(tr.history)
        if len(hist) < 2:
            return np.zeros(2, np.float32)
        past = hist[max(0, len(hist) - 1 - window)]
        step = (hist[-1][1] - past[1]).astype(np.float32)
        norm = float(np.linalg.norm(step))
        return step / norm if norm > 2.0 else np.zeros(2, np.float32)

    def _speed_and_approach(self, tr: Track) -> tuple[float, float, str]:
        """Scale-normalised speed and closing speed toward the storefront."""
        window = max(2, int(round(self.p.speed_window_s * self.fps)))
        hist = list(tr.history)
        if len(hist) < 2:
            return 0.0, 0.0, "m/s" if self.scene.has_homography else "bh/s"
        now = hist[-1]
        past = hist[max(0, len(hist) - 1 - window)]
        dt = (now[0] - past[0]) / self.fps
        if dt <= 1e-6:
            return 0.0, 0.0, "m/s" if self.scene.has_homography else "bh/s"

        target_now = self.scene.storefront_target(now[1])
        target_past = self.scene.storefront_target(past[1])
        if self.scene.has_homography and now[2] is not None and past[2] is not None:
            speed = float(np.linalg.norm(now[2] - past[2])) / dt
            tw_now = self._world_or_none(target_now)
            tw_past = self._world_or_none(target_past)
            if tw_now is None or tw_past is None:
                return speed, 0.0, "m/s"
            d_now = float(np.linalg.norm(now[2] - tw_now))
            d_past = float(np.linalg.norm(past[2] - tw_past))
            return speed, (d_past - d_now) / dt, "m/s"

        scale = max(float(np.mean([h[3] for h in hist[-window:]])), 1.0)
        speed = float(np.linalg.norm(now[1] - past[1])) / scale / dt
        d_now = float(np.linalg.norm(now[1] - target_now)) / scale
        d_past = float(np.linalg.norm(past[1] - target_past)) / scale
        return speed, (d_past - d_now) / dt, "bh/s"

    # -------------------------------------------------------------------- Task 1

    def _score_interest(self, tr: Track) -> TrackCues:
        c = TrackCues()
        det: PoseDet = tr.det
        foot = tr.foot_px
        c.in_exterior = self.scene.in_exterior(foot)
        c.in_interior = self.scene.in_interior(foot)

        target = self.scene.storefront_target(foot)
        to_shop = np.asarray(target, np.float32) - np.asarray(foot, np.float32)

        torso = det.facing_vector()
        head = det.attention_vector()
        if not torso.any():
            # Distant walkers often have no confident shoulders. People walk
            # forwards, so heading is a usable stand-in for body orientation.
            torso = self._motion_direction(tr)
        if not head.any():
            head = torso
        ang_head = _angle_deg(head, to_shop)
        ang_torso = _angle_deg(torso, to_shop)
        if ang_head is not None:
            c.orient = float(np.clip(1.0 - ang_head / self.p.attend_deg, 0.0, 1.0))
        if ang_head is not None and ang_torso is not None:
            c.head = float(
                np.clip((ang_torso - ang_head) / self.p.head_bonus_deg, 0.0, 1.0)
            )

        speed, closing, unit = self._speed_and_approach(tr)
        c.speed, c.speed_unit = speed, unit
        walk = self.p.walk_mps if unit == "m/s" else self.p.walk_bh
        slow = self.p.slow_mps if unit == "m/s" else self.p.slow_bh
        c.slow = float(np.clip((walk - speed) / max(walk - slow, 1e-6), 0.0, 1.0))
        c.approach = float(np.clip(closing / self.p.approach_ref_mps, 0.0, 1.0))

        c.score = (
            self.p.w_orient * c.orient
            + self.p.w_slow * c.slow
            + self.p.w_approach * c.approach
            + self.p.w_head * c.head
        )
        return c

    def _update_task1(self, tr: Track) -> TrackCues:
        c = self._score_interest(tr)
        if c.in_exterior:
            tr.seen_exterior = True

        # Interest is only judged out on the walkway, which is the zone the
        # brief describes as passers-by in front of the store.
        if c.in_exterior and not tr.interested:
            tr.interest_ema = self.p.ema * tr.interest_ema + (1.0 - self.p.ema) * c.score
            tr.best_interest = max(tr.best_interest, tr.interest_ema)
            if tr.interest_ema >= self.p.score_threshold:
                tr.interest_run += 1
            else:
                tr.interest_run = 0
            if tr.interest_run >= max(1, int(round(self.p.sustain_s * self.fps))):
                tr.interested = True
                tr.interest_frame = tr.last_frame

        if c.in_interior:
            tr.inside_run += 1
            if tr.inside_run >= max(1, int(round(self.p.entered_dwell_s * self.fps))):
                tr.entered = True
        else:
            tr.inside_run = 0

        self.records[tr.track_id] = {
            "interested": tr.interested,
            "entered": tr.entered,
            "hits": tr.hits,
            "seen_exterior": tr.seen_exterior,
            "best_interest": tr.best_interest,
            "first_frame": tr.first_frame,
            "last_frame": tr.last_frame,
        }
        return c

    # -------------------------------------------------------------------- Task 3

    def _update_role(self, tr: Track, frame_bgr: np.ndarray) -> None:
        if tr.is_staff:
            self._touch_staff(tr)
            return

        gallery = self.scene.staff_matrix
        if gallery is not None and tr.emb.size:
            sim = float(np.max(gallery @ tr.emb))
            tr.staff_sim = max(tr.staff_sim, sim)
            if sim >= self.sp.reid_similarity:
                tr.role = "staff"
                self._touch_staff(tr)
                return

        if self.scene.apron_hsv and tr.apron_checks < self.sp.max_checks:
            frac = tr.det.apron_fraction(frame_bgr, self.scene.apron_hsv)
            if frac is not None:
                tr.apron_checks += 1
                if frac >= self.sp.apron_fraction:
                    tr.apron_votes += 1
        # Sticky by majority: one bad frame must not flip a label, and the brief
        # treats a staff member as the same instance for as long as they are in view.
        if (
            tr.apron_votes >= self.sp.min_votes
            and tr.apron_checks > 0
            and tr.apron_votes / tr.apron_checks >= self.sp.min_ratio
        ):
            tr.role = "staff"
            self._touch_staff(tr)

    def _touch_staff(self, tr: Track) -> None:
        rec = self.staff_instances.setdefault(
            tr.track_id,
            {"first_frame": tr.first_frame, "last_frame": tr.last_frame, "sessions": 0},
        )
        rec["last_frame"] = tr.last_frame

    def _pair_distance(self, a: Track, b: Track) -> tuple[float, str]:
        if self.scene.has_homography and a.world is not None and b.world is not None:
            return float(np.linalg.norm(a.world - b.world)), "m"
        scale = max((a.det.height + b.det.height) * 0.5, 1.0)
        return float(np.linalg.norm(a.foot_px - b.foot_px)) / scale, "bh"

    def _update_interactions(self, frame_idx: int, live: list[Track]) -> list[tuple[int, int, float]]:
        staff = [t for t in live if t.is_staff]
        customers = [t for t in live if not t.is_staff]
        open_frames = max(1, int(round(self.ip.open_s * self.fps)))
        close_frames = max(1, int(round(self.ip.close_s * self.fps)))

        seen_pairs = set()
        for s in staff:
            for cst in customers:
                key = (s.track_id, cst.track_id)
                seen_pairs.add(key)
                dist, unit = self._pair_distance(s, cst)
                limit = self.ip.near_m if unit == "m" else self.ip.near_body_heights
                very = self.ip.very_near_m if unit == "m" else self.ip.near_body_heights * 0.7
                near = dist <= limit

                facing_ok = True
                if near and dist > very:
                    a, b = s.det.attention_vector(), cst.det.attention_vector()
                    if a.any() and b.any():
                        facing_ok = float(a @ b) < self.ip.face_dot
                engaged = near and facing_ok

                st = self.pairs.setdefault(
                    key, {"run": 0, "gap": 0, "open": False, "start": None, "sessions": 0}
                )
                if engaged:
                    st["run"] += 1
                    st["gap"] = 0
                    if not st["open"] and st["run"] >= open_frames:
                        st["open"] = True
                        st["start"] = frame_idx - st["run"] + 1
                        st["sessions"] += 1
                        self.staff_instances.setdefault(
                            s.track_id,
                            {
                                "first_frame": s.first_frame,
                                "last_frame": s.last_frame,
                                "sessions": 0,
                            },
                        )
                        self.staff_instances[s.track_id]["sessions"] += 1
                        self.sessions_log.append(
                            {
                                "staff_id": s.track_id,
                                "customer_id": cst.track_id,
                                "start_frame": st["start"],
                                "end_frame": None,
                            }
                        )
                else:
                    st["run"] = 0
                    st["gap"] += 1
                    if st["open"] and st["gap"] >= close_frames:
                        st["open"] = False
                        for entry in reversed(self.sessions_log):
                            if (
                                entry["staff_id"] == s.track_id
                                and entry["customer_id"] == cst.track_id
                                and entry["end_frame"] is None
                            ):
                                entry["end_frame"] = frame_idx - st["gap"]
                                break

        # Pairs where one side left the frame: let the gap counter close them.
        for key, st in self.pairs.items():
            if key in seen_pairs or not st["open"]:
                continue
            st["gap"] += 1
            st["run"] = 0
            if st["gap"] >= close_frames:
                st["open"] = False
                for entry in reversed(self.sessions_log):
                    if (
                        entry["staff_id"] == key[0]
                        and entry["customer_id"] == key[1]
                        and entry["end_frame"] is None
                    ):
                        entry["end_frame"] = frame_idx - st["gap"]
                        break

        active = []
        for (sid, cid), st in self.pairs.items():
            if st["open"] and st["start"] is not None:
                active.append((sid, cid, (frame_idx - st["start"]) / self.fps))
        return active

    # --------------------------------------------------------------------- main

    def update(self, frame_idx: int, frame_bgr: np.ndarray, live: list[Track]) -> FrameInfo:
        info = FrameInfo(frame_idx=frame_idx)
        for tr in live:
            tr.world = self._world_or_none(tr.foot_px)
            if tr.history:
                f, foot, _w, h = tr.history[-1]
                tr.history[-1] = (f, foot, tr.world, h)
            info.cues[tr.track_id] = self._update_task1(tr)
            self._update_role(tr, frame_bgr)
        info.active_sessions = self._update_interactions(frame_idx, live)
        return info

    # ------------------------------------------------------------------ results

    def task1_counts(self, live_ids: set[int] | None = None) -> dict[str, int]:
        """Counts so far. `live_ids` are people still on screen: an interested
        person who has not entered yet is pending, not a passer-by, so the three
        reported numbers stay honest mid-video and exact once the clip ends.
        """
        live_ids = live_ids or set()
        counted = [
            (tid, r)
            for tid, r in self.records.items()
            if r["hits"] >= self.p.min_hits and r["interested"]
        ]
        entered = sum(1 for _tid, r in counted if r["entered"])
        pending = sum(
            1 for tid, r in counted if not r["entered"] and tid in live_ids
        )
        return {
            "total_interested": len(counted),
            "interested_entered": entered,
            "interested_passed_by": len(counted) - entered - pending,
            "pending": pending,
        }

    def task3_summary(self) -> dict:
        rows = []
        for sid, rec in sorted(self.staff_instances.items()):
            rows.append(
                {
                    "staff_instance": f"staff_{sid}",
                    "track_id": sid,
                    "interaction_sessions": rec["sessions"],
                    "first_frame": rec["first_frame"],
                    "last_frame": rec["last_frame"],
                }
            )
        total = sum(r["interaction_sessions"] for r in rows)
        avg = (total / len(rows)) if rows else 0.0
        return {
            "staff": rows,
            "total_sessions": total,
            "staff_count": len(rows),
            "average_sessions_per_staff": avg,
        }
