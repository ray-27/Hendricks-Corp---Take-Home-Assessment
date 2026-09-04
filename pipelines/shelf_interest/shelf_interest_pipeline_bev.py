#!/usr/bin/env python3
"""
Alternative shelf-interest pipeline with strict BEV geometry.

Implements the suggested Task-2 logic:
- shelf association from shelf polygons on floor
- foot-point projection to ground plane (homography / BEV)
- facing validation toward shelf
- sustained engagement >= 2.0s
- new event only after leaving > 5.0s

This file is standalone and does not modify existing pipeline files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from analytics.pose import L_EAR, NOSE, R_EAR, PoseDetector  # noqa: E402
from analytics.zones import load_scene  # noqa: E402
from boundary.boundary_store import load_boundary  # noqa: E402
from boundary.shelf_store import load_shelf_layout, shelf_color  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402
from shelf_interest.tracker import IoUTracker  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Params:
    engage_s: float = 2.0
    cooldown_s: float = 5.0
    gap_close_s: float = 0.8
    max_assign_dist_m: float = 2.2
    max_engage_dist_m: float = 1.4
    face_deg: float = 50.0
    min_hits: int = 4


@dataclass
class Assoc:
    shelf_id: str | None = None
    shelf_name: str = ""
    dist_m: float = 1e9
    angle_deg: float = 180.0
    facing_ok: bool = False
    engaged: bool = False


class Homography:
    def __init__(self, H: np.ndarray):
        self.H = H.astype(np.float32)

    def to_world(self, pt_xy: np.ndarray) -> np.ndarray:
        src = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], np.float32)
        out = cv2.perspectiveTransform(src, self.H)
        return out[0, 0].astype(np.float32)


def _inside_strict(det, boundary) -> bool:
    foot = det.foot_point()
    cx = 0.5 * (det.box[0] + det.box[2])
    cy = 0.5 * (det.box[1] + det.box[3])
    c = np.array([cx, cy], np.float32)
    return boundary.in_inside(foot) and boundary.in_inside(c)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, np.float32)


def _closest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-6:
        return a
    t = float((p - a) @ ab) / denom
    t = min(max(t, 0.0), 1.0)
    return a + t * ab


def _closest_on_poly(p: np.ndarray, pts: list[list[int]]) -> np.ndarray:
    arr = np.asarray(pts, np.float32)
    if len(arr) == 0:
        return p
    if len(arr) == 1:
        return arr[0]
    best = arr[0]
    best_d = float("inf")
    for i in range(len(arr)):
        q = _closest_on_segment(p, arr[i], arr[(i + 1) % len(arr)])
        d = float(np.linalg.norm(q - p))
        if d < best_d:
            best_d = d
            best = q
    return best


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return 180.0
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _facing_vector(det) -> np.ndarray:
    # Suggested head vector: nose - midpoint(ears), fallback to existing attention.
    if det.has(NOSE, L_EAR, R_EAR):
        ear_mid = 0.5 * (det.kpts[L_EAR] + det.kpts[R_EAR])
        head = _unit((det.kpts[NOSE] - ear_mid).astype(np.float32))
        torso = det.facing_vector()
        if torso.any():
            return _unit(0.7 * head + 0.3 * torso)
        return head
    att = det.attention_vector()
    if att.any():
        return att
    torso = det.facing_vector()
    if torso.any():
        return torso
    return np.zeros(2, np.float32)


def _build_homography(video_name: str, floor_pts_arg, floor_size_arg) -> Homography:
    if floor_pts_arg is not None and len(floor_pts_arg) == 8:
        src = np.array(floor_pts_arg, np.float32).reshape(4, 2)
        w, h = float(floor_size_arg[0]), float(floor_size_arg[1])
        dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        return Homography(H)

    # Preferred project-local config for this pipeline:
    # pipelines/configs/bev_floor_points.json
    # {
    #   "interior.mp4": {
    #     "floor_pts": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
    #     "floor_w_m": 4.0,
    #     "floor_h_m": 3.0
    #   }
    # }
    bev_cfg = ROOT / "pipelines" / "configs" / "bev_floor_points.json"
    if bev_cfg.exists():
        try:
            data = json.loads(bev_cfg.read_text())
            entry = data.get(video_name, {})
            pts = entry.get("floor_pts", [])
            if len(pts) == 4:
                src = np.array(pts, np.float32)
                w = float(entry.get("floor_w_m", floor_size_arg[0]))
                h = float(entry.get("floor_h_m", floor_size_arg[1]))
                dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], np.float32)
                H = cv2.getPerspectiveTransform(src, dst)
                return Homography(H)
        except Exception:
            pass

    scene = load_scene(video_name)
    if len(scene.floor_pts) == 4:
        H = scene.H
        if H is not None:
            return Homography(H)

    raise SystemExit(
        "No floor homography for this video. Provide floor points:\n"
        "  --floor-pts x1 y1 x2 y2 x3 y3 x4 y4 --floor-size-m 4.0 3.0\n"
        "or add floor_pts in pipelines/configs/bev_floor_points.json (preferred)\n"
        "or add floor_pts in configs/entrance_zones.json for this video key."
    )


def associate_shelf(det, shelves, hom: Homography, p: Params) -> Assoc:
    foot_img = det.foot_point().astype(np.float32)
    foot_w = hom.to_world(foot_img)
    face = _facing_vector(det)

    best = Assoc()
    for shelf in shelves:
        tgt_img = shelf.look_target(foot_img)
        tgt_w = hom.to_world(tgt_img)
        to_shelf_img = tgt_img - foot_img
        dist_m = float(np.linalg.norm(tgt_w - foot_w))
        ang = _angle_deg(face, to_shelf_img)
        facing_ok = ang <= p.face_deg

        if dist_m < best.dist_m:
            best = Assoc(
                shelf_id=shelf.shelf_id,
                shelf_name=shelf.label(),
                dist_m=dist_m,
                angle_deg=ang,
                facing_ok=facing_ok,
                engaged=False,
            )

    if best.shelf_id is None:
        return best
    if best.dist_m > p.max_assign_dist_m:
        return Assoc()
    best.engaged = best.facing_ok and best.dist_m <= p.max_engage_dist_m
    return best


def write_outputs(out_dir: Path, shelves, events: list[dict], counts: dict[str, int]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    p1 = out_dir / "shelf_interest_bev_summary.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shelf_id", "shelf_name", "interest_events"])
        for s in shelves:
            w.writerow([s.shelf_id, s.label(), int(counts.get(s.shelf_id, 0))])
    written.append(p1)

    p2 = out_dir / "shelf_interest_bev_events.csv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "track_id", "shelf_id", "shelf_name", "start_frame", "end_frame", "duration_s"])
        for i, e in enumerate(events, start=1):
            w.writerow([i, e["track_id"], e["shelf_id"], e["shelf_name"], e["start_frame"], e["end_frame"], f"{e['duration_s']:.3f}"])
    written.append(p2)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "interior.mp4")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "shelf_interest_bev")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--pose-weights", type=str, default=None)
    ap.add_argument("--pose-imgsz", type=int, default=1280)
    ap.add_argument("--floor-pts", type=float, nargs=8, default=None, metavar=("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"))
    ap.add_argument("--floor-size-m", type=float, nargs=2, default=(4.0, 3.0), metavar=("W", "H"))
    ap.add_argument("--engage-s", type=float, default=1.0)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--gap-close-s", type=float, default=0.8)
    ap.add_argument("--max-assign-dist-m", type=float, default=2.2)
    ap.add_argument("--max-engage-dist-m", type=float, default=1.7)
    ap.add_argument("--face-deg", type=float, default=50.0)
    ap.add_argument("--reid-threshold", type=float, default=0.52)
    ap.add_argument("--reid-relink", type=float, default=0.58)
    ap.add_argument("--reid-weight", type=float, default=0.65)
    ap.add_argument("--max-missed", type=int, default=90)
    ap.add_argument("--retired-ttl-frames", type=int, default=450)
    ap.add_argument("--reid-model", type=Path, default=None)
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    p = Params(
        engage_s=args.engage_s,
        cooldown_s=args.cooldown_s,
        gap_close_s=args.gap_close_s,
        max_assign_dist_m=args.max_assign_dist_m,
        max_engage_dist_m=args.max_engage_dist_m,
        face_deg=args.face_deg,
    )

    boundary = load_boundary(args.video.name)
    layout = load_shelf_layout(args.video.name)
    shelves = layout.ready_shelves()
    if len(boundary.inside) < 3:
        raise SystemExit("Inside boundary missing. Run boundary_gui.py and draw inside polygon.")
    if not shelves:
        raise SystemExit("Shelf polygons missing. Run shelf_gui.py and draw shelves.")
    hom = _build_homography(args.video.name, args.floor_pts, args.floor_size_m)

    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    fps_eff = src_fps / stride

    pose = PoseDetector(weights=args.pose_weights, imgsz=int(args.pose_imgsz))
    reid = ReIDEmbedder(model_path=args.reid_model)
    tracker = IoUTracker(
        max_missed=args.max_missed,
        reid_weight=args.reid_weight,
        reid_threshold=args.reid_threshold,
        relink_similarity=args.reid_relink,
        retired_ttl_frames=args.retired_ttl_frames,
    )

    writer = None
    if not args.no_video:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "shelf_interest_bev_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh))

    # track_id -> state
    state = {}
    # track_id -> shelf_id -> last exit frame
    last_exit = defaultdict(dict)
    events = []
    counts = defaultdict(int)

    sustain_frames = max(1, int(round(p.engage_s * fps_eff)))
    cooldown_frames = max(1, int(round(p.cooldown_s * fps_eff)))
    gap_frames = max(1, int(round(p.gap_close_s * fps_eff)))

    frame_step = 0
    frame_raw = 0
    print(
        f"{args.video.name}: {total} frames @ {src_fps:.1f}fps stride={stride} -> {fps_eff:.1f}fps; "
        f"BEV sustain={p.engage_s:.1f}s cooldown={p.cooldown_s:.1f}s"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_raw += 1
        if (frame_raw - 1) % stride:
            continue
        frame_step += 1

        dets_all = pose.detect(frame)
        dets = [d for d in dets_all if _inside_strict(d, boundary)]
        embs = reid.embed([d.crop(frame) for d in dets]) if dets else None
        live, retired = tracker.update(frame_step, dets, embeddings=embs)

        for tr in retired:
            st = state.get(tr.track_id)
            if st and st["open"]:
                start = st["start_frame"]
                end = max(start, st["last_engaged_frame"])
                dur = (end - start + 1) / max(fps_eff, 1e-6)
                sid = st["shelf_id"]
                sname = st["shelf_name"]
                events.append(
                    {
                        "track_id": tr.track_id,
                        "shelf_id": sid,
                        "shelf_name": sname,
                        "start_frame": start,
                        "end_frame": end,
                        "duration_s": dur,
                    }
                )
                counts[sid] += 1
                last_exit[tr.track_id][sid] = frame_step
            state.pop(tr.track_id, None)

        assoc_by_id = {}
        for tr in live:
            a = associate_shelf(tr.det, shelves, hom, p)
            assoc_by_id[tr.track_id] = a

            st = state.setdefault(
                tr.track_id,
                {
                    "open": False,
                    "shelf_id": None,
                    "shelf_name": "",
                    "candidate_shelf_id": None,
                    "candidate_name": "",
                    "candidate_run": 0,
                    "start_frame": None,
                    "last_engaged_frame": None,
                    "gap_run": 0,
                },
            )

            if st["open"]:
                if a.engaged and a.shelf_id == st["shelf_id"]:
                    st["last_engaged_frame"] = frame_step
                    st["gap_run"] = 0
                else:
                    st["gap_run"] += 1
                    if st["gap_run"] > gap_frames:
                        start = st["start_frame"]
                        end = max(start, st["last_engaged_frame"])
                        dur = (end - start + 1) / max(fps_eff, 1e-6)
                        sid = st["shelf_id"]
                        sname = st["shelf_name"]
                        events.append(
                            {
                                "track_id": tr.track_id,
                                "shelf_id": sid,
                                "shelf_name": sname,
                                "start_frame": start,
                                "end_frame": end,
                                "duration_s": dur,
                            }
                        )
                        counts[sid] += 1
                        last_exit[tr.track_id][sid] = frame_step
                        st["open"] = False
                        st["shelf_id"] = None
                        st["shelf_name"] = ""
                        st["candidate_run"] = 0
                        st["candidate_shelf_id"] = None
                        st["gap_run"] = 0
                continue

            # Not open: build sustained candidate.
            if a.engaged and tr.hits >= p.min_hits:
                if st["candidate_shelf_id"] == a.shelf_id:
                    st["candidate_run"] += 1
                else:
                    st["candidate_shelf_id"] = a.shelf_id
                    st["candidate_name"] = a.shelf_name
                    st["candidate_run"] = 1
                if st["candidate_run"] >= sustain_frames:
                    sid = st["candidate_shelf_id"]
                    ex = last_exit[tr.track_id].get(sid, -10**9)
                    if frame_step - ex > cooldown_frames:
                        st["open"] = True
                        st["shelf_id"] = sid
                        st["shelf_name"] = st["candidate_name"]
                        st["start_frame"] = frame_step - st["candidate_run"] + 1
                        st["last_engaged_frame"] = frame_step
                        st["gap_run"] = 0
            else:
                st["candidate_run"] = 0
                st["candidate_shelf_id"] = None

        if writer is not None or args.preview:
            vis = frame.copy()
            cv2.polylines(vis, [np.array(boundary.inside, np.int32)], True, (80, 180, 80), 1)
            for i, s in enumerate(shelves):
                c = shelf_color(i)
                cv2.polylines(vis, [np.array(s.polygon, np.int32)], True, c, 2)
                cv2.putText(vis, s.label(), tuple(np.array(s.polygon[0], np.int32)), FONT, 0.5, c, 2)

            for tr in live:
                a = assoc_by_id.get(tr.track_id, Assoc())
                shelf = layout.shelf_by_id(a.shelf_id) if a.shelf_id else None
                color = (170, 170, 170) if shelf is None else shelf_color(
                    next((i for i, s in enumerate(shelves) if s.shelf_id == shelf.shelf_id), 0)
                )
                x1, y1, x2, y2 = tr.box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                tag = f"ID {tr.track_id} "
                tag += f"{a.shelf_name}" if a.shelf_id else "UNASSIGNED"
                st = state.get(tr.track_id, {})
                if st.get("open"):
                    tag += " INTEREST"
                cv2.putText(vis, tag, (x1 + 2, max(16, y1 - 4)), FONT, 0.45, color, 2)
                cv2.putText(
                    vis,
                    f"d={a.dist_m:.2f}m ang={a.angle_deg:.0f} fac={int(a.facing_ok)} sim={tr.similarity:.2f}",
                    (x1, min(vis.shape[0] - 4, y2 + 16)),
                    FONT,
                    0.34,
                    color,
                    1,
                )

            y = 20
            total_events = int(sum(counts.values()))
            cv2.putText(vis, f"SHELF INTEREST BEV  events={total_events}", (10, y), FONT, 0.55, (255, 255, 255), 2)
            y += 20
            for i, s in enumerate(shelves):
                c = shelf_color(i)
                cv2.putText(vis, f"{s.label()}: {int(counts[s.shelf_id])}", (10, y), FONT, 0.50, c, 2)
                y += 18

            if writer is not None:
                writer.write(vis)
            if args.preview:
                cv2.imshow("shelf interest bev (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        if args.max_frames and frame_step >= args.max_frames:
            break

    # close any open event at end
    for tid, st in state.items():
        if st["open"]:
            start = st["start_frame"]
            end = max(start, st["last_engaged_frame"])
            dur = (end - start + 1) / max(fps_eff, 1e-6)
            events.append(
                {
                    "track_id": tid,
                    "shelf_id": st["shelf_id"],
                    "shelf_name": st["shelf_name"],
                    "start_frame": start,
                    "end_frame": end,
                    "duration_s": dur,
                }
            )
            counts[st["shelf_id"]] += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    paths = write_outputs(args.out_dir, shelves, events, counts)
    print("\n--- Shelf interest BEV ---")
    for s in shelves:
        print(f"{s.label():<20} {int(counts[s.shelf_id])}")
    print(f"{'TOTAL':<20} {int(sum(counts.values()))}")
    print("\nwrote:")
    for pth in paths:
        print(" ", pth)
    if writer is not None:
        print(" ", args.out_dir / "shelf_interest_bev_annotated.mp4")


if __name__ == "__main__":
    main()

