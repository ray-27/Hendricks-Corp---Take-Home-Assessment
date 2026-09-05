#!/usr/bin/env python3
"""
Interest pipeline: store-interest detection for the walking area.

    frame -> YOLO11x-pose (posture: orientation, head turn)
          -> IoU tracker (position over time, no ReID)
          -> per-person spiking retina / SNN (gait motion energy -> speed &
             slowdown, see snn_motion.py)
          -> interest scoring (orient, turn, slow, approach -> see scoring.py)
          -> entered / passed-by decision from the boundary file

This pipeline is fully independent of the ReID pipeline and of the original
`run_entrance_analytics.py` at the project root: it only needs
  - the boundary file produced by `pipelines/boundary/boundary_gui.py`
    (`pipelines/configs/store_boundary_zones.json`)
  - the YOLO-pose weights (`yolo11x-pose.pt` by default; Ultralytics downloads on first run)
It does not read or write any ReID state, and can be run on its own, on any
video, as long as that video's boundary has been drawn once.

Run the boundary GUI first:
    python3 pipelines/boundary/boundary_gui.py --video raw_videos/entrance.mp4

Then:
    python3 pipelines/interest/interest_pipeline.py
    python3 pipelines/interest/interest_pipeline.py --video raw_videos/entrance.mp4 --preview
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from analytics.pose import PoseDetector  # noqa: E402
from boundary.boundary_store import load_boundary  # noqa: E402
from interest.scoring import InterestParams, score_track, update_track_state  # noqa: E402
from interest.snn_motion import SNNMotionTracker  # noqa: E402
from interest.tracker import IoUTracker  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
PASSER_BY = (200, 200, 200)
INTERESTED = (0, 215, 255)
ENTERED = (0, 220, 80)
LEFT_STORE = (180, 140, 60)


def _dist_pt_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-6:
        return float(np.linalg.norm(pt - a))
    t = float(((pt - a) @ ab) / denom)
    t = min(max(t, 0.0), 1.0)
    q = a + t * ab
    return float(np.linalg.norm(pt - q))


def infer_entered_from_retirement(tr, boundary, last_cue, p: InterestParams) -> bool:
    """Best-effort conversion rescue when tracking drops near the doorway."""
    if tr.entered or not tr.seen_outside:
        return False
    if len(boundary.entrance_line) != 2:
        return False
    if tr.hits < p.min_hits:
        return False
    if last_cue is None:
        return False
    if last_cue.approach < 0.35 and tr.best_interest < p.entered_backfill_score:
        return False

    a = np.asarray(boundary.entrance_line[0], np.float32)
    b = np.asarray(boundary.entrance_line[1], np.float32)
    foot = np.asarray(tr.foot_px, np.float32)
    h = max(1.0, float(tr.det.height))
    near_line_bh = _dist_pt_segment(foot, a, b) / h
    return near_line_bh <= 0.55


def draw_boundary(vis, boundary) -> None:
    if len(boundary.outside) >= 3:
        cv2.polylines(vis, [np.array(boundary.outside, np.int32)], True, (90, 160, 190), 1)
    if len(boundary.inside) >= 3:
        cv2.polylines(vis, [np.array(boundary.inside, np.int32)], True, (60, 170, 90), 1)
    if len(boundary.entrance_line) == 2:
        a, b = (tuple(map(int, p)) for p in boundary.entrance_line)
        cv2.line(vis, a, b, (60, 120, 255), 2)


def draw_person(vis, tr, cues) -> None:
    x1, y1, x2, y2 = tr.box
    if tr.entered and cues is not None and cues.in_outside:
        # Same track id as before they went in -- a returning/leaving
        # customer, not a fresh interested passer-by (see scoring.py).
        color, tag = LEFT_STORE, "LEFT STORE"
    elif tr.entered:
        color, tag = ENTERED, "ENTERED"
    elif tr.interested:
        color, tag = INTERESTED, "INTERESTED"
    else:
        color, tag = PASSER_BY, ""
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    label = f"ID {tr.track_id}"
    if tag:
        label += f" {tag}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(vis, label, (x1 + 3, y1 - 4), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if cues is not None:
        bar_w = max(24, x2 - x1)
        filled = int(bar_w * float(np.clip(tr.interest_ema, 0, 1)))
        by = min(vis.shape[0] - 3, y2 + 5)
        cv2.rectangle(vis, (x1, by), (x1 + bar_w, by + 4), (60, 60, 60), -1)
        cv2.rectangle(vis, (x1, by), (x1 + filled, by + 4), INTERESTED, -1)
        cv2.putText(
            vis,
            f"score {tr.interest_ema:.2f}  v={cues.speed_bh:.2f}bh/s  "
            f"E={cues.motion_energy:.2f}  trend={cues.motion_trend:+.2f}",
            (x1, by + 16),
            FONT,
            0.34,
            INTERESTED,
            1,
            cv2.LINE_AA,
        )

    if tr.det is not None:
        a = tr.det.attention_vector()
        if a.any():
            p = tr.foot_px
            tip = p + a * max(28.0, tr.det.height * 0.3)
            cv2.arrowedLine(
                vis, (int(p[0]), int(p[1])), (int(tip[0]), int(tip[1])), color, 2, tipLength=0.3
            )


def draw_hud(vis, counts, frame_idx, fps) -> None:
    lines = [
        ("INTEREST PIPELINE  (SNN motion + YOLO posture)", (255, 255, 255)),
        (f"  Total Interested     : {counts['total_interested']}", INTERESTED),
        (f"  Interested Entered   : {counts['interested_entered']}", ENTERED),
        (f"  Interested Passed By : {counts['interested_passed_by']}", PASSER_BY),
    ]
    if counts.get("pending"):
        lines.append((f"  (still in view, unresolved: {counts['pending']})", (150, 150, 150)))
    lines.append((f"  t = {frame_idx / max(fps, 1e-6):6.1f}s   frame {frame_idx}", (150, 150, 150)))

    pad, lh = 10, 19
    w = 430
    h = pad * 2 + lh * len(lines)
    panel = vis[0:h, 0:w].copy()
    vis[0:h, 0:w] = cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0)
    y = pad + 13
    for text, color in lines:
        cv2.putText(vis, text, (pad, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        y += lh


def task_counts(records: dict, live_ids: set[int], p: InterestParams) -> dict:
    counted = [(tid, r) for tid, r in records.items() if r["hits"] >= p.min_hits and r["interested"]]
    entered = sum(1 for _tid, r in counted if r["entered"])
    pending = sum(1 for tid, r in counted if not r["entered"] and tid in live_ids)
    return {
        "total_interested": len(counted),
        "interested_entered": entered,
        "interested_passed_by": len(counted) - entered - pending,
        "pending": pending,
    }


def write_outputs(out_dir: Path, counts: dict, records: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    p1 = out_dir / "interest_summary.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count"])
        w.writerow(["Total Interested", counts["total_interested"]])
        w.writerow(["Interested Entered", counts["interested_entered"]])
        w.writerow(["Interested Passed By", counts["interested_passed_by"]])
    written.append(p1)

    p2 = out_dir / "interest_track_log.csv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["track_id", "hits", "seen_outside", "interested", "interest_frame",
             "best_interest", "entered", "first_frame", "last_frame"]
        )
        for tid, r in sorted(records.items()):
            w.writerow(
                [tid, r["hits"], int(r["seen_outside"]), int(r["interested"]),
                 r["interest_frame"], f"{r['best_interest']:.3f}", int(r["entered"]),
                 r["first_frame"], r["last_frame"]]
            )
    written.append(p2)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "interest")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--preview", action="store_true", help="show a window while running")
    ap.add_argument("--no-video", action="store_true", help="CSV only, skip encoding")
    ap.add_argument("--dump-cues", action="store_true", help="per-frame cue CSV for tuning")
    ap.add_argument("--pose-weights", type=str, default=None, help="YOLO-pose checkpoint (default: yolo11x-pose.pt)")
    ap.add_argument("--pose-imgsz", type=int, default=1280)
    ap.add_argument("--snn-grid", type=int, nargs=2, default=(14, 14), metavar=("H", "W"))
    ap.add_argument("--snn-diff-thresh", type=float, default=10.0)
    ap.add_argument("--snn-vth", type=float, default=0.55)
    ap.add_argument("--snn-tau", type=float, default=3.0)
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    boundary = load_boundary(args.video.name)
    ok, why = boundary.ready()
    if not ok:
        raise SystemExit(
            f"Boundary not configured for {args.video.name}: {why}.\n"
            f"Run:  python3 pipelines/boundary/boundary_gui.py --video {args.video}"
        )

    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    fps_eff = src_fps / stride

    pose = PoseDetector(weights=args.pose_weights, imgsz=int(args.pose_imgsz))
    # New tracks may only spawn on a detection out on the walkway. YOLO still
    # sees everyone in frame (staff, shoppers already inside), but this
    # pipeline never starts tracking someone who only ever appears inside the
    # shop; a track that starts outside and later walks in keeps matching
    # frame-to-frame as usual (see tracker.py / scoring.py docstrings).
    tracker = IoUTracker(spawn_predicate=lambda det: boundary.in_outside(det.foot_point()))
    snn = SNNMotionTracker(
        grid=(int(args.snn_grid[0]), int(args.snn_grid[1])),
        tau=float(args.snn_tau),
        v_th=float(args.snn_vth),
        diff_thresh=float(args.snn_diff_thresh),
    )
    params = InterestParams()

    records: dict[int, dict] = {}
    last_cue_by_id: dict[int, object] = {}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        out_path = args.out_dir / "interest_annotated.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh)
        )

    cue_rows = []
    step = 0
    raw = 0
    t0 = time.time()
    print(
        f"{args.video.name}: {total} frames @ {src_fps:.1f}fps, stride {stride} "
        f"-> {fps_eff:.1f}fps effective. Interest pipeline (YOLO-pose + SNN motion, no ReID)."
    )
    print(
        f"thresholds: score>={params.score_threshold:.2f} for {params.sustain_s:.2f}s, "
        f"attend<={params.attend_deg:.0f}deg, turn_ref={params.turn_deg:.0f}deg/{params.turn_window_s:.2f}s, "
        f"speed slow<={params.slow_bh:.2f}bh/s (walk={params.walk_bh:.2f}), "
        f"approach_ref={params.approach_ref_bh:.2f}bh/s, entered_dwell={params.entered_dwell_s:.2f}s"
    )
    print(
        f"SNN: grid={args.snn_grid[0]}x{args.snn_grid[1]}, diff_thresh={args.snn_diff_thresh:.1f}, "
        f"v_th={args.snn_vth:.2f}, tau={args.snn_tau:.2f}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if (raw - 1) % stride:
            continue
        step += 1

        dets = pose.detect(frame)
        live, retired = tracker.update(step, dets)
        for tr in retired:
            # Rescue conversions lost to tracker drop right at the doorway.
            last_cue = last_cue_by_id.get(tr.track_id)
            if infer_entered_from_retirement(tr, boundary, last_cue, params):
                tr.entered = True
                if (not tr.interested) and tr.best_interest >= params.entered_backfill_score:
                    tr.interested = True
                    tr.interest_frame = tr.last_frame
            records[tr.track_id] = {
                "interested": tr.interested,
                "entered": tr.entered,
                "hits": tr.hits,
                "seen_outside": tr.seen_outside,
                "best_interest": tr.best_interest,
                "interest_frame": tr.interest_frame,
                "first_frame": tr.first_frame,
                "last_frame": tr.last_frame,
            }
            snn.forget(tr.track_id)

        live_ids = {t.track_id for t in live}
        cues_by_id = {}
        for tr in live:
            crop = tr.det.crop(frame)
            snn_out = snn.update(tr.track_id, crop)
            cues = score_track(tr, tr.det, boundary, snn_out, fps_eff, params)
            update_track_state(tr, cues, fps_eff, params)
            cues_by_id[tr.track_id] = cues
            last_cue_by_id[tr.track_id] = cues
            records[tr.track_id] = {
                "interested": tr.interested,
                "entered": tr.entered,
                "hits": tr.hits,
                "seen_outside": tr.seen_outside,
                "best_interest": tr.best_interest,
                "interest_frame": tr.interest_frame,
                "first_frame": tr.first_frame,
                "last_frame": tr.last_frame,
            }
        snn.prune(live_ids)

        counts = task_counts(records, live_ids, params)

        if args.dump_cues:
            for tid, c in cues_by_id.items():
                cue_rows.append(
                    [step, tid, f"{c.orient:.3f}", f"{c.turn:.3f}", f"{c.slow:.3f}",
                     f"{c.approach:.3f}", f"{c.score:.3f}", f"{c.speed_bh:.3f}",
                     f"{c.motion_energy:.3f}", f"{c.motion_trend:.3f}",
                     int(c.in_outside), int(c.in_inside)]
                )

        if writer is not None or args.preview:
            vis = frame.copy()
            draw_boundary(vis, boundary)
            for tr in live:
                cues = cues_by_id.get(tr.track_id)
                # This pipeline's job is the outside/walking area; a person
                # currently inside the shop (still tracked internally so we
                # can tell "returning customer" from "new passer-by" later,
                # see tracker.py's spawn_predicate) is not drawn here.
                if cues is not None and not cues.in_outside:
                    continue
                draw_person(vis, tr, cues)
            draw_hud(vis, counts, step, fps_eff)
            if writer is not None:
                writer.write(vis)
            if args.preview:
                cv2.imshow("interest pipeline (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        if step % 100 == 0:
            rate = step / max(time.time() - t0, 1e-6)
            print(
                f"  frame {raw}/{total}  people {len(live)}  "
                f"interested {counts['total_interested']}  entered {counts['interested_entered']}  "
                f"({rate:.1f} fps)"
            )
        if args.max_frames and step >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    counts = task_counts(records, set(), params)  # nobody is still in view once the clip ends
    paths = write_outputs(args.out_dir, counts, records)

    if args.dump_cues and cue_rows:
        p = args.out_dir / "interest_cues.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["frame", "track_id", "orient", "turn", "slow", "approach", "score",
                 "speed_bh", "motion_energy", "motion_trend", "in_outside", "in_inside"]
            )
            w.writerows(cue_rows)
        paths.append(p)

    print("\n--- Interest pipeline ---")
    print(f"Total Interested     : {counts['total_interested']}")
    print(f"Interested Entered   : {counts['interested_entered']}")
    print(f"Interested Passed By : {counts['interested_passed_by']}")
    print("\nwrote:")
    for p in paths:
        print(" ", p)
    if writer is not None:
        print(" ", args.out_dir / "interest_annotated.mp4")


if __name__ == "__main__":
    main()
