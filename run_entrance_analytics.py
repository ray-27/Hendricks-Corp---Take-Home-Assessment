#!/usr/bin/env python3
"""
Entrance analytics: Task 1 (store interest and conversion) + Task 3 (staff).

    frame -> YOLOv8n-pose -> ReID embeddings -> tracker
          -> interest score (orientation, slowdown, approach, head turn)
          -> entered / passed-by decision
          -> staff vs customer (apron colour votes + ReID staff gallery)
          -> staff-customer interaction sessions

Run run_entrance_setup.py first to draw the zones and enrol staff.

Usage:
    python3 run_entrance_analytics.py
    python3 run_entrance_analytics.py --max-frames 900 --preview
    python3 run_entrance_analytics.py --stride 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analytics.events import EntranceAnalytics  # noqa: E402
from analytics.pose import PoseDetector  # noqa: E402
from analytics.tracker import Tracker  # noqa: E402
from analytics.zones import load_scene  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
CUSTOMER = (200, 200, 200)
INTERESTED = (0, 215, 255)
ENTERED = (0, 220, 80)
STAFF = (255, 140, 0)
SESSION = (255, 80, 255)


def draw_zones(vis, scene) -> None:
    if len(scene.exterior) >= 3:
        cv2.polylines(vis, [np.array(scene.exterior, np.int32)], True, (90, 160, 190), 1)
    if len(scene.interior) >= 3:
        cv2.polylines(vis, [np.array(scene.interior, np.int32)], True, (60, 170, 90), 1)
    if len(scene.entrance_line) == 2:
        a, b = (tuple(map(int, p)) for p in scene.entrance_line)
        cv2.line(vis, a, b, (60, 120, 255), 2)


def draw_person(vis, tr, cue) -> None:
    x1, y1, x2, y2 = tr.box
    if tr.is_staff:
        color, tag = STAFF, "STAFF"
    elif tr.entered:
        color, tag = ENTERED, "ENTERED"
    elif tr.interested:
        color, tag = INTERESTED, "INTERESTED"
    else:
        color, tag = CUSTOMER, ""
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    label = f"ID {tr.track_id}"
    if tag:
        label += f" {tag}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(vis, label, (x1 + 3, y1 - 4), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if cue is not None and not tr.is_staff:
        bar_w = max(24, x2 - x1)
        filled = int(bar_w * float(np.clip(tr.interest_ema, 0, 1)))
        by = min(vis.shape[0] - 3, y2 + 5)
        cv2.rectangle(vis, (x1, by), (x1 + bar_w, by + 4), (60, 60, 60), -1)
        cv2.rectangle(vis, (x1, by), (x1 + filled, by + 4), INTERESTED, -1)
        cv2.putText(
            vis,
            f"{tr.interest_ema:.2f} {cue.speed:.2f}{cue.speed_unit}",
            (x1, by + 16),
            FONT,
            0.36,
            INTERESTED,
            1,
            cv2.LINE_AA,
        )

    # Attention direction: what the interest score is actually reading.
    if tr.det is not None:
        a = tr.det.attention_vector()
        if a.any():
            p = tr.foot_px
            tip = p + a * max(28.0, tr.det.height * 0.3)
            cv2.arrowedLine(
                vis, (int(p[0]), int(p[1])), (int(tip[0]), int(tip[1])), color, 2, tipLength=0.3
            )


def draw_sessions(vis, live_by_id, active) -> None:
    for sid, cid, dur in active:
        s, c = live_by_id.get(sid), live_by_id.get(cid)
        if s is None or c is None:
            continue
        ps = (int((s.box[0] + s.box[2]) * 0.5), int(s.box[3]))
        pc = (int((c.box[0] + c.box[2]) * 0.5), int(c.box[3]))
        cv2.line(vis, ps, pc, SESSION, 2)
        mid = ((ps[0] + pc[0]) // 2, (ps[1] + pc[1]) // 2)
        text = f"staff {sid} <-> cust {cid}  {dur:.1f}s"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
        cv2.rectangle(vis, (mid[0] - 3, mid[1] - th - 4), (mid[0] + tw + 3, mid[1] + 3), (0, 0, 0), -1)
        cv2.putText(vis, text, (mid[0], mid[1]), FONT, 0.42, SESSION, 1, cv2.LINE_AA)


def draw_hud(vis, counts, task3, frame_idx, fps) -> None:
    lines = [
        ("TASK 1  Store Interest and Conversion", (255, 255, 255)),
        (f"  Total Interested        : {counts['total_interested']}", INTERESTED),
        (f"  Interested Entered      : {counts['interested_entered']}", ENTERED),
        (f"  Interested Passed By    : {counts['interested_passed_by']}", CUSTOMER),
    ]
    if counts.get("pending"):
        lines.append((f"  (still in view, unresolved: {counts['pending']})", (150, 150, 150)))
    lines.append(("TASK 3  Staff-Customer Interaction", (255, 255, 255)))
    lines.append(
        (
            f"  Staff instances         : {task3['staff_count']}"
            f"   sessions: {task3['total_sessions']}",
            STAFF,
        )
    )
    for row in task3["staff"][:6]:
        lines.append(
            (f"    {row['staff_instance']:<10} sessions: {row['interaction_sessions']}", STAFF)
        )
    lines.append(
        (f"  Avg sessions per staff  : {task3['average_sessions_per_staff']:.2f}", STAFF)
    )
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


def write_csvs(out_dir: Path, counts, task3, sessions_log) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    p1 = out_dir / "task1_store_interest.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count"])
        w.writerow(["Total Interested", counts["total_interested"]])
        w.writerow(["Interested Entered", counts["interested_entered"]])
        w.writerow(["Interested Passed By", counts["interested_passed_by"]])
    written.append(p1)

    p3 = out_dir / "task3_staff_interactions.csv"
    with p3.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["staff_instance", "interaction_sessions", "first_frame", "last_frame"])
        for row in task3["staff"]:
            w.writerow(
                [
                    row["staff_instance"],
                    row["interaction_sessions"],
                    row["first_frame"],
                    row["last_frame"],
                ]
            )
        w.writerow([])
        w.writerow(["staff_instances", task3["staff_count"]])
        w.writerow(["total_interaction_sessions", task3["total_sessions"]])
        w.writerow(
            ["average_sessions_per_staff", f"{task3['average_sessions_per_staff']:.4f}"]
        )
    written.append(p3)

    p4 = out_dir / "task3_sessions_detail.csv"
    with p4.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["staff_id", "customer_id", "start_frame", "end_frame"])
        for s in sessions_log:
            w.writerow([s["staff_id"], s["customer_id"], s["start_frame"], s["end_frame"]])
    written.append(p4)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--preview", action="store_true", help="show a window while running")
    ap.add_argument("--no-video", action="store_true", help="CSV only, skip encoding")
    ap.add_argument("--dump-cues", action="store_true", help="per-frame cue CSV for tuning")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    scene = load_scene(args.video.name)
    ok, why = scene.ready()
    if not ok:
        raise SystemExit(
            f"Scene not configured for {args.video.name}: {why}.\n"
            f"Run:  python3 run_entrance_setup.py --video {args.video}"
        )
    if not scene.has_homography:
        print("No floor homography: speeds/distances fall back to body-height units.")
    if not scene.apron_hsv and not scene.staff_gallery:
        print("No apron colour and no staff gallery: every person is treated as a customer.")

    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    fps_eff = src_fps / stride

    pose = PoseDetector()
    reid = ReIDEmbedder()
    tracker = Tracker()
    analytics = EntranceAnalytics(scene, fps=fps_eff)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        out_path = args.out_dir / "entrance_annotated.mp4"
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh)
        )

    cue_rows = []
    step = 0
    raw = 0
    t0 = time.time()
    print(
        f"{args.video.name}: {total} frames @ {src_fps:.1f}fps, stride {stride} "
        f"-> {fps_eff:.1f}fps effective. ReID on {reid.device_name}."
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
        crops = [d.crop(frame) for d in dets]
        embs = reid.embed(crops) if crops else np.zeros((0, 256), np.float32)
        live, _retired = tracker.update(step, dets, embs)
        info = analytics.update(step, frame, live)

        live_ids = {t.track_id for t in live}
        counts = analytics.task1_counts(live_ids)
        task3 = analytics.task3_summary()

        if args.dump_cues:
            for tid, c in info.cues.items():
                cue_rows.append(
                    [
                        step, tid, f"{c.orient:.3f}", f"{c.slow:.3f}", f"{c.approach:.3f}",
                        f"{c.head:.3f}", f"{c.score:.3f}", f"{c.speed:.3f}", c.speed_unit,
                        int(c.in_exterior), int(c.in_interior),
                    ]
                )

        if writer is not None or args.preview:
            vis = frame.copy()
            draw_zones(vis, scene)
            for tr in live:
                draw_person(vis, tr, info.cues.get(tr.track_id))
            draw_sessions(vis, {t.track_id: t for t in live}, info.active_sessions)
            draw_hud(vis, counts, task3, step, fps_eff)
            if writer is not None:
                writer.write(vis)
            if args.preview:
                cv2.imshow("entrance analytics (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        if step % 100 == 0:
            rate = step / max(time.time() - t0, 1e-6)
            print(
                f"  frame {raw}/{total}  people {len(live)}  "
                f"interested {counts['total_interested']}  entered {counts['interested_entered']}  "
                f"staff {task3['staff_count']}  sessions {task3['total_sessions']}  "
                f"({rate:.1f} fps)"
            )
        if args.max_frames and step >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    counts = analytics.task1_counts(set())  # nobody is still in view once the clip ends
    task3 = analytics.task3_summary()
    paths = write_csvs(args.out_dir, counts, task3, analytics.sessions_log)

    if args.dump_cues and cue_rows:
        p = args.out_dir / "interest_cues.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["frame", "track_id", "orient", "slow", "approach", "head", "score",
                 "speed", "speed_unit", "in_exterior", "in_interior"]
            )
            w.writerows(cue_rows)
        paths.append(p)

    print("\n--- Task 1 ---")
    print(f"Total Interested     : {counts['total_interested']}")
    print(f"Interested Entered   : {counts['interested_entered']}")
    print(f"Interested Passed By : {counts['interested_passed_by']}")
    print("\n--- Task 3 ---")
    for row in task3["staff"]:
        print(f"{row['staff_instance']}: {row['interaction_sessions']} sessions")
    print(f"staff instances      : {task3['staff_count']}")
    print(f"total sessions       : {task3['total_sessions']}")
    print(f"average per staff    : {task3['average_sessions_per_staff']:.2f}")
    print("\nwrote:")
    for p in paths:
        print(" ", p)
    if writer is not None:
        print(" ", args.out_dir / "entrance_annotated.mp4")


if __name__ == "__main__":
    main()
