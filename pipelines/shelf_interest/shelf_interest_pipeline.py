#!/usr/bin/env python3
"""
Per-shelf customer interest pipeline.

This keeps the same philosophy as `pipelines/interest/`: lightweight, explainable
geometry + pose cues, no VLM required. It tracks people in the interior and
produces per-shelf interest episodes with de-duplication over continuous time.
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
from boundary.shelf_store import load_shelf_layout, shelf_color  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402
from shelf_interest.scoring import ShelfInterestParams, close_event, score_track, update_track_state  # noqa: E402
from shelf_interest.tracker import IoUTracker  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _inside_shop_strict(det, store_boundary) -> bool:
    """Strict interior gate so outside-edge people are not tracked."""
    foot = det.foot_point()
    cx = 0.5 * (det.box[0] + det.box[2])
    cy = 0.5 * (det.box[1] + det.box[3])
    center = np.array([cx, cy], np.float32)
    # Require both standing point and bbox center inside interior.
    return store_boundary.in_inside(foot) and store_boundary.in_inside(center)


def draw_geometry(vis, store_boundary, shelf_layout) -> None:
    if len(store_boundary.inside) >= 3:
        cv2.polylines(vis, [np.array(store_boundary.inside, np.int32)], True, (60, 170, 90), 1)
    for i, sh in enumerate(shelf_layout.ready_shelves()):
        c = shelf_color(i)
        poly = np.array(sh.polygon, np.int32)
        cv2.polylines(vis, [poly], True, c, 2)
        cv2.putText(vis, sh.label(), tuple(poly[0]), FONT, 0.5, c, 2, cv2.LINE_AA)
        if len(sh.front_line) == 2:
            a, b = (tuple(map(int, p)) for p in sh.front_line)
            cv2.line(vis, a, b, c, 2)


def draw_track(vis, tr, cues, shelf_layout, counts_by_shelf: dict, fps_eff: float) -> None:
    x1, y1, x2, y2 = tr.box
    sid = cues.shelf_id if cues is not None else tr.shelf_id
    shelf = shelf_layout.shelf_by_id(sid) if sid else None
    color = (170, 170, 170)
    if shelf is not None:
        ridx = next((i for i, s in enumerate(shelf_layout.ready_shelves()) if s.shelf_id == shelf.shelf_id), 0)
        color = shelf_color(ridx)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    label = f"ID {tr.track_id}"
    if shelf is not None:
        label += f" {shelf.label()}"
    else:
        label += " UNASSIGNED"
    if tr.event_open:
        label += " INTEREST"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(vis, label, (x1 + 3, y1 - 4), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if cues is not None:
        cv2.putText(
            vis,
            f"score={tr.interest_ema:.2f} orient={cues.orient:.2f} prox={cues.proximity:.2f} d={cues.dist_bh:.2f}bh slow={cues.slow:.2f} reid={tr.similarity:.2f}",
            (x1, min(vis.shape[0] - 4, y2 + 16)),
            FONT,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    if tr.event_open and tr.event_start_frame is not None:
        dur_s = (tr.last_frame - tr.event_start_frame + 1) / max(fps_eff, 1e-6)
        cv2.putText(
            vis,
            f"dur={dur_s:.1f}s",
            (x1, min(vis.shape[0] - 4, y2 + 30)),
            FONT,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )
    if shelf is not None:
        cnt = counts_by_shelf.get(shelf.shelf_id, 0)
        cv2.putText(
            vis,
            f"events={cnt}",
            (x1, min(vis.shape[0] - 4, y2 + 44)),
            FONT,
            0.34,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_hud(vis, shelf_layout, counts_by_shelf: dict, frame_idx: int, fps_eff: float) -> None:
    lines = [("SHELF INTEREST PIPELINE  (YOLO pose + geometry)", (255, 255, 255))]
    total = 0
    for i, sh in enumerate(shelf_layout.ready_shelves()):
        c = shelf_color(i)
        n = int(counts_by_shelf.get(sh.shelf_id, 0))
        total += n
        lines.append((f"  {sh.label():<18} : {n}", c))
    lines.append((f"  TOTAL EVENTS        : {total}", (230, 230, 230)))
    lines.append((f"  t={frame_idx / max(fps_eff, 1e-6):6.1f}s frame={frame_idx}", (150, 150, 150)))

    pad, lh = 10, 18
    w = 510
    h = pad * 2 + lh * len(lines)
    panel = vis[0:h, 0:w].copy()
    vis[0:h, 0:w] = cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0)
    y = pad + 13
    for text, color in lines:
        cv2.putText(vis, text, (pad, y), FONT, 0.44, color, 1, cv2.LINE_AA)
        y += lh


def write_outputs(out_dir: Path, shelf_layout, counts_by_shelf: dict, events: list, tracks: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    p1 = out_dir / "shelf_interest_summary.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shelf_id", "shelf_name", "interest_events"])
        for sh in shelf_layout.ready_shelves():
            w.writerow([sh.shelf_id, sh.label(), int(counts_by_shelf.get(sh.shelf_id, 0))])
    written.append(p1)

    p2 = out_dir / "shelf_interest_events.csv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "track_id", "shelf_id", "shelf_name", "start_frame", "end_frame", "duration_s", "end_reason"])
        for i, ev in enumerate(events, start=1):
            sh = shelf_layout.shelf_by_id(ev["shelf_id"])
            w.writerow(
                [
                    i,
                    ev["track_id"],
                    ev["shelf_id"],
                    sh.label() if sh is not None else ev["shelf_id"],
                    ev["start_frame"],
                    ev["end_frame"],
                    f"{ev['duration_s']:.3f}",
                    ev.get("reason", ""),
                ]
            )
    written.append(p2)

    p3 = out_dir / "shelf_interest_track_log.csv"
    with p3.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "hits", "best_interest", "counts_by_shelf"])
        for tid, tr in sorted(tracks.items()):
            compact = ";".join([f"{k}:{int(v)}" for k, v in sorted(tr.event_counts.items())])
            w.writerow([tid, tr.hits, f"{tr.best_interest:.3f}", compact])
    written.append(p3)

    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "interior.mp4")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "shelf_interest",
    )
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--pose-weights", type=str, default=None, help="optional pose model path (e.g. yolov8x-pose/yolo11-pose)")
    ap.add_argument("--pose-imgsz", type=int, default=1280, help="pose inference resolution")
    ap.add_argument("--no-reid", action="store_true", help="disable ReID-assisted identity continuity")
    ap.add_argument("--reid-model", type=Path, default=None, help="optional ONNX path for ReID model")
    ap.add_argument("--reid-threshold", type=float, default=0.52, help="fused ReID/IoU match threshold")
    ap.add_argument("--reid-relink", type=float, default=0.58, help="min cosine similarity for relinking missed IDs")
    ap.add_argument("--reid-weight", type=float, default=0.65, help="appearance weight in fused ReID/IoU matching")
    ap.add_argument("--max-missed", type=int, default=90, help="frames to keep unmatched track before retire")
    ap.add_argument("--relink-max-dist-bh", type=float, default=6.0, help="max relink distance in body heights")
    ap.add_argument(
        "--retired-ttl-frames",
        type=int,
        default=450,
        help="how long retired IDs stay re-linkable by ReID",
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    store_boundary = load_boundary(args.video.name)
    layout = load_shelf_layout(args.video.name)
    ok_store = len(store_boundary.inside) >= 3
    why_store = "inside/shop area not drawn" if not ok_store else "ok"
    ok_shelf, why_shelf = layout.ready()
    ok = ok_store and ok_shelf
    why = why_store if not ok_store else why_shelf
    if not ok:
        raise SystemExit(
            f"Geometry not ready for shelf interest on {args.video.name}: {why}.\n"
            f"Run both:\n"
            f"  python3 pipelines/boundary/boundary_gui.py --video {args.video}\n"
            f"  python3 pipelines/boundary/shelf_gui.py --video {args.video}"
        )

    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    fps_eff = src_fps / stride

    pose = PoseDetector(weights=args.pose_weights, imgsz=int(args.pose_imgsz))
    tracker = IoUTracker(
        max_missed=args.max_missed,
        reid_weight=args.reid_weight,
        reid_threshold=args.reid_threshold,
        relink_similarity=args.reid_relink,
        relink_max_dist_bh=args.relink_max_dist_bh,
        retired_ttl_frames=args.retired_ttl_frames,
    )
    params = ShelfInterestParams()
    reid = None
    if not args.no_reid:
        try:
            reid = ReIDEmbedder(model_path=args.reid_model)
        except Exception as exc:
            print(f"ReID disabled ({exc.__class__.__name__}: {exc})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        out_path = args.out_dir / "shelf_interest_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh))

    step = 0
    raw = 0
    t0 = time.time()
    counts_by_shelf = {s.shelf_id: 0 for s in layout.ready_shelves()}
    all_events = []
    closed_tracks = {}

    print(
        f"{args.video.name}: {total} frames @ {src_fps:.1f}fps stride={stride} -> {fps_eff:.1f}fps"
    )
    if reid is None:
        print("running shelf-interest (YOLO-pose + geometry, IoU only, no VLM)")
    else:
        print("running shelf-interest (YOLO-pose + geometry + ReID continuity, no VLM)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw += 1
        if (raw - 1) % stride:
            continue
        step += 1

        dets_all = pose.detect(frame)
        dets = [d for d in dets_all if _inside_shop_strict(d, store_boundary)]

        embs = None
        if reid is not None and dets:
            crops = [d.crop(frame) for d in dets]
            embs = reid.embed(crops)
        live, retired = tracker.update(step, dets, embeddings=embs)
        for tr in retired:
            ev = close_event(
                tr,
                tr.last_frame,
                fps_eff,
                reason="retired",
                min_duration_s=params.min_event_duration_s,
            )
            if ev is not None:
                all_events.append(ev)
            closed_tracks[tr.track_id] = tr

        cues_by_id = {}
        for tr in live:
            cues = score_track(tr, tr.det, layout, fps_eff, params)
            ev = update_track_state(tr, cues, fps_eff, params)
            if ev is not None:
                all_events.append(ev)
            cues_by_id[tr.track_id] = cues

        counts_by_shelf = {s.shelf_id: 0 for s in layout.ready_shelves()}
        for tr in list(closed_tracks.values()) + live:
            for sid, n in tr.event_counts.items():
                counts_by_shelf[sid] = counts_by_shelf.get(sid, 0) + int(n)

        if writer is not None or args.preview:
            vis = frame.copy()
            draw_geometry(vis, store_boundary, layout)
            for tr in live:
                draw_track(vis, tr, cues_by_id.get(tr.track_id), layout, counts_by_shelf, fps_eff)
            draw_hud(vis, layout, counts_by_shelf, step, fps_eff)
            if writer is not None:
                writer.write(vis)
            if args.preview:
                cv2.imshow("shelf interest (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        if step % 120 == 0:
            rate = step / max(time.time() - t0, 1e-6)
            print(f"  frame {raw}/{total}  tracks {len(live)}  total_events {sum(counts_by_shelf.values())}  ({rate:.1f} fps)")
        if args.max_frames and step >= args.max_frames:
            break

    for tr in live:
        ev = close_event(
            tr,
            tr.last_frame,
            fps_eff,
            reason="video_end",
            min_duration_s=params.min_event_duration_s,
        )
        if ev is not None:
            all_events.append(ev)
        closed_tracks[tr.track_id] = tr

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    paths = write_outputs(args.out_dir, layout, counts_by_shelf, all_events, closed_tracks)
    print("\n--- Shelf interest pipeline ---")
    for sh in layout.ready_shelves():
        print(f"{sh.label():<20} {int(counts_by_shelf.get(sh.shelf_id, 0))}")
    print(f"{'TOTAL':<20} {sum(counts_by_shelf.values())}")
    print("\nwrote:")
    for p in paths:
        print(" ", p)
    if writer is not None:
        print(" ", args.out_dir / "shelf_interest_annotated.mp4")


if __name__ == "__main__":
    main()
