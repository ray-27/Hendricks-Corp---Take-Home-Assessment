#!/usr/bin/env python3
"""
Shelf-vector interest pipeline.

Determines shelf interest from explicit geometry instead of a per-frame
distance heuristic:

  - each shelf face has a fixed outward `normal` vector and a `zone`
    polygon (marked once with `shelf_face_gui.py`, stored in
    `pipelines/configs/shelf_faces.json`)
  - each person's facing vector comes from YOLO pose keypoints (head
    vector blended with torso orientation)
  - a person "engages" a face iff their foot point is inside its zone AND
    their facing vector points back toward the shelf (within `face_deg` of
    `-normal`)
  - sustained engagement (`--engage-s`) opens an event; a short lapse
    (`--gap-close-s`) is tolerated without closing it; a close only counts
    if it lasted `--min-event-s`; after closing, the same (person, shelf)
    pair is on `--cooldown-s` cooldown before a new event can open

This pipeline is self-contained: it does not import from
`pipelines/shelf_interest` (see `tracker.py` in this folder for why).

Usage:
    python3 pipelines/shelf_vector_interest/shelf_vector_pipeline.py --video raw_videos/interior.mp4 --preview
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from analytics.pose import L_EAR, L_SHO, NOSE, R_EAR, R_SHO, PoseDetector  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402

from shelf_vector_interest.scoring import (  # noqa: E402
    EngagementTracker,
    VectorParams,
    associate,
)
from shelf_vector_interest.shelf_face_store import face_color, load_shelf_faces  # noqa: E402
from shelf_vector_interest.tracker import PersonTracker  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, np.float32)


def _chest_point(det) -> np.ndarray:
    if det.has(L_SHO, R_SHO):
        return (0.5 * (det.kpts[L_SHO] + det.kpts[R_SHO])).astype(np.float32)
    x1, y1, x2, y2 = det.box
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], np.float32)


def _draw_vec(vis, origin: np.ndarray, vec: np.ndarray, color, length: float = 70.0, thickness: int = 2) -> None:
    if not vec.any():
        return
    tip = origin + _unit(vec) * length
    cv2.arrowedLine(
        vis,
        tuple(map(int, origin)),
        tuple(map(int, tip)),
        color,
        thickness,
        tipLength=0.25,
    )


def _facing_vector(det) -> np.ndarray:
    """Person direction vector: head vector blended with torso orientation.

    Head vector (nose - ear midpoint) is the most direct read of where
    someone is looking; torso orientation is a steadier fallback/blend when
    the head is turned further than the shoulders (e.g. a quick glance).
    """
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
    return det.facing_vector()


def write_outputs(out_dir: Path, faces, events: list, totals: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    p1 = out_dir / "shelf_vector_summary.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shelf_id", "shelf_name", "interest_events"])
        for face in faces:
            w.writerow([face.shelf_id, face.label(), int(totals.get(face.shelf_id, 0))])
    written.append(p1)

    p2 = out_dir / "shelf_vector_events.csv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "track_id", "shelf_id", "shelf_name", "start_frame", "end_frame", "duration_s"])
        for i, e in enumerate(events, start=1):
            w.writerow([i, e.track_id, e.shelf_id, e.shelf_name, e.start_frame, e.end_frame, f"{e.duration_s:.3f}"])
    written.append(p2)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "interior.mp4")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT / "csv" / "shelf_vector_interest",
        help="CSV output folder",
    )
    ap.add_argument(
        "--video-out",
        type=Path,
        default=OUTPUT_ROOT / "interior_annotated.mp4",
        help="annotated mp4 path",
    )
    ap.add_argument("--preview", action="store_true")
    ap.add_argument(
        "--show-person-vector",
        action="store_true",
        help="draw each person's facing vector (cyan) and the shelf's expected facing direction (magenta)",
    )
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)

    ap.add_argument("--pose-weights", type=str, default=None)
    ap.add_argument("--pose-imgsz", type=int, default=1280)

    ap.add_argument("--face-deg", type=float, default=55.0, help="max angle between facing vec and -normal")
    ap.add_argument("--engage-s", type=float, default=1.2, help="sustained engagement to open an event")
    ap.add_argument("--gap-close-s", type=float, default=1.0, help="tolerated disengaged gap before closing")
    ap.add_argument("--cooldown-s", type=float, default=5.0, help="per (person, shelf) cooldown after close")
    ap.add_argument("--min-event-s", type=float, default=0.8, help="minimum duration for a close to count")
    ap.add_argument("--min-hits", type=int, default=4)

    ap.add_argument("--no-reid", action="store_true", help="disable ReID; IoU-only tracking")
    ap.add_argument("--reid-model", type=Path, default=None)
    ap.add_argument("--reid-threshold", type=float, default=0.52)
    ap.add_argument("--reid-relink", type=float, default=0.58)
    ap.add_argument("--reid-weight", type=float, default=0.65)
    ap.add_argument("--max-missed", type=int, default=90)
    ap.add_argument("--retired-ttl-frames", type=int, default=450)
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    layout = load_shelf_faces(args.video.name)
    faces = layout.ready_faces()
    if not faces:
        raise SystemExit(
            "No shelf faces ready. Run:\n"
            "  python3 pipelines/shelf_vector_interest/shelf_face_gui.py --video "
            f"{args.video}\n"
            "and mark edge + normal + zone for at least one shelf face."
        )

    params = VectorParams(
        face_deg=args.face_deg,
        engage_s=args.engage_s,
        gap_close_s=args.gap_close_s,
        cooldown_s=args.cooldown_s,
        min_event_s=args.min_event_s,
        min_hits=args.min_hits,
    )

    cap = cv2.VideoCapture(str(args.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    fps_eff = src_fps / stride

    pose = PoseDetector(weights=args.pose_weights, imgsz=int(args.pose_imgsz))
    use_reid = not args.no_reid
    reid = ReIDEmbedder(model_path=args.reid_model) if use_reid else None
    tracker = PersonTracker(
        max_missed=args.max_missed,
        reid_weight=args.reid_weight,
        reid_threshold=args.reid_threshold,
        relink_similarity=args.reid_relink,
        retired_ttl_frames=args.retired_ttl_frames,
    )
    engagement = EngagementTracker(params, fps=fps_eff)

    writer = None
    if not args.no_video:
        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh))

    events: list = []
    frame_step = 0
    frame_raw = 0
    print(
        f"{args.video.name}: {total} frames @ {src_fps:.1f}fps stride={stride} -> {fps_eff:.1f}fps; "
        f"faces={len(faces)} engage={params.engage_s:.1f}s cooldown={params.cooldown_s:.1f}s reid={use_reid}"
    )

    pbar = tqdm(
        total=total if total > 0 else None,
        desc=args.video.name,
        unit="frame",
        dynamic_ncols=True,
    )
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_raw += 1
        pbar.update(1)
        if (frame_raw - 1) % stride:
            continue
        frame_step += 1

        dets = pose.detect(frame)
        embs = reid.embed([d.crop(frame) for d in dets]) if (use_reid and dets) else None
        live, retired = tracker.update(frame_step, dets, embeddings=embs)

        for tr in retired:
            ev = engagement.close(tr.track_id, frame_step, reason="lost")
            if ev is not None:
                events.append(ev)

        assoc_by_id = {}
        facing_by_id = {}
        for tr in live:
            facing = _facing_vector(tr.det)
            a = associate(tr.foot_px, facing, faces, params)
            assoc_by_id[tr.track_id] = a
            facing_by_id[tr.track_id] = facing
            ev = engagement.update(tr.track_id, tr.hits, a, frame_step)
            if ev is not None:
                events.append(ev)

        if writer is not None or args.preview:
            vis = frame.copy()
            for i, face in enumerate(faces):
                c = face_color(i)
                a, b = (tuple(map(int, p)) for p in face.edge)
                cv2.line(vis, a, b, c, 2)
                mid = face.edge_midpoint()
                tip = mid + face.normal_vec() * 60.0
                cv2.arrowedLine(vis, tuple(map(int, mid)), tuple(map(int, tip)), c, 2, tipLength=0.3)
                cv2.polylines(vis, [np.array(face.zone, np.int32)], True, c, 1)
                cv2.putText(vis, face.label(), a, FONT, 0.5, c, 2)

            for tr in live:
                a = assoc_by_id.get(tr.track_id)
                face_idx = next((i for i, f in enumerate(faces) if a and f.shelf_id == a.shelf_id), None)
                color = (170, 170, 170) if face_idx is None else face_color(face_idx)
                x1, y1, x2, y2 = tr.box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                is_open, open_shelf_id, open_shelf_name = engagement.is_open(tr.track_id)
                tag = f"ID {tr.track_id} "
                tag += open_shelf_name if is_open else (a.shelf_name if a and a.shelf_id else "UNASSIGNED")
                if is_open:
                    dur = engagement.current_duration_s(tr.track_id, frame_step)
                    tag += f" {dur:.1f}s"
                cv2.putText(vis, tag, (x1 + 2, max(16, y1 - 4)), FONT, 0.45, color, 2)

                if a is not None:
                    detail = (
                        f"zone={int(a.in_zone)} ang={a.angle_deg:.0f}/{params.face_deg:.0f} "
                        f"fac={int(a.facing_ok)} eng={int(a.engaged)} sim={tr.similarity:.2f}"
                    )
                    cv2.putText(vis, detail, (x1, min(vis.shape[0] - 4, y2 + 16)), FONT, 0.34, color, 1)

                if args.show_person_vector:
                    origin = _chest_point(tr.det)
                    person_vec = facing_by_id.get(tr.track_id)
                    if person_vec is not None and person_vec.any():
                        vec_color = (0, 220, 0) if (a is not None and a.facing_ok) else (255, 220, 0)
                        _draw_vec(vis, origin, person_vec, vec_color, length=80.0, thickness=2)
                    if a is not None and a.shelf_id is not None:
                        face = next((f for f in faces if f.shelf_id == a.shelf_id), None)
                        if face is not None:
                            # magenta = direction the person must face (toward the shelf = -normal)
                            _draw_vec(vis, origin, -face.normal_vec(), (255, 0, 255), length=55.0, thickness=1)

                # visual association: line from person's foot to the shelf they're engaging
                if is_open and open_shelf_id is not None:
                    face = next((f for f in faces if f.shelf_id == open_shelf_id), None)
                    if face is not None:
                        cv2.line(
                            vis,
                            tuple(map(int, tr.foot_px)),
                            tuple(map(int, face.edge_midpoint())),
                            color,
                            2,
                        )

            totals = engagement.total_counts()
            y = 20
            total_events = int(sum(totals.values()))
            cv2.putText(vis, f"SHELF VECTOR INTEREST  events={total_events}", (10, y), FONT, 0.55, (255, 255, 255), 2)
            y += 20
            for i, face in enumerate(faces):
                c = face_color(i)
                cv2.putText(vis, f"{face.label()}: {int(totals.get(face.shelf_id, 0))}", (10, y), FONT, 0.5, c, 2)
                y += 18

            if writer is not None:
                writer.write(vis)
            if args.preview:
                cv2.imshow("shelf vector interest (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        pbar.set_postfix(people=len(live), events=len(events), refresh=False)

        if args.max_frames and frame_step >= args.max_frames:
            break

    pbar.close()

    for tid in list(engagement.state.keys()):
        ev = engagement.close(tid, frame_step, reason="end")
        if ev is not None:
            events.append(ev)

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    totals = engagement.total_counts()
    paths = write_outputs(args.out_dir, faces, events, totals)
    print("\n--- Shelf vector interest ---")
    for face in faces:
        print(f"{face.label():<20} {int(totals.get(face.shelf_id, 0))}")
    print(f"{'TOTAL':<20} {int(sum(totals.values()))}")
    print("\nwrote:")
    for pth in paths:
        print(" ", pth)
    if writer is not None:
        print(" ", args.video_out)


if __name__ == "__main__":
    main()
