#!/usr/bin/env python3
"""
Run the project pipelines together.

1. Entrance clip — interest + staff-customer interaction in ONE annotated video
   (pose runs once; each task keeps its own tracker/scoring).
2. Interior clip — shelf-vector interest, after the entrance pass.

Does not run `shelf_interest` (polygon/distance). Use `shelf_vector_pipeline.py`.

    python3 run_all.py
    python3 run_all.py --preview
    python3 run_all.py --skip-shelf          # entrance combined video only
    python3 run_all.py --skip-entrance       # shelf-vector only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from analytics.pose import PoseDetector  # noqa: E402
from boundary.boundary_store import load_boundary  # noqa: E402
from interest.scoring import InterestParams, score_track, update_track_state  # noqa: E402
from interest.snn_motion import SNNMotionTracker  # noqa: E402
from interest.tracker import IoUTracker  # noqa: E402
from interest.interest_pipeline import (  # noqa: E402
    draw_boundary,
    draw_hud,
    draw_person,
    infer_entered_from_retirement,
    task_counts,
    write_outputs as write_interest_outputs,
)
from reid.embedder import ReIDEmbedder  # noqa: E402
from staff_interaction.scoring import (  # noqa: E402
    RoleReIDParams,
    RuleInteractionParams,
    RuleInteractionTracker,
    rule_pair_cues,
    update_role_reid,
)
from staff_interaction.staff_interaction_pipeline import (  # noqa: E402
    CANDIDATE_COLOR,
    CUSTOMER_COLOR,
    ENGAGED_COLOR,
    FONT,
    QUERIED_COLOR,
    STAFF_COLOR,
    draw_metric_table,
    write_outputs as write_staff_outputs,
)
from staff_interaction.staff_store import load_staff_marks  # noqa: E402
from staff_interaction.tracker import PersonTracker  # noqa: E402


def _draw_staff_overlay(vis, live, pair_cues, interactions, rule_params, frame_step, staff_records):
    staff = [t for t in live if t.is_staff]
    customers = [t for t in live if not t.is_staff]
    for tr in live:
        x1, y1, x2, y2 = tr.box
        color = STAFF_COLOR if tr.is_staff else CUSTOMER_COLOR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        role_tag = "STAFF" if tr.is_staff else "customer"
        cv2.putText(vis, f"ID {tr.track_id} {role_tag}", (x1 + 2, max(16, y1 - 4)), FONT, 0.5, color, 2)

    for s in staff:
        for c in customers:
            key = (s.track_id, c.track_id)
            cues = pair_cues.get(key)
            if cues is None:
                continue
            is_open = interactions.is_open(s.track_id, c.track_id)
            mid = ((s.foot_px + c.foot_px) * 0.5).astype(int)
            show_line = is_open or cues.dist_bh <= rule_params.near_bh
            if show_line:
                line_color = ENGAGED_COLOR if is_open else (QUERIED_COLOR if cues.engaged else CANDIDATE_COLOR)
                cv2.line(
                    vis,
                    tuple(map(int, s.foot_px)),
                    tuple(map(int, c.foot_px)),
                    line_color,
                    2 if is_open else 1,
                )
            if is_open:
                dur = interactions.current_duration_s(s.track_id, c.track_id, frame_step)
                cv2.putText(vis, f"{dur:.1f}s", tuple(mid), FONT, 0.45, ENGAGED_COLOR, 2)
            elif show_line:
                cv2.putText(vis, f"{cues.score:.2f}", tuple(mid), FONT, 0.42, QUERIED_COLOR, 2)

    return draw_metric_table(vis, staff_records, interactions.summary())


def run_entrance_combined(
    video: Path,
    out_dir: Path,
    *,
    preview: bool,
    no_video: bool,
    stride: int,
    max_frames: int,
    pose_weights: str | None,
    pose_imgsz: int,
) -> None:
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")

    boundary = load_boundary(video.name)
    ok, why = boundary.ready()
    if not ok:
        raise SystemExit(
            f"Boundary not configured for {video.name}: {why}.\n"
            f"Run:  python3 pipelines/boundary/boundary_gui.py --video {video}"
        )

    marks = load_staff_marks(video.name)
    ok, why = marks.ready()
    if not ok:
        raise SystemExit(
            f"Staff gallery missing for {video.name} ({why}). Run:\n"
            f"  python3 pipelines/staff_interaction/staff_gui.py --video {video}"
        )
    gallery = marks.gallery_matrix()

    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, stride)
    fps_eff = src_fps / stride

    pose = PoseDetector(weights=pose_weights, imgsz=int(pose_imgsz))
    reid = ReIDEmbedder()
    interest_tracker = IoUTracker(spawn_predicate=lambda det: boundary.in_outside(det.foot_point()))
    snn = SNNMotionTracker(grid=(14, 14), tau=3.0, v_th=0.55, diff_thresh=10.0)
    interest_params = InterestParams()

    staff_tracker = PersonTracker(max_missed=45, retired_ttl_frames=90)
    role_reid_params = RoleReIDParams()
    rule_params = RuleInteractionParams()
    interactions = RuleInteractionTracker(rule_params, fps=fps_eff)

    interest_out = ROOT / "pipelines" / "outputs" / "interest"
    staff_out = ROOT / "pipelines" / "outputs" / "staff_interaction"
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if not no_video:
        out_path = out_dir / "entrance_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh))

    interest_records: dict[int, dict] = {}
    last_cue_by_id: dict[int, object] = {}
    sessions: list = []
    staff_records: dict[int, dict] = {}

    frame_step = 0
    frame_raw = 0
    print(
        f"\n=== Entrance (interest + staff) ===\n"
        f"{video.name}: {total} frames @ {src_fps:.1f}fps stride={stride} -> {fps_eff:.1f}fps"
    )
    print(f"Loaded staff gallery: {gallery.shape[0]} embedding(s)")

    pbar = tqdm(total=total if total > 0 else None, desc=video.name, unit="frame", dynamic_ncols=True)
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
        embs = reid.embed([d.crop(frame) for d in dets]) if dets else None

        # --- interest ---
        i_live, i_retired = interest_tracker.update(frame_step, dets)
        for tr in i_retired:
            last_cue = last_cue_by_id.get(tr.track_id)
            if infer_entered_from_retirement(tr, boundary, last_cue, interest_params):
                tr.entered = True
                if (not tr.interested) and tr.best_interest >= interest_params.entered_backfill_score:
                    tr.interested = True
                    tr.interest_frame = tr.last_frame
            interest_records[tr.track_id] = {
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

        live_ids = {t.track_id for t in i_live}
        cues_by_id = {}
        for tr in i_live:
            snn_out = snn.update(tr.track_id, tr.det.crop(frame))
            cues = score_track(tr, tr.det, boundary, snn_out, fps_eff, interest_params)
            update_track_state(tr, cues, fps_eff, interest_params)
            cues_by_id[tr.track_id] = cues
            last_cue_by_id[tr.track_id] = cues
            interest_records[tr.track_id] = {
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
        counts = task_counts(interest_records, live_ids, interest_params)

        # --- staff ---
        s_live, s_retired = staff_tracker.update(frame_step, dets, embeddings=embs)
        for tr in s_retired:
            sessions.extend(interactions.close_all_for(tr.track_id, frame_step))

        for tr in s_live:
            update_role_reid(tr, gallery, role_reid_params)
            if tr.is_staff:
                rec = staff_records.setdefault(
                    tr.track_id, {"first_frame": tr.first_frame, "last_frame": tr.last_frame}
                )
                rec["last_frame"] = tr.last_frame
                interactions.register_staff(tr.track_id)

        staff = [t for t in s_live if t.is_staff]
        customers = [t for t in s_live if not t.is_staff]
        pair_cues = {}
        for s in staff:
            for c in customers:
                cues = rule_pair_cues(s, c, rule_params)
                pair_cues[(s.track_id, c.track_id)] = cues
                ev = interactions.update(s, c, cues, frame_step)
                if ev is not None:
                    sessions.append(ev)

        if writer is not None or preview:
            vis = frame.copy()
            draw_boundary(vis, boundary)
            for tr in i_live:
                cues = cues_by_id.get(tr.track_id)
                if cues is not None and not cues.in_outside:
                    continue
                draw_person(vis, tr, cues)
            vis = _draw_staff_overlay(
                vis, s_live, pair_cues, interactions, rule_params, frame_step, staff_records
            )
            draw_hud(vis, counts, frame_step, fps_eff)
            if writer is not None:
                writer.write(vis)
            if preview:
                cv2.imshow("entrance: interest + staff (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        pbar.set_postfix(
            interested=counts["total_interested"],
            staff=len(staff_records),
            sessions=len(sessions),
            refresh=False,
        )
        if max_frames and frame_step >= max_frames:
            break

    pbar.close()
    sessions.extend(interactions.close_end(frame_step))
    cap.release()
    if writer is not None:
        writer.release()
    if preview:
        cv2.destroyAllWindows()

    counts = task_counts(interest_records, set(), interest_params)
    i_paths = write_interest_outputs(interest_out, counts, interest_records)

    summary = interactions.summary()
    for sid in staff_records:
        summary.setdefault(sid, 0)
    s_paths, rows, total_sessions, avg = write_staff_outputs(staff_out, staff_records, summary, sessions)

    print("\n--- Interest ---")
    print(f"Total Interested     : {counts['total_interested']}")
    print(f"Interested Entered   : {counts['interested_entered']}")
    print(f"Interested Passed By : {counts['interested_passed_by']}")
    print("\n--- Staff interaction ---")
    for r in rows:
        print(f"{r['staff_instance']:<14} sessions={r['interaction_sessions']}")
    print(f"{'average/staff':<14} {avg:.3f}")
    print("\nwrote:")
    for p in i_paths + s_paths:
        print(" ", p)
    if writer is not None:
        print(" ", out_dir / "entrance_annotated.mp4")


def run_shelf_vector(
    video: Path,
    *,
    preview: bool,
    no_video: bool,
    stride: int,
    max_frames: int,
    pose_weights: str | None,
    pose_imgsz: int,
    show_person_vector: bool,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "pipelines" / "shelf_vector_interest" / "shelf_vector_pipeline.py"),
        "--video",
        str(video),
        "--stride",
        str(stride),
        "--pose-imgsz",
        str(pose_imgsz),
    ]
    if pose_weights:
        cmd += ["--pose-weights", pose_weights]
    if preview:
        cmd.append("--preview")
    if no_video:
        cmd.append("--no-video")
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    if show_person_vector:
        cmd.append("--show-person-vector")
    print(f"\n=== Interior (shelf vector interest) ===\n{' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc:
        raise SystemExit(rc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run interest+staff (one video) then shelf-vector interest.")
    ap.add_argument("--entrance-video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    ap.add_argument("--interior-video", type=Path, default=ROOT / "raw_videos" / "interior.mp4")
    ap.add_argument(
        "--combined-out-dir",
        type=Path,
        default=ROOT / "pipelines" / "outputs" / "combined",
    )
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--pose-weights", type=str, default=None)
    ap.add_argument("--pose-imgsz", type=int, default=1280)
    ap.add_argument("--show-person-vector", action="store_true", help="forwarded to shelf-vector pipeline")
    ap.add_argument("--skip-entrance", action="store_true", help="only run shelf-vector")
    ap.add_argument("--skip-shelf", action="store_true", help="only run combined entrance")
    args = ap.parse_args()

    if not args.skip_entrance:
        run_entrance_combined(
            args.entrance_video,
            args.combined_out_dir,
            preview=args.preview,
            no_video=args.no_video,
            stride=args.stride,
            max_frames=args.max_frames,
            pose_weights=args.pose_weights,
            pose_imgsz=args.pose_imgsz,
        )

    if args.skip_shelf:
        return

    run_shelf_vector(
        args.interior_video,
        preview=args.preview,
        no_video=args.no_video,
        stride=args.stride,
        max_frames=args.max_frames,
        pose_weights=args.pose_weights,
        pose_imgsz=args.pose_imgsz,
        show_person_vector=args.show_person_vector,
    )


if __name__ == "__main__":
    main()
