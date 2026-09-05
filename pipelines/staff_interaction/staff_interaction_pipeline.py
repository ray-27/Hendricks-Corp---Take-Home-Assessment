#!/usr/bin/env python3
"""
Staff-customer interaction pipeline (Task 3), for entrance.mp4.

Two independent judgments, each with a non-VLM (default) and a VLM
implementation selectable per-run (see `scoring.py`'s module docstring for
the full reasoning behind the defaults):

  1. staff vs customer, per track.
       --role-method reid (default)  cosine similarity between the track's
                                      running ReID embedding (already
                                      computed for tracking) and a small
                                      gallery enrolled once via
                                      `staff_gui.py`. Requires that
                                      one-time setup step.
       --role-method vlm             a VLM (Qwen2-VL-2B) asked "is this a
                                      staff apron?" per track crop. No
                                      setup step, but slower and, on this
                                      footage, less reliable than the
                                      enrolled-gallery approach.
  2. is this (staff, customer) pair actively interacting.
       --interaction-method rule (default)  a weighted proximity +
                                             mutual-facing score from pose
                                             keypoints already extracted
                                             for tracking. No model call.
       --interaction-method vlm             a VLM asked "are these two
                                             people interacting?" per pair,
                                             throttled by a cooldown.

Both trackers/classifiers share the same open/gap/cooldown/min-event
session-state-machine shape either way -- only the per-frame/per-query
signal that drives it differs.

Self-contained: has its own `tracker.py` (IoU + optional ReID for identity
continuity only -- unrelated to role) and does not import from any sibling
pipeline.

Usage:
    # one-time setup for the default ReID role method:
    python3 pipelines/staff_interaction/staff_gui.py --video raw_videos/entrance.mp4

    python3 pipelines/staff_interaction/staff_interaction_pipeline.py --video raw_videos/entrance.mp4 --preview

    # fully rule-based, no setup, no VLM at all:
    python3 pipelines/staff_interaction/staff_interaction_pipeline.py --video raw_videos/entrance.mp4 \\
        --role-method vlm --interaction-method rule
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from analytics.pose import PoseDetector  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402

from staff_interaction.scoring import (  # noqa: E402
    InteractionParams,
    InteractionTracker,
    RoleParams,
    RoleReIDParams,
    RuleInteractionParams,
    RuleInteractionTracker,
    rule_pair_cues,
    update_role,
    update_role_reid,
)
from staff_interaction.staff_store import load_staff_marks  # noqa: E402
from staff_interaction.tracker import PersonTracker  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
STAFF_COLOR = (40, 210, 140)
CUSTOMER_COLOR = (170, 170, 170)
ENGAGED_COLOR = (40, 140, 255)
QUERIED_COLOR = (0, 220, 255)
CANDIDATE_COLOR = (90, 90, 90)


def draw_metric_table(vis, staff_records: dict, summary: dict):
    """Live version of the brief's own metric-calculation example:

        Staff instance  Interaction sessions
        Staff 1         5
        Staff 2         4
        Staff 3         0
        Average         (5 + 4 + 0) / 3 = 3.0

    `staff_records` insertion order is first-seen order (Python dict order),
    so "Staff 1", "Staff 2", ... here matches the order staff instances were
    first detected in this run, not raw track IDs. Every staff instance
    detected so far is listed, including ones with 0 sessions, and the
    average updates live -- this panel is exactly the number reported in
    `staff_interaction_summary.csv` once the video ends.
    """
    ids = list(staff_records.keys())
    rows = [(f"Staff {i + 1}", int(summary.get(tid, 0))) for i, tid in enumerate(ids)]
    total = sum(c for _, c in rows)
    avg = (total / len(rows)) if rows else 0.0

    header = "Staff instance   Sessions"
    lines = [header]
    lines += [f"{name:<15} {count}" for name, count in rows] if rows else ["  (none detected yet)"]
    lines.append("-" * len(header))
    lines.append(f"Average          {avg:.2f}")

    pad, line_h = 8, 20
    text_w = max(cv2.getTextSize(l, FONT, 0.5, 1)[0][0] for l in lines)
    w, h = text_w + pad * 2, line_h * len(lines) + pad * 2
    x0, y0 = vis.shape[1] - w - 10, 10

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
    vis = cv2.addWeighted(overlay, 0.70, vis, 0.30, 0)

    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 6
        color = (0, 220, 255) if i == 0 else (255, 255, 255)
        if line.startswith("Average"):
            color = (40, 210, 140)
        cv2.putText(vis, line, (x0 + pad, y), FONT, 0.5, color, 1, cv2.LINE_AA)
    return vis


def write_outputs(out_dir: Path, staff_records: dict, summary: dict, sessions: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    rows = []
    for sid, count in sorted(summary.items()):
        rec = staff_records.get(sid, {})
        rows.append(
            {
                "staff_instance": f"staff_{sid}",
                "track_id": sid,
                "interaction_sessions": count,
                "first_frame": rec.get("first_frame", ""),
                "last_frame": rec.get("last_frame", ""),
            }
        )
    total = sum(r["interaction_sessions"] for r in rows)
    avg = (total / len(rows)) if rows else 0.0

    p1 = out_dir / "staff_interaction_summary.csv"
    with p1.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["staff_instance", "track_id", "interaction_sessions", "first_frame", "last_frame"])
        for r in rows:
            w.writerow([r["staff_instance"], r["track_id"], r["interaction_sessions"], r["first_frame"], r["last_frame"]])
        w.writerow([])
        w.writerow(["staff_count", len(rows)])
        w.writerow(["total_sessions", total])
        w.writerow(["average_sessions_per_staff", f"{avg:.4f}"])
    written.append(p1)

    p2 = out_dir / "staff_interaction_sessions.csv"
    with p2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "staff_id", "customer_id", "start_frame", "end_frame", "duration_s", "reason"])
        for i, ev in enumerate(sessions, start=1):
            w.writerow([i, ev.staff_id, ev.customer_id, ev.start_frame, ev.end_frame, f"{ev.duration_s:.3f}", ev.reason])
    written.append(p2)

    return written, rows, total, avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    ap.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "staff_interaction"
    )
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)

    ap.add_argument("--pose-weights", type=str, default=None, help="YOLO-pose checkpoint (default: yolo11x-pose.pt)")
    ap.add_argument("--pose-imgsz", type=int, default=1280)

    ap.add_argument(
        "--role-method",
        choices=["reid", "vlm"],
        default="reid",
        help="reid (default): cosine match against a gallery enrolled once via staff_gui.py. "
        "vlm: ask a VLM per track crop, no setup step needed but slower/less reliable here.",
    )
    ap.add_argument(
        "--interaction-method",
        choices=["rule", "vlm"],
        default="rule",
        help="rule (default): weighted proximity + mutual-facing score from pose keypoints, no model call. "
        "vlm: ask a VLM per pair, throttled by a cooldown.",
    )

    # role: reid (default)
    ap.add_argument("--role-sim-threshold", type=float, default=0.62, help="cosine similarity to latch staff")
    ap.add_argument("--role-reid-min-votes", type=int, default=3)
    ap.add_argument("--role-reid-min-ratio", type=float, default=0.55)
    ap.add_argument("--role-reid-min-checks", type=int, default=5)

    # role: vlm (opt-in)
    ap.add_argument("--vlm-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--vlm-max-side", type=int, default=448, help="crops are downsized to this before the VLM call")
    ap.add_argument(
        "--staff-description",
        type=str,
        default="",
        help="optional free-text description of what staff aprons actually look like in this "
        "footage -- appended to the VLM role prompt; only used with --role-method vlm",
    )
    ap.add_argument("--role-vlm-min-votes", type=int, default=2)
    ap.add_argument("--role-vlm-min-ratio", type=float, default=0.60)
    ap.add_argument("--role-vlm-max-checks", type=int, default=4)
    ap.add_argument("--role-vlm-query-cooldown-s", type=float, default=1.0)
    ap.add_argument("--role-vlm-recheck-cooldown-s", type=float, default=6.0)

    # interaction: rule (default)
    ap.add_argument("--near-bh", type=float, default=1.45, help="conversational distance, body-heights")
    ap.add_argument("--very-near-bh", type=float, default=0.75, help="this close, skip the facing requirement")
    ap.add_argument("--face-deg", type=float, default=75.0, help="facing-cone half-angle, degrees")
    ap.add_argument("--w-prox", type=float, default=0.55, help="weight of the proximity cue")
    ap.add_argument("--w-face", type=float, default=0.45, help="weight of the facing cue")
    ap.add_argument("--score-threshold", type=float, default=0.50, help="blended score to count as engaged")
    ap.add_argument("--open-s", type=float, default=1.00, help="sustained engagement required to open a session")
    ap.add_argument("--gap-close-s", type=float, default=1.50, help="tolerated disengaged gap before closing")

    # interaction: vlm (opt-in)
    ap.add_argument("--far-bh", type=float, default=2.60, help="force-close distance, body-heights (vlm only)")
    ap.add_argument("--query-cooldown-s", type=float, default=2.50, help="spacing between VLM queries per pair")
    ap.add_argument("--retry-cooldown-s", type=float, default=1.00, help="retry spacing after an unparseable answer")
    ap.add_argument("--open-confirmations", type=int, default=1)
    ap.add_argument("--close-confirmations", type=int, default=2)

    # interaction: shared by both methods
    ap.add_argument("--min-event-s", type=float, default=1.20)
    ap.add_argument("--cooldown-s", type=float, default=3.00, help="per (staff, customer) cooldown after a close")
    ap.add_argument("--min-hits", type=int, default=4)

    # tracker (IoU + optional ReID) -- identity continuity only, unrelated to role
    ap.add_argument(
        "--no-reid",
        action="store_true",
        help="disable ReID for IDENTITY TRACKING (IoU-only); incompatible with --role-method reid, "
        "which needs the ReID embedding to classify staff",
    )
    ap.add_argument("--reid-model", type=Path, default=None)
    ap.add_argument("--reid-threshold", type=float, default=0.52)
    ap.add_argument("--reid-relink", type=float, default=0.58)
    ap.add_argument("--reid-weight", type=float, default=0.65)
    ap.add_argument("--max-missed", type=int, default=45)
    ap.add_argument(
        "--retired-ttl-frames",
        type=int,
        default=90,
        help="short on purpose -- bridges brief occlusion only; the brief allows treating a staff "
        "member who fully leaves and re-enters view as a new instance",
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    if args.no_reid and args.role_method == "reid":
        raise SystemExit("--no-reid disables the ReID embedding that --role-method reid needs; drop one of them.")

    gallery = None
    if args.role_method == "reid":
        marks = load_staff_marks(args.video.name)
        ok, why = marks.ready()
        if not ok:
            raise SystemExit(
                f"--role-method reid needs an enrolled staff gallery for {args.video.name} ({why}). Run:\n"
                f"  python3 pipelines/staff_interaction/staff_gui.py --video {args.video}\n"
                f"...then click each staff member once, and Save. Or pass --role-method vlm instead."
            )
        gallery = marks.gallery_matrix()
        print(f"Loaded staff gallery: {gallery.shape[0]} embedding(s) for {args.video.name}")

    role_reid_params = RoleReIDParams(
        similarity_threshold=args.role_sim_threshold,
        min_votes=args.role_reid_min_votes,
        min_ratio=args.role_reid_min_ratio,
        min_checks=args.role_reid_min_checks,
    )
    role_vlm_params = RoleParams(
        min_votes=args.role_vlm_min_votes,
        min_ratio=args.role_vlm_min_ratio,
        max_checks=args.role_vlm_max_checks,
        role_query_cooldown_s=args.role_vlm_query_cooldown_s,
        recheck_cooldown_s=args.role_vlm_recheck_cooldown_s,
    )
    rule_interaction_params = RuleInteractionParams(
        near_bh=args.near_bh,
        very_near_bh=args.very_near_bh,
        face_deg=args.face_deg,
        w_prox=args.w_prox,
        w_face=args.w_face,
        score_threshold=args.score_threshold,
        open_s=args.open_s,
        gap_close_s=args.gap_close_s,
        cooldown_s=args.cooldown_s,
        min_event_s=args.min_event_s,
        min_hits=args.min_hits,
    )
    vlm_interaction_params = InteractionParams(
        near_bh=args.near_bh,
        far_bh=args.far_bh,
        query_cooldown_s=args.query_cooldown_s,
        retry_cooldown_s=args.retry_cooldown_s,
        open_confirmations=args.open_confirmations,
        close_confirmations=args.close_confirmations,
        min_event_s=args.min_event_s,
        cooldown_s=args.cooldown_s,
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
    # ReID embeddings are needed both for identity tracking (unless --no-reid)
    # and for the reid role classifier -- either reason is enough to compute them.
    use_reid = (not args.no_reid) or (args.role_method == "reid")
    reid = ReIDEmbedder(model_path=args.reid_model) if use_reid else None
    tracker = PersonTracker(
        max_missed=args.max_missed,
        reid_weight=args.reid_weight,
        reid_threshold=args.reid_threshold,
        relink_similarity=args.reid_relink,
        retired_ttl_frames=args.retired_ttl_frames,
    )

    judge = None
    if args.role_method == "vlm" or args.interaction_method == "vlm":
        from staff_interaction.vlm_judge import VLMJudge

        judge = VLMJudge(model_id=args.vlm_model, max_side=args.vlm_max_side, role_hint=args.staff_description)

    if args.interaction_method == "rule":
        interactions = RuleInteractionTracker(rule_interaction_params, fps=fps_eff)
    else:
        interactions = InteractionTracker(vlm_interaction_params, fps=fps_eff)

    writer = None
    if not args.no_video:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "staff_interaction_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_eff, (fw, fh))

    sessions: list = []
    staff_records: dict[int, dict] = {}
    frame_step = 0
    frame_raw = 0
    print(
        f"{args.video.name}: {total} frames @ {src_fps:.1f}fps stride={stride} -> {fps_eff:.1f}fps; "
        f"role={args.role_method} interaction={args.interaction_method}"
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
            evs = interactions.close_all_for(tr.track_id, frame_step)
            sessions.extend(evs)

        for tr in live:
            if args.role_method == "reid":
                update_role_reid(tr, gallery, role_reid_params)
            else:
                update_role(tr, frame, judge, role_vlm_params, frame_step, fps_eff)
            if tr.is_staff:
                rec = staff_records.setdefault(tr.track_id, {"first_frame": tr.first_frame, "last_frame": tr.last_frame})
                rec["last_frame"] = tr.last_frame
                interactions.register_staff(tr.track_id)

        staff = [t for t in live if t.is_staff]
        customers = [t for t in live if not t.is_staff]
        pair_cues = {}
        for s in staff:
            for c in customers:
                key = (s.track_id, c.track_id)
                if args.interaction_method == "rule":
                    cues = rule_pair_cues(s, c, rule_interaction_params)
                    pair_cues[key] = cues
                    ev = interactions.update(s, c, cues, frame_step)
                else:
                    cues = interactions.eligible(s, c)
                    pair_cues[key] = cues
                    ev = interactions.update(s, c, cues, frame_step, frame, judge) if cues.eligible else None
                if ev is not None:
                    sessions.append(ev)

        if writer is not None or args.preview:
            vis = frame.copy()
            for tr in live:
                x1, y1, x2, y2 = tr.box
                color = STAFF_COLOR if tr.is_staff else CUSTOMER_COLOR
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                role_tag = "STAFF" if tr.is_staff else "customer"
                cv2.putText(vis, f"ID {tr.track_id} {role_tag}", (x1 + 2, max(16, y1 - 4)), FONT, 0.5, color, 2)
                if not tr.is_staff and tr.role_checks:
                    cv2.putText(
                        vis,
                        f"role {tr.role_votes}/{tr.role_checks}",
                        (x1, min(vis.shape[0] - 4, y2 + 16)),
                        FONT,
                        0.4,
                        color,
                        1,
                    )

            for s in staff:
                for c in customers:
                    key = (s.track_id, c.track_id)
                    cues = pair_cues.get(key)
                    if cues is None:
                        continue
                    is_open = interactions.is_open(s.track_id, c.track_id)
                    mid = ((s.foot_px + c.foot_px) * 0.5).astype(int)

                    if args.interaction_method == "rule":
                        show_line = is_open or cues.dist_bh <= rule_interaction_params.near_bh
                        if show_line:
                            line_color = ENGAGED_COLOR if is_open else (QUERIED_COLOR if cues.engaged else CANDIDATE_COLOR)
                            cv2.line(vis, tuple(map(int, s.foot_px)), tuple(map(int, c.foot_px)), line_color, 2 if is_open else 1)
                        if is_open:
                            dur = interactions.current_duration_s(s.track_id, c.track_id, frame_step)
                            cv2.putText(vis, f"{dur:.1f}s", tuple(mid), FONT, 0.45, ENGAGED_COLOR, 2)
                        elif show_line:
                            cv2.putText(vis, f"{cues.score:.2f}", tuple(mid), FONT, 0.42, QUERIED_COLOR, 2)
                    else:
                        if is_open or cues.eligible:
                            line_color = ENGAGED_COLOR if is_open else (QUERIED_COLOR if cues.queried else CANDIDATE_COLOR)
                            cv2.line(vis, tuple(map(int, s.foot_px)), tuple(map(int, c.foot_px)), line_color, 2 if is_open else 1)
                        if cues.queried:
                            ans = "?" if cues.answer is None else ("YES" if cues.answer else "no")
                            cv2.putText(vis, f"vlm:{ans}", tuple(mid), FONT, 0.42, QUERIED_COLOR, 2)
                        elif is_open:
                            dur = interactions.current_duration_s(s.track_id, c.track_id, frame_step)
                            cv2.putText(vis, f"{dur:.1f}s", tuple(mid), FONT, 0.45, ENGAGED_COLOR, 2)

            cv2.putText(vis, "STAFF INTERACTIONS", (10, 20), FONT, 0.55, (255, 255, 255), 2)
            vis = draw_metric_table(vis, staff_records, interactions.summary())

            if writer is not None:
                writer.write(vis)
            if args.preview:
                cv2.imshow("staff interaction (q to stop)", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        pbar.set_postfix(staff=len(staff_records), sessions=len(sessions), refresh=False)

        if args.max_frames and frame_step >= args.max_frames:
            break

    pbar.close()
    sessions.extend(interactions.close_end(frame_step))

    cap.release()
    if writer is not None:
        writer.release()
    if args.preview:
        cv2.destroyAllWindows()

    # every staff instance that was ever detected is included, even at 0 sessions
    summary = interactions.summary()
    for sid in staff_records:
        summary.setdefault(sid, 0)

    paths, rows, total_sessions, avg = write_outputs(args.out_dir, staff_records, summary, sessions)
    print("\n--- Staff-customer interaction (Task 3) ---")
    for r in rows:
        print(f"{r['staff_instance']:<14} sessions={r['interaction_sessions']}")
    print(f"{'staff_count':<14} {len(rows)}")
    print(f"{'total_sessions':<14} {total_sessions}")
    print(f"{'average/staff':<14} {avg:.3f}")
    print("\nwrote:")
    for pth in paths:
        print(" ", pth)
    if writer is not None:
        print(" ", args.out_dir / "staff_interaction_annotated.mp4")


if __name__ == "__main__":
    main()
