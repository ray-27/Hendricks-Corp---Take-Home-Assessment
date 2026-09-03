#!/usr/bin/env python3
"""Headless smoke test: process a few frames and write an annotated snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reid.pipeline import ReIDPipeline  # noqa: E402


def main() -> None:
    video = ROOT / "raw_videos" / "entrance.mp4"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    pipe = ReIDPipeline()
    last = None
    for i in range(12):
        ok, frame = cap.read()
        if not ok:
            break
        if i % 2:
            continue
        last = pipe.process_frame(frame)
        print(
            f"frame {last.frame_index}: in-view={len(last.tracks)} "
            f"ids={[t.track_id for t in last.tracks]} unique={last.unique_ids}"
        )
    cap.release()
    if last is None:
        raise SystemExit("No frames decoded")
    snap = out_dir / "reid_preview.jpg"
    cv2.imwrite(str(snap), last.annotated)
    print(f"wrote {snap}")


if __name__ == "__main__":
    main()
