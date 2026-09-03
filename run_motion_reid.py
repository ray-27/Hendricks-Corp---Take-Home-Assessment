#!/usr/bin/env python3
"""
Motion-gated ReID — leaky integrate-and-fire, no YOLO.

Usage:
    python3 run_motion_reid.py
    python3 run_motion_reid.py --video raw_videos/entrance.mp4

Idea
----
The original pipeline runs two neural nets on every frame:
    full frame -> YOLO (find people) -> crops -> ReID (who is this)

That is the standard design. ReID cannot search a frame by itself; it only
turns a body crop into an embedding. YOLO-nano is cheap. ReID is the heavy
step, and it already runs only on the N person crops, not the whole image.

This file tries a cheaper *front door*:
    frame difference -> leaky integrate-and-fire -> moving blobs -> ReID

A leaky integrate-and-fire (LIF) cell is a tiny analog of a neuron:
    voltage = leak * voltage + incoming_motion
    if voltage > threshold: FIRE, then reset

We put one LIF cell on each coarse grid cell of the frame. Motion charges
the cell; silence leaks it back down. Only fired regions are cropped and
sent to ReIdentificationNet.

This is efficient on static CCTV (empty aisle = no ReID). It is weaker than
YOLO when someone stands still looking at a shelf — little motion, so the
person can vanish. Treat this as an experiment, not a replacement yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reid.embedder import ReIDEmbedder  # noqa: E402  (existing ONNX wrapper, unchanged)


# --- LIF motion detector ----------------------------------------------------

class LeakyFireMotion:
    """Find moving regions with leaky integrate-and-fire cells.

    The frame is downsampled to a coarse grid so this stays cheap (numpy only).
    """

    def __init__(
        self,
        grid_w: int = 80,
        leak: float = 0.85,       # 0 = forget instantly, 1 = never forget
        fire_threshold: float = 18.0,
        motion_gate: float = 12.0,  # ignore tiny pixel flicker before integrating
    ) -> None:
        self.grid_w = grid_w
        self.leak = leak
        self.fire_threshold = fire_threshold
        self.motion_gate = motion_gate
        self.prev_gray = None
        self.voltage = None  # membrane potential of each grid cell

    def blobs(self, frame_bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """Return (bounding boxes, fire heatmap for drawing)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        h, w = gray.shape
        grid_h = max(1, int(self.grid_w * h / w))
        small = cv2.resize(gray, (self.grid_w, grid_h), interpolation=cv2.INTER_AREA)

        if self.prev_gray is None or self.voltage is None or self.voltage.shape != small.shape:
            self.prev_gray = small
            self.voltage = np.zeros_like(small, dtype=np.float32)
            return [], np.zeros((h, w), dtype=np.uint8)

        motion = cv2.absdiff(small, self.prev_gray).astype(np.float32)
        self.prev_gray = small
        motion[motion < self.motion_gate] = 0.0  # drop camera noise

        # leaky integrate
        self.voltage = self.leak * self.voltage + motion

        # fire where potential crossed threshold, then reset those cells
        fired = self.voltage > self.fire_threshold
        self.voltage[fired] = 0.0

        fire_small = (fired.astype(np.uint8) * 255)
        fire_map = cv2.resize(fire_small, (w, h), interpolation=cv2.INTER_NEAREST)
        fire_map = cv2.dilate(fire_map, np.ones((9, 9), np.uint8), iterations=2)

        boxes = []
        contours, _ = cv2.findContours(fire_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 800:  # skip tiny sparkle
                continue
            # Expand a bit so the crop still looks like a person, not a limb.
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            boxes.append((x1, y1, x2, y2))
        return boxes, fire_map


# --- tiny appearance gallery ------------------------------------------------

class SimpleGallery:
    """Assign IDs by cosine similarity of ReID embeddings."""

    def __init__(self, match_threshold: float = 0.55) -> None:
        self.match_threshold = match_threshold
        self.embeddings: list[np.ndarray] = []
        self.next_id = 1

    def assign(self, feat: np.ndarray) -> tuple[int, float]:
        if not self.embeddings:
            self.embeddings.append(feat)
            tid = self.next_id
            self.next_id += 1
            return tid, 1.0

        gallery = np.stack(self.embeddings, axis=0)
        sims = gallery @ feat
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= self.match_threshold:
            # slow average so the ID stays stable
            self.embeddings[best] = 0.85 * self.embeddings[best] + 0.15 * feat
            n = np.linalg.norm(self.embeddings[best])
            self.embeddings[best] /= max(n, 1e-12)
            return best + 1, score

        self.embeddings.append(feat)
        tid = self.next_id
        self.next_id += 1
        return tid, 1.0


def draw(frame, boxes, labels, fire_map) -> np.ndarray:
    vis = frame.copy()
    tint = vis.copy()
    tint[fire_map > 0] = (0, 80, 0)  # green wash over cells that fired
    vis = cv2.addWeighted(vis, 0.78, tint, 0.22, 0)
    for (x1, y1, x2, y2), (tid, sim) in zip(boxes, labels):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
        text = f"ID {tid}  {sim:.2f}"
        cv2.putText(vis, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
    hud = "LIF motion -> ReID  |  space=pause  q=quit  (no YOLO)"
    cv2.putText(vis, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"moving blobs: {len(boxes)}   identities: {len(labels) and max(t for t,_ in labels) or 0}",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leaky-integrate-and-fire motion ReID")
    p.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    p.add_argument("--max-frames", type=int, default=0, help="0 = play whole video")
    p.add_argument("--no-window", action="store_true", help="headless (for a quick test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    print("Loading ReIdentificationNet (ONNX)...")
    reid = ReIDEmbedder()
    motion = LeakyFireMotion()
    gallery = SimpleGallery()

    cap = cv2.VideoCapture(str(args.video))
    paused = False
    idx = 0
    print("Playing. q = quit, space = pause.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            idx += 1
            boxes, fire_map = motion.blobs(frame)
            crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
            labels: list[tuple[int, float]] = []
            if crops:
                feats = reid.embed(crops)
                for feat in feats:
                    labels.append(gallery.assign(feat))
            vis = draw(frame, boxes, labels, fire_map)
            print(f"frame {idx:4d}  blobs={len(boxes)}  ids={[t for t, _ in labels]}")
        else:
            vis = frame

        if not args.no_window:
            cv2.imshow("Motion LIF + ReID", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused

        if args.max_frames and idx >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
