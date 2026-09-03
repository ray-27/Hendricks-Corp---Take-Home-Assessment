#!/usr/bin/env python3
"""
LIF gate -> YOLO on the active region -> ReID.

Usage:
    python3 run_hybrid_reid.py
    python3 run_hybrid_reid.py --video raw_videos/entrance.mp4

Empty / still frames: LIF does not fire, YOLO and ReID are skipped.
Moving frames: take the fired region, run YOLO only there, then ReID.

Left panel  = video with ID boxes
Right panel = LIF view (moving pixels kept, still pixels black)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reid.detector import PersonDetector  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402


MIN_FIRE_PIXELS = 400  # below this, treat the frame as empty
ROI_PAD = 48           # grow the LIF region so YOLO sees the full body
HOLD_FRAMES = 20       # keep last IDs briefly when someone stops moving


class LIF:
    """Coarse-grid leaky integrate-and-fire. Still scene -> no spikes."""

    def __init__(self, grid_w: int = 80, leak: float = 0.85, thresh: float = 18.0, gate: float = 12.0):
        self.grid_w, self.leak, self.thresh, self.gate = grid_w, leak, thresh, gate
        self.prev = None
        self.v = None

    def fire_map(self, frame) -> np.ndarray:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        h, w = gray.shape
        gh = max(1, int(self.grid_w * h / w))
        small = cv2.resize(gray, (self.grid_w, gh), interpolation=cv2.INTER_AREA)
        if self.prev is None or self.v is None or self.v.shape != small.shape:
            self.prev, self.v = small, np.zeros_like(small, dtype=np.float32)
            return np.zeros((h, w), dtype=np.uint8)
        motion = cv2.absdiff(small, self.prev).astype(np.float32)
        self.prev = small
        motion[motion < self.gate] = 0
        self.v = self.leak * self.v + motion
        fired = self.v > self.thresh
        self.v[fired] = 0
        fmap = cv2.resize((fired.astype(np.uint8) * 255), (w, h), interpolation=cv2.INTER_NEAREST)
        return cv2.dilate(fmap, np.ones((11, 11), np.uint8), iterations=2)


class Hybrid:
    def __init__(self):
        self.lif = LIF()
        self.yolo = PersonDetector()
        self.reid = ReIDEmbedder()
        self.embs: list[np.ndarray] = []
        self.tracks: list[dict] = []
        self.skipped = True

    def _assign(self, feat: np.ndarray, taken: set[int], thresh: float = 0.55) -> tuple[int, float]:
        if not self.embs:
            self.embs.append(feat)
            taken.add(0)
            return 1, 1.0
        sims = np.stack(self.embs) @ feat
        for j in np.argsort(-sims):
            j = int(j)
            if j in taken:
                continue
            s = float(sims[j])
            if s >= thresh:
                self.embs[j] = 0.85 * self.embs[j] + 0.15 * feat
                self.embs[j] /= max(np.linalg.norm(self.embs[j]), 1e-12)
                taken.add(j)
                return j + 1, s
            break
        self.embs.append(feat)
        taken.add(len(self.embs) - 1)
        return len(self.embs), 1.0

    def step(self, frame):
        fmap = self.lif.fire_map(frame)
        lif_view = np.zeros_like(frame)
        lif_view[fmap > 0] = frame[fmap > 0]  # moving pixels on, still = black

        ys, xs = np.where(fmap > 0)
        empty = xs.size < MIN_FIRE_PIXELS
        self.skipped = empty
        if empty:
            for t in self.tracks:
                t["missed"] += 1
            self.tracks = [t for t in self.tracks if t["missed"] < HOLD_FRAMES]
            return self._draw(frame), lif_view

        x1, x2 = max(0, int(xs.min()) - ROI_PAD), min(frame.shape[1], int(xs.max()) + ROI_PAD)
        y1, y2 = max(0, int(ys.min()) - ROI_PAD), min(frame.shape[0], int(ys.max()) + ROI_PAD)
        roi = frame[y1:y2, x1:x2]
        dets = self.yolo.detect(roi)
        crops, boxes = [], []
        for d in dets:
            boxes.append((d.x1 + x1, d.y1 + y1, d.x2 + x1, d.y2 + y1))
            crops.append(d.crop(roi))
        feats = self.reid.embed(crops) if crops else []
        live, taken = [], set()
        for box, feat in zip(boxes, feats):
            tid, sim = self._assign(feat, taken)
            live.append({"id": tid, "box": box, "sim": sim, "missed": 0})
        self.tracks = live
        return self._draw(frame, roi_box=(x1, y1, x2, y2)), lif_view

    def _draw(self, frame, roi_box=None):
        vis = frame.copy()
        if roi_box and not self.skipped:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 80, 80), 1)
        for t in self.tracks:
            x1, y1, x2, y2 = t["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
            label = f"ID {t['id']}  {t['sim']:.2f}"
            cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(vis, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        gate = "LIF skip (still)" if self.skipped else "YOLO+ReID on LIF region"
        cv2.putText(vis, gate, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis


class App(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("LIF → YOLO region → ReID")
        self.geometry("1280x640")
        ctk.set_appearance_mode("dark")
        self.video = video
        self.pipe = Hybrid()
        self.cap = cv2.VideoCapture(str(video))
        self.playing = False
        self.photos: list[ImageTk.PhotoImage] = []

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        names = [p.name for p in sorted((ROOT / "raw_videos").glob("*.mp4"))]
        self.menu = ctk.CTkOptionMenu(bar, values=names or [video.name], command=self._switch)
        self.menu.set(video.name)
        self.menu.pack(side="left", padx=4)
        self.btn = ctk.CTkButton(bar, text="Play", width=80, command=self.toggle)
        self.btn.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Step", width=70, command=self.step).pack(side="left", padx=4)
        self.status = ctk.CTkLabel(bar, text="Ready")
        self.status.pack(side="left", padx=12)

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        left = ctk.CTkFrame(body)
        right = ctk.CTkFrame(body)
        left.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        right.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        ctk.CTkLabel(left, text="Video + IDs").pack()
        ctk.CTkLabel(right, text="LIF  (moving pixels · still is black)").pack()
        self.video_lbl = ctk.CTkLabel(left, text="")
        self.lif_lbl = ctk.CTkLabel(right, text="")
        self.video_lbl.pack(fill="both", expand=True)
        self.lif_lbl.pack(fill="both", expand=True)
        self.after(80, self.step)

    def _switch(self, name: str):
        self.playing = False
        self.btn.configure(text="Play")
        self.cap.release()
        self.cap = cv2.VideoCapture(str(ROOT / "raw_videos" / name))
        self.pipe = Hybrid()
        self.step()

    def toggle(self):
        self.playing = not self.playing
        self.btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._loop()

    def _loop(self):
        if not self.playing:
            return
        self.step()
        self.after(15, self._loop)

    def step(self):
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.btn.configure(text="Play")
            self.status.configure(text="End of video")
            return
        vis, lif_view = self.pipe.step(frame)
        self._show(self.video_lbl, vis, 620)
        self._show(self.lif_lbl, lif_view, 620)
        n = len(self.pipe.tracks)
        gate = "skipped" if self.pipe.skipped else "ran YOLO+ReID"
        self.status.configure(text=f"{n} IDs · {gate}")

    def _show(self, label, bgr, max_w):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = max_w / w
        img = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self.photos.append(photo)
        self.photos = self.photos[-4:]
        label.configure(image=photo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-window", action="store_true")
    args = p.parse_args()
    if args.no_window:
        pipe, cap = Hybrid(), cv2.VideoCapture(str(args.video))
        n = args.max_frames or 15
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            vis, lif_view = pipe.step(frame)
            print(f"frame {i+1}  skipped={pipe.skipped}  ids={[t['id'] for t in pipe.tracks]}  "
                  f"lif_pixels={(lif_view > 0).any(axis=2).sum()}")
        cap.release()
        return
    App(args.video).mainloop()


if __name__ == "__main__":
    main()
