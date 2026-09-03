#!/usr/bin/env python3
"""
SNN gate -> YOLO on padded motion crops -> ReID.

Shop outline is drawn before Play (same configs/shop_roi.json as the other GUI).

    still frame  -> click shop polygon
    video frame  -> SNN spikes inside shop
                 -> if shop is active (or keepalive)
                 -> YOLO on the whole shop crop
                 -> ReID on YOLO full-body boxes

This stops legs and torso from becoming two ReID identities.

SNN only decides when the shop is worth scanning. YOLO always runs on the
full shop polygon (not a thin spike crop), so a person bending over a shelf
still gets a body box and a ReID id.

Usage:
    python3 run_snn_yolo_reid.py
    python3 run_snn_yolo_reid.py --video raw_videos/interior.mp4
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import video_snn as snn  # noqa: E402
from reid.detector import PersonDetector  # noqa: E402
from reid.embedder import ReIDEmbedder  # noqa: E402
from run_snn_reid_gui import (  # noqa: E402
    DISPLAY_W,
    load_rois,
    overlay_shop,
    save_roi,
)

GRID_MAX_SIDE = 240  # coarse motion gate; YOLO does the person box
MATCH_THRESH = 0.55
KEEPALIVE = 5  # YOLO the shop every N frames even if nobody is moving
HOLD_FRAMES = 10
SPIKE_ON = 8  # grid cells that must fire before we treat the shop as active


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-6)


def nms_boxes(dets: list[tuple[tuple[int, int, int, int], float]], thresh: float = 0.45):
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    keep = []
    for box, score in dets:
        if any(_iou(box, k[0]) > thresh for k in keep):
            continue
        keep.append((box, score))
    return keep


def shop_bbox(poly, fw: int, fh: int, pad: int = 12):
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(fw, max(xs) + pad),
        min(fh, max(ys) + pad),
    )


def in_shop(box, poly) -> bool:
    """Keep a person if feet, hips, or box-center sit in the shop polygon."""
    if poly is None or len(poly) < 3:
        return True
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    h = max(y2 - y1, 1)
    pts = ((cx, float(y2)), (cx, y1 + 0.7 * h), (cx, (y1 + y2) * 0.5))
    poly_np = np.array(poly, np.float32)
    return any(cv2.pointPolygonTest(poly_np, pt, False) >= 0 for pt in pts)


class SNNYoloReID:
    def __init__(self, frame_w: int, frame_h: int):
        long_side = max(frame_w, frame_h)
        scale = min(1.0, GRID_MAX_SIDE / long_side)
        self.W = max(8, int(round(frame_w * scale)))
        self.H = max(8, int(round(frame_h * scale)))
        self.sx = frame_w / self.W
        self.sy = frame_h / self.H
        self.retina = snn.SpikingRetina((self.H, self.W), tau=3.0, v_th=1.0)
        self.prev = None
        self.yolo = PersonDetector(conf=0.25, min_aspect=0.45, min_h=22, min_w=14)
        self.reid = ReIDEmbedder()
        self.notebook: list[dict] = []
        self.live: list[dict] = []
        self.rois: list[tuple[int, int, int, int]] = []
        self.fired_on = np.zeros((self.H, self.W), dtype=bool)
        self.fired_off = np.zeros((self.H, self.W), dtype=bool)
        self.frame_i = 0
        self.shop_poly = None

    def reset(self):
        self.retina.reset()
        self.prev = None
        self.notebook.clear()
        self.live.clear()
        self.rois.clear()
        self.frame_i = 0

    def _assign(self, feat: np.ndarray, crop: np.ndarray, taken: set[int]) -> tuple[int, float]:
        if not self.notebook:
            self.notebook.append(self._note(1, feat, crop, 1.0))
            taken.add(0)
            return 1, 1.0
        sims = np.stack([n["emb"] for n in self.notebook]) @ feat
        for j in np.argsort(-sims):
            j = int(j)
            if j in taken:
                continue
            s = float(sims[j])
            if s >= MATCH_THRESH:
                n = self.notebook[j]
                n["emb"] = 0.85 * n["emb"] + 0.15 * feat
                n["emb"] /= max(np.linalg.norm(n["emb"]), 1e-12)
                n["hits"] += 1
                n["sim"] = s
                n["last"] = self.frame_i
                n["thumb"] = _thumb(crop)
                taken.add(j)
                return n["id"], s
            break
        tid = self.notebook[-1]["id"] + 1
        self.notebook.append(self._note(tid, feat, crop, 1.0))
        taken.add(len(self.notebook) - 1)
        return tid, 1.0

    def _note(self, tid, feat, crop, sim):
        return {
            "id": tid,
            "emb": feat,
            "thumb": _thumb(crop),
            "hits": 1,
            "sim": sim,
            "first": self.frame_i,
            "last": self.frame_i,
        }

    def step(self, frame):
        self.frame_i += 1
        fh, fw = frame.shape[:2]
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.W, self.H), interpolation=cv2.INTER_AREA)
        if self.prev is None:
            self.prev = gray
            self.live, self.rois = [], []
            return self._draw(frame), self._spike_view()

        s_on, s_off, _ = snn.events_from_frames(self.prev, gray, 12.0, 12.0)
        self.fired_on, self.fired_off = self.retina.step(s_on, s_off)
        self.prev = gray
        if self.shop_poly is not None and len(self.shop_poly) >= 3:
            grid_mask = np.zeros((self.H, self.W), np.uint8)
            gpoly = np.array([[int(x / self.sx), int(y / self.sy)] for x, y in self.shop_poly], np.int32)
            cv2.fillPoly(grid_mask, [gpoly], 1)
            inside = grid_mask.astype(bool)
            self.fired_on &= inside
            self.fired_off &= inside

        n_spikes = int(self.fired_on.sum() + self.fired_off.sum())
        scan_shop = n_spikes >= SPIKE_ON or (self.frame_i % KEEPALIVE == 0)

        if self.shop_poly is not None and len(self.shop_poly) >= 3:
            shop_roi = shop_bbox(self.shop_poly, fw, fh)
        else:
            shop_roi = (0, 0, fw, fh)

        # SNN only decides *when* to look. YOLO always sees the whole shop,
        # not a thin limb crop — otherwise standing/bending people are skipped.
        self.rois = [shop_roi] if scan_shop else []
        yolo_dets = []
        if scan_shop:
            rx1, ry1, rx2, ry2 = shop_roi
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size > 0:
                for det in self.yolo.detect(crop):
                    box = (det.x1 + rx1, det.y1 + ry1, det.x2 + rx1, det.y2 + ry2)
                    if in_shop(box, self.shop_poly):
                        yolo_dets.append((box, det.score))
        yolo_dets = nms_boxes(yolo_dets)

        crops, boxes = [], []
        for box, _score in yolo_dets:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            person = frame[y1:y2, x1:x2]
            if person.size == 0:
                continue
            crops.append(person)
            boxes.append((x1, y1, x2, y2))

        feats = self.reid.embed(crops) if crops else []
        live, taken = [], set()
        for box, feat, crop in zip(boxes, feats, crops):
            tid, sim = self._assign(feat, crop, taken)
            live.append({"id": tid, "box": box, "sim": sim, "missed": 0})
        matched = {t["id"] for t in live}
        for old in self.live:
            if old["id"] in matched:
                continue
            missed = old.get("missed", 0) + 1
            if missed <= HOLD_FRAMES:
                old = dict(old)
                old["missed"] = missed
                live.append(old)
        self.live = live
        return self._draw(frame), self._spike_view()

    def _spike_view(self):
        img = snn.colorize_events(self.fired_on, self.fired_off, scale=1)
        img = cv2.resize(img, (int(self.W * self.sx), int(self.H * self.sy)), interpolation=cv2.INTER_NEAREST)
        if self.shop_poly is not None and len(self.shop_poly) >= 3:
            cv2.polylines(img, [np.array(self.shop_poly, np.int32)], True, (0, 220, 80), 1)
        for roi in self.rois:
            x1, y1, x2, y2 = roi
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)
        for t in self.live:
            x1, y1, x2, y2 = t["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(img, f"ID {t['id']}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    def _draw(self, frame):
        vis = overlay_shop(frame, self.shop_poly, locked=True)
        for roi in self.rois:
            x1, y1, x2, y2 = roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (90, 90, 90), 1)
        for t in self.live:
            x1, y1, x2, y2 = t["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
            text = f"ID {t['id']}  {t['sim']:.2f}"
            cv2.putText(vis, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(vis, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        cv2.putText(
            vis,
            f"SNN gate -> YOLO shop -> ReID   people {len(self.live)}   noted {len(self.notebook)}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return vis


def _thumb(crop, size=(64, 128)):
    if crop is None or crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


class App(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("SNN → YOLO → ReID  —  draw shop, then identify")
        self.geometry("1380x760")
        ctk.set_appearance_mode("dark")
        self.video = video
        self.playing = False
        self.locked = False
        self.draw_pts: list[list[int]] = []
        self.photos: list[ImageTk.PhotoImage] = []
        self.still = None
        self.disp_scale = 1.0
        self._open_video(video)

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        names = [p.name for p in sorted((ROOT / "raw_videos").glob("*.mp4"))]
        self.menu = ctk.CTkOptionMenu(bar, values=names or [video.name], command=self._switch)
        self.menu.set(video.name)
        self.menu.pack(side="left", padx=4)
        self.btn = ctk.CTkButton(bar, text="Play", width=80, command=self.toggle, state="disabled")
        self.btn.pack(side="left", padx=4)
        self.step_btn = ctk.CTkButton(bar, text="Step", width=70, command=self.step, state="disabled")
        self.step_btn.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Close boundary", width=120, command=self._lock).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Undo point", width=90, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Redraw shop", width=100, command=self._redraw).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear notebook", width=120, command=self._clear).pack(side="left", padx=4)
        self.status = ctk.CTkLabel(bar, text="Draw the shop, Close boundary, then Play.")
        self.status.pack(side="left", padx=12)

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        left, mid, right = ctk.CTkFrame(body), ctk.CTkFrame(body), ctk.CTkFrame(body, width=280)
        left.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        right.pack(side="left", fill="y", padx=4, pady=4)
        right.pack_propagate(False)
        ctk.CTkLabel(left, text="Shop + YOLO body boxes").pack()
        ctk.CTkLabel(mid, text="SNN spikes  |  gray = YOLO crop  |  white = person").pack()
        ctk.CTkLabel(right, text="Noted people", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(6, 2))
        self.video_canvas = ctk.CTkCanvas(left, highlightthickness=0, cursor="crosshair")
        self.video_canvas.pack(fill="both", expand=True)
        self.video_canvas.bind("<Button-1>", self._on_click)
        self.video_canvas.bind("<Double-Button-1>", lambda e: self._lock())
        self.snn_lbl = ctk.CTkLabel(mid, text="")
        self.snn_lbl.pack(fill="both", expand=True)
        self.notes = ctk.CTkScrollableFrame(right, width=260)
        self.notes.pack(fill="both", expand=True, padx=6, pady=6)
        self.after(80, self._show_setup)

    def _open_video(self, video: Path):
        self.video = video
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(video))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        ok, still = self.cap.read()
        self.still = still if ok else np.zeros((h, w, 3), np.uint8)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.pipe = SNNYoloReID(w, h)
        saved = load_rois().get(video.name)
        self.draw_pts = [list(p) for p in saved] if saved else []
        self.locked = False
        self.playing = False

    def _switch(self, name: str):
        self.playing = False
        self._open_video(ROOT / "raw_videos" / name)
        self.btn.configure(text="Play", state="disabled")
        self.step_btn.configure(state="disabled")
        self._show_setup()

    def _show_setup(self):
        vis = overlay_shop(self.still, None, False, preview_pts=self.draw_pts)
        self._show_canvas(vis)
        self._show(self.snn_lbl, np.zeros_like(self.still), DISPLAY_W)
        self.status.configure(
            text=f"Setup: {len(self.draw_pts)} points. Click shop floor, then Close boundary."
        )

    def _on_click(self, event):
        if self.locked:
            return
        fh, fw = self.still.shape[:2]
        x = int(event.x / max(self.disp_scale, 1e-6))
        y = int(event.y / max(self.disp_scale, 1e-6))
        self.draw_pts.append([min(max(0, x), fw - 1), min(max(0, y), fh - 1)])
        self._show_setup()

    def _undo(self):
        if not self.locked and self.draw_pts:
            self.draw_pts.pop()
            self._show_setup()

    def _redraw(self):
        self.playing = False
        self.locked = False
        self.draw_pts = []
        self.pipe.shop_poly = None
        self.pipe.reset()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.btn.configure(text="Play", state="disabled")
        self.step_btn.configure(state="disabled")
        self._show_setup()

    def _lock(self):
        if len(self.draw_pts) < 3:
            self.status.configure(text="Need at least 3 clicks to close the shop.")
            return
        self.locked = True
        self.pipe.shop_poly = [list(p) for p in self.draw_pts]
        save_roi(self.video.name, self.pipe.shop_poly)
        self.btn.configure(state="normal")
        self.step_btn.configure(state="normal")
        self._show_canvas(overlay_shop(self.still, self.pipe.shop_poly, locked=True))
        self.status.configure(text="Shop locked. SNN+YOLO+ReID ready. Press Play.")

    def _clear(self):
        self.pipe.notebook.clear()
        self.pipe.live.clear()
        self._render_notes()
        self.status.configure(text="Notebook cleared")

    def toggle(self):
        if not self.locked:
            self.status.configure(text="Close the shop boundary first.")
            return
        self.playing = not self.playing
        self.btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._loop()

    def _loop(self):
        if self.playing:
            self.step()
            self.after(10, self._loop)

    def step(self):
        if not self.locked:
            self.status.configure(text="Close the shop boundary first.")
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            self.btn.configure(text="Play")
            self.status.configure(text="End of video")
            return
        vis, spikes = self.pipe.step(frame)
        self._show_canvas(vis)
        self._show(self.snn_lbl, spikes, DISPLAY_W)
        self._render_notes()
        self.status.configure(
            text=f"ReID {self.pipe.reid.device_name}  YOLO crops {len(self.pipe.rois)}  "
            f"people {len(self.pipe.live)}  noted {len(self.pipe.notebook)}"
        )

    def _show_canvas(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.disp_scale = DISPLAY_W / max(w, 1)
        img = Image.fromarray(rgb).resize((int(w * self.disp_scale), int(h * self.disp_scale)), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self.photos.append(photo)
        self.photos = self.photos[-8:]
        self.video_canvas.configure(width=photo.width(), height=photo.height())
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, image=photo, anchor="nw")

    def _show(self, label, bgr, max_w):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = max_w / max(w, 1)
        img = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        self.photos.append(photo)
        self.photos = self.photos[-8:]
        label.configure(image=photo)

    def _render_notes(self):
        for child in self.notes.winfo_children():
            child.destroy()
        seen = {t["id"] for t in self.pipe.live}
        if not self.pipe.notebook:
            ctk.CTkLabel(self.notes, text="No one noted yet.\nLock shop, then Play.").pack(pady=12)
            return
        for n in self.pipe.notebook:
            row = ctk.CTkFrame(self.notes, fg_color=("#1f6aa5" if n["id"] in seen else "#2b2b2b"))
            row.pack(fill="x", pady=4, padx=2)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(n["thumb"], cv2.COLOR_BGR2RGB)))
            self.photos.append(photo)
            ctk.CTkLabel(row, image=photo, text="").pack(side="left", padx=4, pady=4)
            where = "IN SHOP" if n["id"] in seen else "left shop"
            ctk.CTkLabel(
                row,
                text=f"ID {n['id']}   {where}\nhits {n['hits']}   sim {n['sim']:.2f}\nfirst f{n['first']}  last f{n['last']}",
                justify="left",
            ).pack(side="left", padx=6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "interior.mp4")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-window", action="store_true")
    args = p.parse_args()
    if args.no_window:
        cap = cv2.VideoCapture(str(args.video))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        pipe = SNNYoloReID(w, h)
        n = args.max_frames or 10
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            pipe.step(frame)
            print(
                f"frame {i+1}  rois={len(pipe.rois)}  "
                f"people={[t['id'] for t in pipe.live]}  noted={len(pipe.notebook)}"
            )
        cap.release()
        return
    App(args.video).mainloop()


if __name__ == "__main__":
    main()
