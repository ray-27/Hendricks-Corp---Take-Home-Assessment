#!/usr/bin/env python3
"""
One-time scene setup for the entrance analytics (Task 1 + Task 3).

The camera is fixed, so everything here is clicked once and saved to
configs/entrance_zones.json:

  1. exterior zone   walkway in front of the shop -> where interest is judged
  2. interior zone   inside the store             -> the "entered" decision
  3. entrance line   the threshold                -> the storefront people look at
  4. floor rectangle 4 corners of a known-size floor patch -> metres, not pixels
  5. apron colour    click a staff apron          -> HSV range for staff votes
  6. staff enrolment click each staff member      -> ReID gallery

Usage:
    python3 run_entrance_setup.py
    python3 run_entrance_setup.py --video raw_videos/entrance.mp4
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

from analytics.zones import Scene, load_scene, save_scene  # noqa: E402

DISPLAY_W = 900

STEPS = [
    ("exterior", "1/6  Exterior zone: click the walkway in front of the shop"),
    ("interior", "2/6  Interior zone: click the area inside the store"),
    ("entrance", "3/6  Entrance line: click 2 points across the doorway"),
    ("floor", "4/6  Floor rectangle: click near-left, near-right, far-right, far-left"),
    ("apron", "5/6  Apron colour: click on a staff apron (a few clicks help)"),
    ("staff", "6/6  Staff enrolment: click on each staff member's body"),
]


def hsv_range_from_samples(samples: list[tuple[int, int, int]]) -> dict:
    """Build a tolerant HSV window around the sampled apron pixels."""
    arr = np.array(samples, np.int32)
    h, s, v = arr[:, 0], arr[:, 1], arr[:, 2]
    return {
        "h": [int(max(0, h.min() - 12)), int(min(179, h.max() + 12))],
        "s": [int(max(30, s.min() - 60)), 255],
        "v": [int(max(30, v.min() - 60)), 255],
    }


class SetupApp(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("Entrance setup — zones, scale, staff")
        self.geometry("1180x860")
        ctk.set_appearance_mode("dark")

        self.video = video
        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.scene = load_scene(video.name)
        self.apron_samples: list[tuple[int, int, int]] = []
        self.step = 0
        self.frame = None
        self.scale = 1.0
        self.photo = None
        self.pose = None
        self.reid = None

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(bar, text="< Back", width=80, command=self._back).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Next >", width=80, command=self._next).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Undo point", width=100, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear step", width=100, command=self._clear).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save", width=80, command=self._save).pack(side="left", padx=12)

        ctk.CTkLabel(bar, text="floor w (m)").pack(side="left", padx=(16, 2))
        self.w_entry = ctk.CTkEntry(bar, width=60)
        self.w_entry.insert(0, str(self.scene.floor_w_m))
        self.w_entry.pack(side="left")
        ctk.CTkLabel(bar, text="h (m)").pack(side="left", padx=(8, 2))
        self.h_entry = ctk.CTkEntry(bar, width=60)
        self.h_entry.insert(0, str(self.scene.floor_h_m))
        self.h_entry.pack(side="left")

        nav = ctk.CTkFrame(self)
        nav.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(nav, text="frame").pack(side="left", padx=(6, 4))
        self.slider = ctk.CTkSlider(
            nav, from_=0, to=max(1, self.total - 1), command=self._seek, width=700
        )
        self.slider.set(0)
        self.slider.pack(side="left", padx=4)
        self.frame_lbl = ctk.CTkLabel(nav, text="0")
        self.frame_lbl.pack(side="left", padx=8)

        self.hint = ctk.CTkLabel(self, text=STEPS[0][1], font=ctk.CTkFont(size=15, weight="bold"))
        self.hint.pack(pady=(0, 4))
        self.status = ctk.CTkLabel(self, text="")
        self.status.pack(pady=(0, 4))

        self.canvas = ctk.CTkCanvas(self, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._click)

        self._seek(0)

    # ----------------------------------------------------------------- plumbing

    def _seek(self, value) -> None:
        idx = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
        self.frame_lbl.configure(text=str(idx))
        self._redraw()

    def _key(self) -> str:
        return STEPS[self.step][0]

    def _next(self) -> None:
        self.step = min(self.step + 1, len(STEPS) - 1)
        if self._key() == "staff":
            self._load_models()
        self.hint.configure(text=STEPS[self.step][1])
        self._redraw()

    def _back(self) -> None:
        self.step = max(self.step - 1, 0)
        self.hint.configure(text=STEPS[self.step][1])
        self._redraw()

    def _clear(self) -> None:
        key = self._key()
        if key == "exterior":
            self.scene.exterior = []
        elif key == "interior":
            self.scene.interior = []
        elif key == "entrance":
            self.scene.entrance_line = []
        elif key == "floor":
            self.scene.floor_pts = []
        elif key == "apron":
            self.apron_samples = []
            self.scene.apron_hsv = None
        elif key == "staff":
            self.scene.staff_gallery = []
        self._redraw()

    def _undo(self) -> None:
        key = self._key()
        target = {
            "exterior": self.scene.exterior,
            "interior": self.scene.interior,
            "entrance": self.scene.entrance_line,
            "floor": self.scene.floor_pts,
            "staff": self.scene.staff_gallery,
        }.get(key)
        if key == "apron":
            if self.apron_samples:
                self.apron_samples.pop()
                self.scene.apron_hsv = (
                    hsv_range_from_samples(self.apron_samples) if self.apron_samples else None
                )
        elif target is not None and target:
            target.pop()
        self._redraw()

    def _load_models(self) -> None:
        if self.pose is not None:
            return
        self.status.configure(text="Loading pose + ReID models…")
        self.update()
        from analytics.pose import PoseDetector
        from reid.embedder import ReIDEmbedder

        self.pose = PoseDetector()
        self.reid = ReIDEmbedder()
        self.status.configure(text="Models ready — click a staff member.")

    # -------------------------------------------------------------------- input

    def _click(self, event) -> None:
        if self.frame is None:
            return
        fh, fw = self.frame.shape[:2]
        x = int(min(max(0, event.x / max(self.scale, 1e-6)), fw - 1))
        y = int(min(max(0, event.y / max(self.scale, 1e-6)), fh - 1))
        key = self._key()

        if key == "exterior":
            self.scene.exterior.append([x, y])
        elif key == "interior":
            self.scene.interior.append([x, y])
        elif key == "entrance":
            if len(self.scene.entrance_line) >= 2:
                self.scene.entrance_line = []
            self.scene.entrance_line.append([x, y])
        elif key == "floor":
            if len(self.scene.floor_pts) >= 4:
                self.scene.floor_pts = []
            self.scene.floor_pts.append([x, y])
        elif key == "apron":
            patch = self.frame[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            med = np.median(hsv, axis=0).astype(int)
            self.apron_samples.append((int(med[0]), int(med[1]), int(med[2])))
            self.scene.apron_hsv = hsv_range_from_samples(self.apron_samples)
        elif key == "staff":
            self._enrol_staff(x, y)

        self._redraw()

    def _enrol_staff(self, x: int, y: int) -> None:
        self._load_models()
        dets = self.pose.detect(self.frame)
        hit = None
        for det in dets:
            bx1, by1, bx2, by2 = det.box
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                if hit is None or det.height < hit.height:
                    hit = det
        if hit is None:
            self.status.configure(text="No person detected at that click — try their torso.")
            return
        emb = self.reid.embed([hit.crop(self.frame)])
        if emb.shape[0] == 0:
            self.status.configure(text="Could not embed that crop.")
            return
        self.scene.staff_gallery.append(emb[0].astype(np.float32))
        self.status.configure(
            text=f"Enrolled staff #{len(self.scene.staff_gallery)} (box height {hit.height}px)"
        )

    def _save(self) -> None:
        try:
            self.scene.floor_w_m = float(self.w_entry.get())
            self.scene.floor_h_m = float(self.h_entry.get())
        except ValueError:
            self.status.configure(text="Floor width/height must be numbers.")
            return
        ok, why = self.scene.ready()
        save_scene(self.video.name, self.scene)
        extra = "" if ok else f"  (incomplete: {why})"
        self.status.configure(text=f"Saved to configs/entrance_zones.json{extra}")

    # ------------------------------------------------------------------ drawing

    def _redraw(self) -> None:
        if self.frame is None:
            return
        vis = self.frame.copy()
        key = self._key()

        if len(self.scene.exterior) >= 2:
            self._poly(vis, self.scene.exterior, (0, 200, 255), key == "exterior", "EXTERIOR")
        if len(self.scene.interior) >= 2:
            self._poly(vis, self.scene.interior, (0, 220, 80), key == "interior", "INTERIOR")
        if len(self.scene.entrance_line) == 2:
            a, b = (tuple(map(int, p)) for p in self.scene.entrance_line)
            cv2.line(vis, a, b, (60, 120, 255), 3)
            cv2.putText(vis, "ENTRANCE", a, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 120, 255), 2)
        for i, p in enumerate(self.scene.floor_pts):
            cv2.circle(vis, tuple(map(int, p)), 6, (255, 255, 255), -1)
            cv2.putText(
                vis, str(i + 1), (int(p[0]) + 8, int(p[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
            )
        if len(self.scene.floor_pts) == 4:
            cv2.polylines(vis, [np.array(self.scene.floor_pts, np.int32)], True, (255, 255, 255), 1)
        for p in self.scene.entrance_line:
            cv2.circle(vis, tuple(map(int, p)), 5, (60, 120, 255), -1)

        if key == "apron" and self.scene.apron_hsv:
            mask = cv2.inRange(
                cv2.cvtColor(vis, cv2.COLOR_BGR2HSV),
                np.array(
                    [
                        self.scene.apron_hsv["h"][0],
                        self.scene.apron_hsv["s"][0],
                        self.scene.apron_hsv["v"][0],
                    ],
                    np.uint8,
                ),
                np.array(
                    [
                        self.scene.apron_hsv["h"][1],
                        self.scene.apron_hsv["s"][1],
                        self.scene.apron_hsv["v"][1],
                    ],
                    np.uint8,
                ),
            )
            vis[mask > 0] = (0, 0, 255)

        self._status_text()
        self._show(vis)

    def _poly(self, vis, pts, color, active: bool, name: str) -> None:
        arr = np.array(pts, np.int32)
        cv2.polylines(vis, [arr], len(pts) >= 3, color, 2 if active else 1)
        for p in pts:
            cv2.circle(vis, tuple(map(int, p)), 4, color, -1)
        cv2.putText(vis, name, tuple(arr[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _status_text(self) -> None:
        s = self.scene
        parts = [
            f"exterior {len(s.exterior)}",
            f"interior {len(s.interior)}",
            f"line {len(s.entrance_line)}/2",
            f"floor {len(s.floor_pts)}/4",
            f"apron {'set' if s.apron_hsv else '-'}",
            f"staff {len(s.staff_gallery)}",
        ]
        self.status.configure(text="   ".join(parts))

    def _show(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.scale = DISPLAY_W / w
        img = Image.fromarray(rgb).resize((DISPLAY_W, int(h * self.scale)), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(width=self.photo.width(), height=self.photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "entrance.mp4")
    args = ap.parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    SetupApp(args.video.resolve()).mainloop()


if __name__ == "__main__":
    main()
