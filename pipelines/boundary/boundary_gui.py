#!/usr/bin/env python3
"""
Store boundary GUI (outside / inside / entrance).

This GUI writes only `configs/store_boundary_zones.json`.
Shelf faces are handled separately by `pipelines/shelf_vector_interest/shelf_face_gui.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipelines"))

from boundary.boundary_store import Boundary, load_boundary, save_boundary  # noqa: E402
from configs.paths import BOUNDARY_JSON, ENTRANCE_VIDEO  # noqa: E402

DISPLAY_W = 960

STEPS = [
    ("outside", "1/3  OUTSIDE (walking area): click the public walkway in front of the shop"),
    ("inside", "2/3  INSIDE (shop): click the area inside the store"),
    ("entrance", "3/3  Entrance line (optional): click 2 points across the doorway"),
]


class BoundaryApp(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("Boundary marker — outside (walking area) / inside (shop)")
        self.geometry("1180x820")
        ctk.set_appearance_mode("dark")

        self.video = video
        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.boundary = load_boundary(video.name)
        self.step = 0
        self.frame = None
        self.scale = 1.0
        self.photo = None

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(bar, text="< Back", width=80, command=self._back).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Next >", width=80, command=self._next).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Undo point", width=100, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear step", width=100, command=self._clear).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save", width=90, command=self._save).pack(side="left", padx=12)
        ctk.CTkLabel(bar, text=f"saving to: {BOUNDARY_JSON}").pack(side="left", padx=12)

        nav = ctk.CTkFrame(self)
        nav.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(nav, text="frame").pack(side="left", padx=(6, 4))
        self.slider = ctk.CTkSlider(
            nav, from_=0, to=max(1, self.total - 1), command=self._seek, width=760
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
        self.hint.configure(text=STEPS[self.step][1])
        self._redraw()

    def _back(self) -> None:
        self.step = max(self.step - 1, 0)
        self.hint.configure(text=STEPS[self.step][1])
        self._redraw()

    def _clear(self) -> None:
        key = self._key()
        if key == "outside":
            self.boundary.outside = []
        elif key == "inside":
            self.boundary.inside = []
        elif key == "entrance":
            self.boundary.entrance_line = []
        self._redraw()

    def _undo(self) -> None:
        key = self._key()
        target = {
            "outside": self.boundary.outside,
            "inside": self.boundary.inside,
            "entrance": self.boundary.entrance_line,
        }.get(key)
        if target:
            target.pop()
        self._redraw()

    # -------------------------------------------------------------------- input

    def _click(self, event) -> None:
        if self.frame is None:
            return
        fh, fw = self.frame.shape[:2]
        x = int(min(max(0, event.x / max(self.scale, 1e-6)), fw - 1))
        y = int(min(max(0, event.y / max(self.scale, 1e-6)), fh - 1))
        key = self._key()

        if key == "outside":
            self.boundary.outside.append([x, y])
        elif key == "inside":
            self.boundary.inside.append([x, y])
        elif key == "entrance":
            if len(self.boundary.entrance_line) >= 2:
                self.boundary.entrance_line = []
            self.boundary.entrance_line.append([x, y])

        self._redraw()

    def _save(self) -> None:
        ok, why = self.boundary.ready()
        save_boundary(self.video.name, self.boundary)
        extra = "" if ok else f"  (incomplete: {why})"
        self.status.configure(text=f"Saved to {BOUNDARY_JSON}{extra}")

    # ------------------------------------------------------------------ drawing

    def _redraw(self) -> None:
        if self.frame is None:
            return
        vis = self.frame.copy()
        key = self._key()
        b = self.boundary

        if len(b.outside) >= 2:
            self._poly(vis, b.outside, (0, 200, 255), key == "outside", "OUTSIDE / WALKING AREA")
        if len(b.inside) >= 2:
            self._poly(vis, b.inside, (0, 220, 80), key == "inside", "INSIDE / SHOP")
        if len(b.entrance_line) == 2:
            a, c = (tuple(map(int, p)) for p in b.entrance_line)
            cv2.line(vis, a, c, (60, 120, 255), 3)
            cv2.putText(vis, "ENTRANCE", a, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 120, 255), 2)
        for p in b.entrance_line:
            cv2.circle(vis, tuple(map(int, p)), 5, (60, 120, 255), -1)
        self._status_text()
        self._show(vis)

    def _poly(self, vis, pts, color, active: bool, name: str) -> None:
        arr = np.array(pts, np.int32)
        cv2.polylines(vis, [arr], len(pts) >= 3, color, 2 if active else 1)
        for p in pts:
            cv2.circle(vis, tuple(map(int, p)), 4, color, -1)
        cv2.putText(vis, name, tuple(arr[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _status_text(self) -> None:
        b = self.boundary
        parts = [
            f"outside pts {len(b.outside)}",
            f"inside pts {len(b.inside)}",
            f"entrance line {len(b.entrance_line)}/2",
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
    ap.add_argument("--video", type=Path, default=ENTRANCE_VIDEO)
    args = ap.parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    BoundaryApp(args.video.resolve()).mainloop()


if __name__ == "__main__":
    main()
