#!/usr/bin/env python3
"""
Shelf boundary GUI (interior shelves only).

Reads store interior from `store_boundary_zones.json` for context and writes
only shelf polygons/front-lines to `shelf_zones.json`.
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
sys.path.insert(0, str(ROOT / "pipelines"))

from boundary.boundary_store import load_boundary  # noqa: E402
from boundary.shelf_store import Shelf, ShelfLayout, load_shelf_layout, save_shelf_layout, shelf_color  # noqa: E402

DISPLAY_W = 960


class ShelfApp(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("Shelf marker")
        self.geometry("1180x820")
        ctk.set_appearance_mode("dark")

        self.video = video
        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.store_boundary = load_boundary(video.name)
        self.layout = load_shelf_layout(video.name)
        if not self.layout.shelves:
            sid = self.layout.next_shelf_id()
            self.layout.shelves = [Shelf(shelf_id=sid, name=f"Shelf {sid}")]
        self.shelf_idx = 0
        self.editing_front_line = False
        self.frame = None
        self.scale = 1.0
        self.photo = None

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(bar, text="Undo point", width=100, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear shelf", width=100, command=self._clear).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save", width=90, command=self._save).pack(side="left", padx=12)
        ctk.CTkLabel(bar, text="saving to: pipelines/configs/shelf_zones.json").pack(side="left", padx=12)

        shelf_bar = ctk.CTkFrame(self)
        shelf_bar.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(shelf_bar, text="shelf").pack(side="left", padx=(6, 4))
        ctk.CTkButton(shelf_bar, text="< Prev shelf", width=110, command=self._prev_shelf).pack(side="left", padx=4)
        ctk.CTkButton(shelf_bar, text="Next shelf >", width=110, command=self._next_shelf).pack(side="left", padx=4)
        ctk.CTkButton(shelf_bar, text="+ Add shelf", width=100, command=self._add_shelf).pack(side="left", padx=4)
        ctk.CTkButton(shelf_bar, text="- Delete shelf", width=110, command=self._delete_shelf).pack(side="left", padx=4)
        ctk.CTkButton(shelf_bar, text="Toggle front-line edit", width=150, command=self._toggle_mode).pack(
            side="left", padx=(16, 4)
        )
        self.shelf_lbl = ctk.CTkLabel(shelf_bar, text="")
        self.shelf_lbl.pack(side="left", padx=10)

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

        self.hint = ctk.CTkLabel(
            self,
            text="SHELVES: click polygon points for selected shelf (front-line optional)",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.hint.pack(pady=(0, 4))
        self.status = ctk.CTkLabel(self, text="")
        self.status.pack(pady=(0, 4))

        self.canvas = ctk.CTkCanvas(self, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._click)

        self._refresh_shelf_label()
        self._seek(0)

    def _active_shelf(self) -> Shelf:
        self.shelf_idx = min(max(0, self.shelf_idx), len(self.layout.shelves) - 1)
        return self.layout.shelves[self.shelf_idx]

    def _refresh_shelf_label(self) -> None:
        sh = self._active_shelf()
        mode = "front-line" if self.editing_front_line else "polygon"
        self.shelf_lbl.configure(text=f"[{self.shelf_idx + 1}/{len(self.layout.shelves)}] {sh.label()} edit={mode}")

    def _seek(self, value) -> None:
        idx = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
        self.frame_lbl.configure(text=str(idx))
        self._redraw()

    def _add_shelf(self) -> None:
        sid = self.layout.next_shelf_id()
        self.layout.shelves.append(Shelf(shelf_id=sid, name=f"Shelf {sid}"))
        self.shelf_idx = len(self.layout.shelves) - 1
        self._refresh_shelf_label()
        self._redraw()

    def _delete_shelf(self) -> None:
        if not self.layout.shelves:
            return
        self.layout.shelves.pop(self.shelf_idx)
        if not self.layout.shelves:
            sid = self.layout.next_shelf_id()
            self.layout.shelves = [Shelf(shelf_id=sid, name=f"Shelf {sid}")]
        self.shelf_idx = min(self.shelf_idx, len(self.layout.shelves) - 1)
        self._refresh_shelf_label()
        self._redraw()

    def _prev_shelf(self) -> None:
        self.shelf_idx = (self.shelf_idx - 1) % len(self.layout.shelves)
        self._refresh_shelf_label()
        self._redraw()

    def _next_shelf(self) -> None:
        self.shelf_idx = (self.shelf_idx + 1) % len(self.layout.shelves)
        self._refresh_shelf_label()
        self._redraw()

    def _toggle_mode(self) -> None:
        self.editing_front_line = not self.editing_front_line
        self._refresh_shelf_label()
        self._redraw()

    def _clear(self) -> None:
        sh = self._active_shelf()
        if self.editing_front_line:
            sh.front_line = []
        else:
            sh.polygon = []
        self._redraw()

    def _undo(self) -> None:
        sh = self._active_shelf()
        if self.editing_front_line:
            if sh.front_line:
                sh.front_line.pop()
        elif sh.polygon:
            sh.polygon.pop()
        self._redraw()

    def _click(self, event) -> None:
        if self.frame is None:
            return
        fh, fw = self.frame.shape[:2]
        x = int(min(max(0, event.x / max(self.scale, 1e-6)), fw - 1))
        y = int(min(max(0, event.y / max(self.scale, 1e-6)), fh - 1))
        sh = self._active_shelf()
        if self.editing_front_line:
            if len(sh.front_line) >= 2:
                sh.front_line = []
            sh.front_line.append([x, y])
        else:
            sh.polygon.append([x, y])
        self._redraw()

    def _save(self) -> None:
        ok, why = self.layout.ready()
        save_shelf_layout(self.video.name, self.layout)
        extra = "" if ok else f"  (incomplete: {why})"
        self.status.configure(text=f"Saved to pipelines/configs/shelf_zones.json{extra}")

    def _draw_poly(self, vis, pts, color, active: bool, name: str) -> None:
        arr = np.array(pts, np.int32)
        cv2.polylines(vis, [arr], len(pts) >= 3, color, 2 if active else 1)
        for p in pts:
            cv2.circle(vis, tuple(map(int, p)), 4, color, -1)
        cv2.putText(vis, name, tuple(arr[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _redraw(self) -> None:
        if self.frame is None:
            return
        vis = self.frame.copy()
        if len(self.store_boundary.inside) >= 2:
            self._draw_poly(vis, self.store_boundary.inside, (0, 220, 80), False, "INSIDE / SHOP")
        for i, sh in enumerate(self.layout.shelves):
            c = shelf_color(i)
            active_poly = i == self.shelf_idx and not self.editing_front_line
            active_line = i == self.shelf_idx and self.editing_front_line
            if len(sh.polygon) >= 2:
                self._draw_poly(vis, sh.polygon, c, active_poly, sh.label())
            if len(sh.front_line) == 2:
                p1, p2 = (tuple(map(int, p)) for p in sh.front_line)
                cv2.line(vis, p1, p2, c, 3 if active_line else 2)
                cv2.putText(vis, f"{sh.label()} FRONT", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)
            for p in sh.front_line:
                cv2.circle(vis, tuple(map(int, p)), 5, c, -1)

        self._refresh_shelf_label()
        ready = len(self.layout.ready_shelves())
        self.status.configure(text=f"shelves ready {ready}/{len(self.layout.shelves)}")
        self._show(vis)

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
    ap.add_argument("--video", type=Path, default=ROOT / "raw_videos" / "interior.mp4")
    args = ap.parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    ShelfApp(args.video.resolve()).mainloop()


if __name__ == "__main__":
    main()
