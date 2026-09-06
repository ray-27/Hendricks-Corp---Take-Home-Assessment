#!/usr/bin/env python3
"""
Shelf-face marking GUI: edge -> outward normal -> customer interest zone.

Per shelf face, in order:
  1. edge    click 2 points along the shelf's front edge
  2. normal  click 1 point on the customer side; the perpendicular to the
             edge pointing toward that click is stored as the face normal
             (click again anywhere to flip which side it points to)
  3. zone    click 3+ points: the polygon a customer must stand in to be
             eligible for this shelf's interest at all

Writes `configs/shelf_faces.json`, independent of every other
boundary/shelf config file in this repo.

Usage:
    python3 pipelines/shelf_vector_interest/shelf_face_gui.py --video raw_videos/interior.mp4
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

from configs.paths import INTERIOR_VIDEO, SHELF_FACES_JSON  # noqa: E402
from shelf_vector_interest.shelf_face_store import (  # noqa: E402
    ShelfFace,
    ShelfFaceLayout,
    compute_normal,
    face_color,
    load_shelf_faces,
    save_shelf_faces,
)

DISPLAY_W = 960
PARTS = [
    ("edge", "1/3  EDGE: click 2 points along the shelf's front edge"),
    ("normal", "2/3  NORMAL: click once on the customer side (click again to flip)"),
    ("zone", "3/3  ZONE: click 3+ points for the customer interest area"),
]


class ShelfFaceApp(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("Shelf face marker — edge / normal / customer zone")
        self.geometry("1180x820")
        ctk.set_appearance_mode("dark")

        self.video = video
        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.layout = load_shelf_faces(video.name)
        if not self.layout.faces:
            fid = self.layout.next_face_id()
            self.layout.faces = [ShelfFace(shelf_id=fid, name=f"Shelf {fid}")]
        self.face_idx = 0
        self.part = 0
        self.frame = None
        self.scale = 1.0
        self.photo = None

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(bar, text="< Back part", width=100, command=self._back_part).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Next part >", width=100, command=self._next_part).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Undo point", width=100, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear part", width=100, command=self._clear).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save", width=90, command=self._save).pack(side="left", padx=12)
        ctk.CTkLabel(bar, text=f"saving to: {SHELF_FACES_JSON}").pack(side="left", padx=12)

        face_bar = ctk.CTkFrame(self)
        face_bar.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(face_bar, text="shelf face").pack(side="left", padx=(6, 4))
        ctk.CTkButton(face_bar, text="< Prev", width=80, command=self._prev_face).pack(side="left", padx=4)
        ctk.CTkButton(face_bar, text="Next >", width=80, command=self._next_face).pack(side="left", padx=4)
        ctk.CTkButton(face_bar, text="+ Add face", width=100, command=self._add_face).pack(side="left", padx=4)
        ctk.CTkButton(face_bar, text="- Delete face", width=110, command=self._delete_face).pack(side="left", padx=4)
        self.face_lbl = ctk.CTkLabel(face_bar, text="")
        self.face_lbl.pack(side="left", padx=10)

        nav = ctk.CTkFrame(self)
        nav.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(nav, text="frame").pack(side="left", padx=(6, 4))
        self.slider = ctk.CTkSlider(nav, from_=0, to=max(1, self.total - 1), command=self._seek, width=760)
        self.slider.set(0)
        self.slider.pack(side="left", padx=4)
        self.frame_lbl = ctk.CTkLabel(nav, text="0")
        self.frame_lbl.pack(side="left", padx=8)

        self.hint = ctk.CTkLabel(self, text=PARTS[0][1], font=ctk.CTkFont(size=15, weight="bold"))
        self.hint.pack(pady=(0, 4))
        self.status = ctk.CTkLabel(self, text="")
        self.status.pack(pady=(0, 4))

        self.canvas = ctk.CTkCanvas(self, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._click)

        self._refresh_labels()
        self._seek(0)

    # ------------------------------------------------------------- state

    def _active_face(self) -> ShelfFace:
        self.face_idx = min(max(0, self.face_idx), len(self.layout.faces) - 1)
        return self.layout.faces[self.face_idx]

    def _part_key(self) -> str:
        return PARTS[self.part][0]

    def _refresh_labels(self) -> None:
        f = self._active_face()
        self.face_lbl.configure(
            text=f"[{self.face_idx + 1}/{len(self.layout.faces)}] {f.label()}  "
            f"edge={len(f.edge)}/2 normal={'set' if len(f.normal) == 2 else 'unset'} zone={len(f.zone)} pts"
        )
        self.hint.configure(text=PARTS[self.part][1])

    def _seek(self, value) -> None:
        idx = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
        self.frame_lbl.configure(text=str(idx))
        self._redraw()

    # ------------------------------------------------------------- faces

    def _add_face(self) -> None:
        fid = self.layout.next_face_id()
        self.layout.faces.append(ShelfFace(shelf_id=fid, name=f"Shelf {fid}"))
        self.face_idx = len(self.layout.faces) - 1
        self.part = 0
        self._refresh_labels()
        self._redraw()

    def _delete_face(self) -> None:
        if not self.layout.faces:
            return
        self.layout.faces.pop(self.face_idx)
        if not self.layout.faces:
            fid = self.layout.next_face_id()
            self.layout.faces = [ShelfFace(shelf_id=fid, name=f"Shelf {fid}")]
        self.face_idx = min(self.face_idx, len(self.layout.faces) - 1)
        self._refresh_labels()
        self._redraw()

    def _prev_face(self) -> None:
        self.face_idx = (self.face_idx - 1) % len(self.layout.faces)
        self._refresh_labels()
        self._redraw()

    def _next_face(self) -> None:
        self.face_idx = (self.face_idx + 1) % len(self.layout.faces)
        self._refresh_labels()
        self._redraw()

    # -------------------------------------------------------------- parts

    def _next_part(self) -> None:
        self.part = min(self.part + 1, len(PARTS) - 1)
        self._refresh_labels()
        self._redraw()

    def _back_part(self) -> None:
        self.part = max(self.part - 1, 0)
        self._refresh_labels()
        self._redraw()

    def _clear(self) -> None:
        f = self._active_face()
        key = self._part_key()
        if key == "edge":
            f.edge = []
            f.normal = []  # normal depends on edge
        elif key == "normal":
            f.normal = []
        elif key == "zone":
            f.zone = []
        self._redraw()

    def _undo(self) -> None:
        f = self._active_face()
        key = self._part_key()
        if key == "edge" and f.edge:
            f.edge.pop()
            f.normal = []
        elif key == "normal":
            f.normal = []
        elif key == "zone" and f.zone:
            f.zone.pop()
        self._redraw()

    # -------------------------------------------------------------- input

    def _click(self, event) -> None:
        if self.frame is None:
            return
        fh, fw = self.frame.shape[:2]
        x = int(min(max(0, event.x / max(self.scale, 1e-6)), fw - 1))
        y = int(min(max(0, event.y / max(self.scale, 1e-6)), fh - 1))
        f = self._active_face()
        key = self._part_key()

        if key == "edge":
            if len(f.edge) >= 2:
                f.edge = []
                f.normal = []
            f.edge.append([x, y])
            if len(f.edge) == 2:
                self.part = 1  # auto-advance to normal
        elif key == "normal":
            if len(f.edge) != 2:
                self.status.configure(text="Draw the edge first (2 points).")
                return
            f.normal = compute_normal(f.edge, [x, y]).tolist()
        elif key == "zone":
            f.zone.append([x, y])

        self._refresh_labels()
        self._redraw()

    def _save(self) -> None:
        ok, why = self.layout.ready()
        save_shelf_faces(self.video.name, self.layout)
        extra = "" if ok else f"  (incomplete: {why})"
        ready = len(self.layout.ready_faces())
        self.status.configure(
            text=f"Saved to {SHELF_FACES_JSON}{extra}   faces ready={ready}/{len(self.layout.faces)}"
        )

    # ------------------------------------------------------------ drawing

    def _redraw(self) -> None:
        if self.frame is None:
            return
        vis = self.frame.copy()
        for i, f in enumerate(self.layout.faces):
            c = face_color(i)
            active = i == self.face_idx
            if len(f.edge) >= 1:
                for p in f.edge:
                    cv2.circle(vis, tuple(map(int, p)), 4, c, -1)
            if len(f.edge) == 2:
                a, b = (tuple(map(int, p)) for p in f.edge)
                cv2.line(vis, a, b, c, 3 if active else 1)
                cv2.putText(vis, f.label(), a, cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                if len(f.normal) == 2:
                    mid = f.edge_midpoint()
                    tip = mid + f.normal_vec() * 70.0
                    cv2.arrowedLine(
                        vis,
                        tuple(map(int, mid)),
                        tuple(map(int, tip)),
                        c,
                        3 if active else 2,
                        tipLength=0.25,
                    )
            if len(f.zone) >= 2:
                arr = np.array(f.zone, np.int32)
                cv2.polylines(vis, [arr], len(f.zone) >= 3, c, 2 if active else 1)
                for p in f.zone:
                    cv2.circle(vis, tuple(map(int, p)), 3, c, -1)

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
    ap.add_argument("--video", type=Path, default=INTERIOR_VIDEO)
    args = ap.parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    ShelfFaceApp(args.video.resolve()).mainloop()


if __name__ == "__main__":
    main()
