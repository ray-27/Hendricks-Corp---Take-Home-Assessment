#!/usr/bin/env python3
"""
Staff-enrolment GUI for the staff-customer interaction pipeline (Task 3).

One-time, one-step manual marking: scrub to a frame, click each staff
member's body once to enrol a ReID embedding into a small gallery, saved
to `pipelines/configs/staff_marks.json`, keyed by video filename. Click
the same person again on a different frame (different pose/lighting) to
add another embedding for them -- a few embeddings per staff member makes
the nearest-neighbour match in `scoring.update_role_reid` more robust than
a single click.

Why a manual click instead of automatic apron-colour or VLM detection:
both were tried (see `pipelines/staff_interaction/README.md`) and both
misclassified real customers as staff or real staff as customers on this
footage, because they were trying to answer "does this look like an
apron/staff member" in general. Pointing at the exact people who *are*
staff in this video removes that ambiguity entirely -- the classifier at
runtime only has to ask "does this look like one of *these* specific
people", a narrower and much better-conditioned question.

Usage:
    python3 pipelines/staff_interaction/staff_gui.py --video raw_videos/entrance.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from staff_interaction.staff_store import (  # noqa: E402
    StaffMarks,
    load_staff_marks,
    save_staff_marks,
)

DISPLAY_W = 960
CLICK_SLOP_PX = 6  # a "click" (not a drag) enrols the person under the pointer


class StaffGuiApp(ctk.CTk):
    def __init__(self, video: Path):
        super().__init__()
        self.title("Staff enrolment — click each staff member")
        self.geometry("1180x820")
        ctk.set_appearance_mode("dark")

        self.video = video
        self.cap = cv2.VideoCapture(str(video))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.marks = load_staff_marks(video.name)
        self.frame = None
        self.scale = 1.0
        self.photo = None
        self.pose = None
        self.reid = None
        self._press_pt = None  # canvas-space (x, y) on mouse-down
        self._dets_frame_id: int | None = None
        self._dets: list = []
        self._last_hit_box: tuple[int, int, int, int] | None = None

        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=8)
        ctk.CTkButton(bar, text="Undo last", width=100, command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Clear all", width=100, command=self._clear).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Save", width=90, command=self._save).pack(side="left", padx=12)
        ctk.CTkLabel(bar, text="saving to: pipelines/configs/staff_marks.json").pack(side="left", padx=12)

        nav = ctk.CTkFrame(self)
        nav.pack(fill="x", padx=8, pady=(0, 6))
        ctk.CTkLabel(nav, text="frame").pack(side="left", padx=(6, 4))
        self.slider = ctk.CTkSlider(nav, from_=0, to=max(1, self.total - 1), command=self._seek, width=760)
        self.slider.set(0)
        self.slider.pack(side="left", padx=4)
        self.frame_lbl = ctk.CTkLabel(nav, text="0")
        self.frame_lbl.pack(side="left", padx=8)

        self.hint = ctk.CTkLabel(
            self,
            text="Click each staff member's body once. Scrub frames to catch everyone; "
            "click the same person again on another frame for a more robust gallery.",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.hint.pack(pady=(0, 4))
        self.status = ctk.CTkLabel(self, text="")
        self.status.pack(pady=(0, 4))

        self.canvas = ctk.CTkCanvas(self, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        self._seek(0)
        self._load_models()

    # ----------------------------------------------------------------- nav

    def _seek(self, value) -> None:
        idx = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
        self.frame_lbl.configure(text=str(idx))
        self._redraw()

    def _clear(self) -> None:
        self.marks.gallery = []
        self._last_hit_box = None
        self._redraw()

    def _undo(self) -> None:
        if self.marks.gallery:
            self.marks.gallery.pop()
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
        self.status.configure(text="Models ready.")
        self._redraw()

    def _current_dets(self) -> list:
        """Pose detections for `self.frame`, cached until the frame changes."""
        if self.pose is None or self.frame is None:
            return []
        fid = id(self.frame)
        if self._dets_frame_id != fid:
            self._dets = self.pose.detect(self.frame)
            self._dets_frame_id = fid
        return self._dets

    # -------------------------------------------------------------- input

    def _to_frame_xy(self, cx: int, cy: int) -> tuple[int, int]:
        fh, fw = self.frame.shape[:2]
        x = int(min(max(0, cx / max(self.scale, 1e-6)), fw - 1))
        y = int(min(max(0, cy / max(self.scale, 1e-6)), fh - 1))
        return x, y

    def _press(self, event) -> None:
        if self.frame is None:
            return
        self._press_pt = (event.x, event.y)

    def _release(self, event) -> None:
        if self.frame is None or self._press_pt is None:
            return
        start = self._press_pt
        end = (event.x, event.y)
        self._press_pt = None
        dx, dy = abs(end[0] - start[0]), abs(end[1] - start[1])
        if dx <= CLICK_SLOP_PX and dy <= CLICK_SLOP_PX:
            x, y = self._to_frame_xy(*end)
            self._enrol_staff(x, y)
        self._redraw()

    def _enrol_staff(self, x: int, y: int) -> None:
        self._load_models()
        dets = self._current_dets()
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
        self.marks.gallery.append(emb[0].astype(float).tolist())
        self._last_hit_box = hit.box
        self.status.configure(text=f"Enrolled staff embedding #{len(self.marks.gallery)} (box height {hit.height}px)")

    def _save(self) -> None:
        ok, why = self.marks.ready()
        save_staff_marks(self.video.name, self.marks)
        extra = "" if ok else f"  (incomplete: {why})"
        self.status.configure(text=f"Saved to pipelines/configs/staff_marks.json{extra}")

    # ------------------------------------------------------------ drawing

    def _redraw(self) -> None:
        if self.frame is None:
            return
        vis = self.frame.copy()
        for det in self._current_dets():
            x1, y1, x2, y2 = det.box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (90, 90, 90), 1)
        if self._last_hit_box is not None:
            x1, y1, x2, y2 = self._last_hit_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 210, 140), 2)
        self._status_text()
        self._show(vis)

    def _status_text(self) -> None:
        self.status.configure(text=f"staff embeddings enrolled: {len(self.marks.gallery)}")

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
    StaffGuiApp(args.video.resolve()).mainloop()


if __name__ == "__main__":
    main()
