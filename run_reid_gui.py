#!/usr/bin/env python3
"""Interactive ReID viewer for NVIDIA ReIdentificationNet.

Usage:
    python3 run_reid_gui.py
    python3 run_reid_gui.py --video raw_videos/entrance.mp4

This Mac is Apple M1 (no NVIDIA GPU), so the official DeepStream/TensorRT path
cannot run. The deployable_v1.2 ONNX model is executed with ONNX Runtime instead.
People are detected with YOLOv8n, then each crop is embedded by ReIdentificationNet.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from reid.pipeline import ReIDPipeline  # noqa: E402


def list_videos() -> list[Path]:
    folder = ROOT / "raw_videos"
    return sorted(folder.glob("*.mp4"))


class ReIDApp(ctk.CTk):
    def __init__(self, video_path: Path) -> None:
        super().__init__()
        self.title("Hendricks ReID — NVIDIA ReIdentificationNet")
        self.geometry("1280x780")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_path = video_path
        self.pipeline: ReIDPipeline | None = None
        self.cap: cv2.VideoCapture | None = None
        self.playing = False
        self.busy = False
        self.photo = None
        self.gallery_photos: list[ImageTk.PhotoImage] = []
        self.skip = ctk.IntVar(value=2)
        self.threshold = ctk.DoubleVar(value=0.55)
        self.status = ctk.StringVar(value="Loading models…")

        self._build_ui()
        self.after(100, self._boot)

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=12, pady=8)

        videos = list_videos()
        names = [p.name for p in videos] or [self.video_path.name]
        self.video_menu = ctk.CTkOptionMenu(top, values=names, command=self._on_video_change)
        self.video_menu.set(self.video_path.name)
        self.video_menu.pack(side="left", padx=6)

        self.play_btn = ctk.CTkButton(top, text="Play", width=90, command=self.toggle_play)
        self.play_btn.pack(side="left", padx=6)
        ctk.CTkButton(top, text="Step", width=70, command=self.step_once).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Reset IDs", width=90, command=self.reset_ids).pack(side="left", padx=6)

        ctk.CTkLabel(top, text="Skip").pack(side="left", padx=(16, 4))
        ctk.CTkSlider(top, from_=1, to=8, number_of_steps=7, variable=self.skip, width=110).pack(
            side="left"
        )
        ctk.CTkLabel(top, text="Match").pack(side="left", padx=(16, 4))
        ctk.CTkSlider(top, from_=0.30, to=0.80, variable=self.threshold, width=140).pack(side="left")

        ctk.CTkLabel(top, textvariable=self.status).pack(side="left", padx=16)

        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.video_label = ctk.CTkLabel(body, text="")
        self.video_label.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)

        right = ctk.CTkFrame(body, width=280)
        right.pack(side="right", fill="y", padx=(4, 8), pady=8)
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="Identities", font=ctk.CTkFont(size=16, weight="bold")).pack(
            pady=(8, 4)
        )
        self.gallery_box = ctk.CTkScrollableFrame(right, width=250)
        self.gallery_box.pack(fill="both", expand=True, padx=6, pady=6)

    def _boot(self) -> None:
        def work() -> None:
            self.pipeline = ReIDPipeline(match_threshold=float(self.threshold.get()))
            self.cap = cv2.VideoCapture(str(self.video_path))
            self.status.set(f"Ready · {self.video_path.name} · ReID {self.pipeline.embedder.device_name}")
            self.after(0, self.step_once)

        threading.Thread(target=work, daemon=True).start()

    def _on_video_change(self, name: str) -> None:
        path = ROOT / "raw_videos" / name
        self.playing = False
        self.play_btn.configure(text="Play")
        self.video_path = path
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        if self.pipeline:
            self.pipeline.reset()
            self.pipeline.match_threshold = float(self.threshold.get())
        self.status.set(f"Loaded {name}")
        self.step_once()

    def reset_ids(self) -> None:
        if self.pipeline:
            self.pipeline.reset()
            self.pipeline.match_threshold = float(self.threshold.get())
        self.status.set("Gallery cleared")

    def toggle_play(self) -> None:
        if self.pipeline is None or self.cap is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._loop()

    def step_once(self) -> None:
        if self.busy or self.pipeline is None or self.cap is None:
            return
        self.busy = True
        skip = max(1, int(self.skip.get()))
        frame = None
        for _ in range(skip):
            ok, frame = self.cap.read()
            if not ok:
                self.playing = False
                self.play_btn.configure(text="Play")
                self.status.set("End of video")
                self.busy = False
                return
        assert frame is not None
        self.pipeline.match_threshold = float(self.threshold.get())
        result = self.pipeline.process_frame(frame)
        self._show_frame(result.annotated)
        self._show_gallery(result.tracks, result.unique_ids)
        self.status.set(
            f"{self.video_path.name}  frame {result.frame_index}  "
            f"in-view {len(result.tracks)}  unique {result.unique_ids}  "
            f"match≥{self.threshold.get():.2f}"
        )
        self.busy = False

    def _loop(self) -> None:
        if not self.playing:
            return
        self.step_once()
        self.after(10, self._loop)

    def _show_frame(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w, max_h = 920, 680
        scale = min(max_w / w, max_h / h)
        img = Image.fromarray(rgb).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.photo)

    def _show_gallery(self, tracks, unique_ids: int) -> None:
        for child in self.gallery_box.winfo_children():
            child.destroy()
        self.gallery_photos.clear()
        ctk.CTkLabel(
            self.gallery_box,
            text=f"{len(tracks)} in view · {unique_ids} unique",
        ).pack(anchor="w", pady=(0, 8))
        for track in tracks:
            row = ctk.CTkFrame(self.gallery_box)
            row.pack(fill="x", pady=4)
            if track.thumbnail is not None:
                thumb = cv2.cvtColor(track.thumbnail, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(thumb)
                photo = ImageTk.PhotoImage(im)
                self.gallery_photos.append(photo)
                ctk.CTkLabel(row, image=photo, text="").pack(side="left", padx=4, pady=4)
            info = (
                f"ID {track.track_id}\n"
                f"sim {track.similarity:.2f}\n"
                f"hits {track.hits}"
            )
            ctk.CTkLabel(row, text=info, justify="left").pack(side="left", padx=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReIdentificationNet video GUI")
    default = ROOT / "raw_videos" / "entrance.mp4"
    parser.add_argument("--video", type=Path, default=default, help="Path to an mp4 clip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")
    app = ReIDApp(args.video.resolve())
    app.mainloop()


if __name__ == "__main__":
    main()
