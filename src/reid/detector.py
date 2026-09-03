"""Person detector. YOLOv8n is used only to crop bodies for ReIdentificationNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def crop(self, frame_bgr: np.ndarray, pad_frac: float = 0.08) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        bw, bh = self.x2 - self.x1, self.y2 - self.y1
        px, py = int(bw * pad_frac), int(bh * pad_frac)
        x1 = max(0, self.x1 - px)
        y1 = max(0, self.y1 - py)
        x2 = min(w, self.x2 + px)
        y2 = min(h, self.y2 + py)
        return frame_bgr[y1:y2, x1:x2]


class PersonDetector:
    def __init__(
        self,
        weights: str | None = None,
        conf: float = 0.40,
        min_aspect: float = 1.15,
        min_h: int = 32,
        min_w: int = 16,
    ) -> None:
        if weights is None:
            weights = str(Path(__file__).resolve().parents[2] / "models" / "yolov8n.pt")
        self.model = YOLO(weights)
        self.conf = conf
        self.min_aspect = min_aspect
        self.min_h = min_h
        self.min_w = min_w

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        dets: list[Detection] = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None:
            return dets
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if x2 - x1 < self.min_w or y2 - y1 < self.min_h:
                continue
            # 1.15 filters chairs; 0.45 still allows a person bending over a shelf.
            if (y2 - y1) / max(x2 - x1, 1) < self.min_aspect:
                continue
            dets.append(Detection(x1, y1, x2, y2, score))
        return dets
