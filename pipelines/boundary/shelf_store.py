"""Shelf geometry contract used by per-shelf pipelines.

Saved to `pipelines/configs/shelf_zones.json`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "shelf_zones.json"
OLD_BOUNDARY_PATH = Path(__file__).resolve().parents[1] / "configs" / "boundary_zones.json"

SHELF_COLORS = [
    (70, 70, 255),
    (40, 180, 255),
    (40, 210, 140),
    (220, 160, 40),
    (210, 80, 210),
    (180, 220, 60),
    (30, 130, 255),
    (160, 160, 160),
]


def shelf_color(idx: int) -> tuple[int, int, int]:
    return SHELF_COLORS[idx % len(SHELF_COLORS)]


def _as_poly(pts) -> np.ndarray | None:
    if not pts or len(pts) < 3:
        return None
    return np.array(pts, np.float32)


def _closest_on_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-6:
        return a
    t = float((pt - a) @ ab) / denom
    t = min(max(t, 0.0), 1.0)
    return a + t * ab


def closest_on_poly(pt, pts) -> np.ndarray:
    p = np.asarray(pt, np.float32)
    arr = np.asarray(pts, np.float32)
    if len(arr) == 0:
        return p
    if len(arr) == 1:
        return arr[0]
    best, best_d = arr[0], float("inf")
    n = len(arr)
    closed = n >= 3
    for i in range(n if closed else n - 1):
        q = _closest_on_segment(p, arr[i], arr[(i + 1) % n])
        d = float(np.linalg.norm(p - q))
        if d < best_d:
            best, best_d = q, d
    return best


@dataclass
class Shelf:
    shelf_id: str
    name: str = ""
    polygon: list = field(default_factory=list)
    front_line: list = field(default_factory=list)

    def label(self) -> str:
        return self.name.strip() or self.shelf_id

    def centroid(self) -> np.ndarray:
        poly = _as_poly(self.polygon)
        if poly is not None:
            return poly.mean(axis=0)
        if self.front_line:
            return np.array(self.front_line, np.float32).mean(axis=0)
        return np.zeros(2, np.float32)

    def signed_dist(self, pt) -> float:
        poly = _as_poly(self.polygon)
        if poly is None:
            if len(self.front_line) == 2:
                q = _closest_on_segment(
                    np.asarray(pt, np.float32),
                    np.array(self.front_line[0], np.float32),
                    np.array(self.front_line[1], np.float32),
                )
                return float(np.linalg.norm(np.asarray(pt, np.float32) - q))
            return float("inf")
        return float(cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True) * -1.0)

    def look_target(self, from_pt) -> np.ndarray:
        p = np.asarray(from_pt, np.float32)
        if len(self.front_line) == 2:
            return _closest_on_segment(
                p,
                np.array(self.front_line[0], np.float32),
                np.array(self.front_line[1], np.float32),
            )
        if self.polygon:
            return closest_on_poly(p, self.polygon)
        return self.centroid()

    def ready(self) -> bool:
        return len(self.polygon) >= 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, fallback_id: str = "S1") -> "Shelf":
        return cls(
            shelf_id=str(d.get("shelf_id") or fallback_id),
            name=str(d.get("name") or ""),
            polygon=[list(map(int, p)) for p in d.get("polygon", [])],
            front_line=[list(map(int, p)) for p in d.get("front_line", [])],
        )


@dataclass
class ShelfLayout:
    shelves: list = field(default_factory=list)

    def shelf_by_id(self, shelf_id: str) -> Shelf | None:
        for s in self.shelves:
            if s.shelf_id == shelf_id:
                return s
        return None

    def ready_shelves(self) -> list[Shelf]:
        return [s for s in self.shelves if s.ready()]

    def next_shelf_id(self) -> str:
        used = {s.shelf_id for s in self.shelves}
        i = 1
        while f"S{i}" in used:
            i += 1
        return f"S{i}"

    def ready(self) -> tuple[bool, str]:
        if not self.ready_shelves():
            return False, "no shelves drawn (need at least one 3-point polygon)"
        return True, "ok"

    def to_dict(self) -> dict:
        return {"shelves": [s.to_dict() if isinstance(s, Shelf) else s for s in self.shelves]}

    @classmethod
    def from_dict(cls, d: dict) -> "ShelfLayout":
        raw_shelves = d.get("shelves") or []
        shelves = [
            Shelf.from_dict(s, fallback_id=f"S{i + 1}") if isinstance(s, dict) else s
            for i, s in enumerate(raw_shelves)
        ]
        return cls(shelves=shelves)


def _extract_from_old_boundary(video_name: str) -> ShelfLayout:
    if not OLD_BOUNDARY_PATH.exists():
        return ShelfLayout()
    data = json.loads(OLD_BOUNDARY_PATH.read_text())
    if video_name not in data:
        return ShelfLayout()
    old = data[video_name]
    if "shelves" not in old:
        return ShelfLayout()
    return ShelfLayout.from_dict({"shelves": old.get("shelves", [])})


def load_shelf_layout(video_name: str) -> ShelfLayout:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        if video_name in data:
            return ShelfLayout.from_dict(data[video_name])
    return _extract_from_old_boundary(video_name)


def save_shelf_layout(video_name: str, layout: ShelfLayout) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    data[video_name] = layout.to_dict()
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
