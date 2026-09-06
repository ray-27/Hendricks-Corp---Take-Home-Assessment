"""Shelf-face geometry: an explicit outward normal vector + interest zone.

Saved to `configs/shelf_faces.json` (path from `configs/paths.py`).

Why this shape, instead of a plain shelf polygon
--------------------------------------------------
`pipelines/shelf_interest` infers "is this person looking at the shelf" from
the nearest point on a shelf polygon/front-line, recomputed every frame from
the person's current foot position. That target point slides along the
shelf as the person walks, which makes the facing-angle check noisier than
it needs to be, and it conflates "how close" with "which way they must be
facing" into one soft distance heuristic.

This module asks the operator to mark, once, per shelf face:

  edge    2 points along the shelf's front edge, in image space. Just used
          to anchor the normal and for drawing; not used for distance.
  normal  a unit vector, perpendicular to `edge`, pointing away from the
          shelf into the area customers stand in. Fixed once per face, not
          recomputed per person -- this is the shelf's "line of sight".
  zone    a polygon: the area a customer must be standing in to be
          eligible for this shelf's interest at all. This is a hard
          point-in-polygon gate, not a soft distance-based cue.

A customer engages a shelf face when both are true:
  1. their foot point is inside `zone` (hard gate), and
  2. their facing/attention vector points back toward the shelf, i.e. it is
     close (in angle) to `-normal`.

When a customer is standing between two shelves whose zones overlap, this
resolves "which shelf are they interacting with" directly from whichever
face's normal is most opposed to their own facing vector -- the same
evidence a human reviewer would use, and the exact scenario the task brief
calls out.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import SHELF_FACES_JSON as CONFIG_PATH  # noqa: E402

FACE_COLORS = [
    (70, 70, 255),
    (40, 180, 255),
    (40, 210, 140),
    (220, 160, 40),
    (210, 80, 210),
    (180, 220, 60),
    (30, 130, 255),
    (160, 160, 160),
]


def face_color(idx: int) -> tuple[int, int, int]:
    return FACE_COLORS[idx % len(FACE_COLORS)]


def _as_poly(pts) -> np.ndarray | None:
    if not pts or len(pts) < 3:
        return None
    return np.array(pts, np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, np.float32)


def compute_normal(edge: list, hint_pt) -> np.ndarray:
    """Perpendicular to `edge`, on whichever side `hint_pt` is."""
    a = np.array(edge[0], np.float32)
    b = np.array(edge[1], np.float32)
    e = b - a
    n1 = _unit(np.array([-e[1], e[0]], np.float32))
    mid = (a + b) * 0.5
    to_hint = np.asarray(hint_pt, np.float32) - mid
    if float(n1 @ to_hint) >= 0:
        return n1
    return -n1


@dataclass
class ShelfFace:
    shelf_id: str
    name: str = ""
    edge: list = field(default_factory=list)  # [[x1,y1],[x2,y2]]
    normal: list = field(default_factory=list)  # [nx, ny], unit, image space
    zone: list = field(default_factory=list)  # polygon, 3+ points

    def label(self) -> str:
        return self.name.strip() or self.shelf_id

    def ready(self) -> bool:
        return len(self.edge) == 2 and len(self.normal) == 2 and len(self.zone) >= 3

    def normal_vec(self) -> np.ndarray:
        if len(self.normal) != 2:
            return np.zeros(2, np.float32)
        return _unit(np.array(self.normal, np.float32))

    def edge_midpoint(self) -> np.ndarray:
        if len(self.edge) != 2:
            return np.zeros(2, np.float32)
        return np.array(self.edge, np.float32).mean(axis=0)

    def in_zone(self, pt) -> bool:
        poly = _as_poly(self.zone)
        if poly is None:
            return False
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, fallback_id: str = "F1") -> "ShelfFace":
        return cls(
            shelf_id=str(d.get("shelf_id") or fallback_id),
            name=str(d.get("name") or ""),
            edge=[list(map(int, p)) for p in d.get("edge", [])],
            normal=[float(v) for v in d.get("normal", [])] if d.get("normal") else [],
            zone=[list(map(int, p)) for p in d.get("zone", [])],
        )


@dataclass
class ShelfFaceLayout:
    faces: list = field(default_factory=list)

    def face_by_id(self, shelf_id: str) -> ShelfFace | None:
        for f in self.faces:
            if f.shelf_id == shelf_id:
                return f
        return None

    def ready_faces(self) -> list[ShelfFace]:
        return [f for f in self.faces if f.ready()]

    def next_face_id(self) -> str:
        used = {f.shelf_id for f in self.faces}
        i = 1
        while f"F{i}" in used:
            i += 1
        return f"F{i}"

    def ready(self) -> tuple[bool, str]:
        if not self.ready_faces():
            return False, "no shelf faces drawn (need edge + normal + zone per face)"
        return True, "ok"

    def to_dict(self) -> dict:
        return {"faces": [f.to_dict() if isinstance(f, ShelfFace) else f for f in self.faces]}

    @classmethod
    def from_dict(cls, d: dict) -> "ShelfFaceLayout":
        raw = d.get("faces") or []
        faces = [
            ShelfFace.from_dict(f, fallback_id=f"F{i + 1}") if isinstance(f, dict) else f
            for i, f in enumerate(raw)
        ]
        return cls(faces=faces)


def load_shelf_faces(video_name: str) -> ShelfFaceLayout:
    if not CONFIG_PATH.exists():
        return ShelfFaceLayout()
    data = json.loads(CONFIG_PATH.read_text())
    if video_name not in data:
        return ShelfFaceLayout()
    return ShelfFaceLayout.from_dict(data[video_name])


def save_shelf_faces(video_name: str, layout: ShelfFaceLayout) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    data[video_name] = layout.to_dict()
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
