"""Boundary/zone data model shared across the independent pipelines.

Design intent
-------------
Each pipeline under `pipelines/` (starting with `interest`) is meant to run
on its own -- its own detector, its own tracker, its own output files -- and
be combined with the others only afterwards (by joining outputs on
`track_id` / timestamp, or simply by looking at them side by side). Because
the pipelines don't share a process or in-memory state, the boundary the
user draws has to be handed over as a *file*, not a Python object passed
between modules.

This module is that file-based contract: `boundary_gui.py` writes it,
every downstream pipeline reads it. The file lives at
`pipelines/configs/boundary_zones.json`, keyed by video filename so more
than one camera/clip can be configured at once (mirrors the convention
already used by `configs/entrance_zones.json` at the project root).

Two areas are recorded, in plain "outside/inside the shop" language:

  outside         the walking area in front of the shop, i.e. the public
                  walkway a passer-by uses. This is the zone in which
                  "interest" is judged -- the brief's passers-by are people
                  out here, not people already inside.
  inside          the interior of the store. Crossing into this polygon and
                  staying is what later pipelines use to decide "entered".
  entrance_line   optional 2 points across the doorway/threshold. When
                  present it is the point people are considered to be
                  looking/walking *at*; when absent, pipelines fall back to
                  the centroid of `inside`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "boundary_zones.json"


def _as_poly(pts) -> np.ndarray | None:
    if not pts or len(pts) < 3:
        return None
    return np.array(pts, np.float32)


@dataclass
class Boundary:
    outside: list = field(default_factory=list)
    inside: list = field(default_factory=list)
    entrance_line: list = field(default_factory=list)

    # -------------------------------------------------------------- queries

    def in_outside(self, pt) -> bool:
        poly = _as_poly(self.outside)
        if poly is None:
            return True
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def in_inside(self, pt) -> bool:
        poly = _as_poly(self.inside)
        if poly is None:
            return False
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def entrance_target(self, pt) -> np.ndarray:
        """Point on the storefront/threshold this person is closest to."""
        if len(self.entrance_line) == 2:
            a = np.array(self.entrance_line[0], np.float32)
            b = np.array(self.entrance_line[1], np.float32)
            ab = b - a
            denom = float(ab @ ab)
            if denom > 1e-6:
                t = float((np.array(pt, np.float32) - a) @ ab) / denom
                t = min(max(t, 0.0), 1.0)
                return a + t * ab
            return a
        poly = _as_poly(self.inside)
        if poly is not None:
            return poly.mean(axis=0)
        return np.array(pt, np.float32)

    def ready(self) -> tuple[bool, str]:
        if len(self.outside) < 3:
            return False, "outside/walking area not drawn"
        if len(self.inside) < 3:
            return False, "inside/shop area not drawn"
        return True, "ok"

    # -------------------------------------------------------------- serial

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Boundary":
        return cls(
            outside=[list(map(int, p)) for p in d.get("outside", [])],
            inside=[list(map(int, p)) for p in d.get("inside", [])],
            entrance_line=[list(map(int, p)) for p in d.get("entrance_line", [])],
        )


def load_boundary(video_name: str) -> Boundary:
    if not CONFIG_PATH.exists():
        return Boundary()
    data = json.loads(CONFIG_PATH.read_text())
    if video_name not in data:
        return Boundary()
    return Boundary.from_dict(data[video_name])


def save_boundary(video_name: str, boundary: Boundary) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    data[video_name] = boundary.to_dict()
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
