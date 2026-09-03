"""Scene geometry for the entrance analytics.

The camera is fixed, so all of this is drawn once by run_entrance_setup.py and
stored in configs/entrance_zones.json:

  exterior       walkway in front of the shop (where "interest" is judged)
  interior       inside the store (used for the "entered" decision)
  entrance_line  the door threshold, also the storefront point people look at
  floor_pts      4 floor corners of a known rectangle -> homography -> metres
  apron_hsv      colour range sampled from a staff apron
  staff_gallery  ReID embeddings of staff enrolled by clicking them

The homography is what makes every distance and speed threshold expressible in
metres instead of pixels, which perspective would otherwise ruin.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "entrance_zones.json"


def _as_poly(pts) -> np.ndarray | None:
    if not pts or len(pts) < 3:
        return None
    return np.array(pts, np.float32)


@dataclass
class Scene:
    exterior: list = field(default_factory=list)
    interior: list = field(default_factory=list)
    entrance_line: list = field(default_factory=list)
    floor_pts: list = field(default_factory=list)
    floor_w_m: float = 2.0
    floor_h_m: float = 2.0
    apron_hsv: dict | None = None
    staff_gallery: list = field(default_factory=list)

    # ---------------------------------------------------------------- homography

    @property
    def has_homography(self) -> bool:
        return len(self.floor_pts) == 4

    @property
    def H(self) -> np.ndarray | None:
        """Image -> floor plane (metres). Click order: near-L, near-R, far-R, far-L."""
        if not self.has_homography:
            return None
        src = np.array(self.floor_pts, np.float32)
        w, h = float(self.floor_w_m), float(self.floor_h_m)
        dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], np.float32)
        return cv2.getPerspectiveTransform(src, dst)

    def to_world(self, pt) -> np.ndarray | None:
        """Project an image point onto the floor plane, in metres."""
        H = self.H
        if H is None:
            return None
        src = np.array([[[float(pt[0]), float(pt[1])]]], np.float32)
        out = cv2.perspectiveTransform(src, H)
        return out[0, 0].astype(np.float32)

    # -------------------------------------------------------------------- zones

    def in_exterior(self, pt) -> bool:
        poly = _as_poly(self.exterior)
        if poly is None:
            return True
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def in_interior(self, pt) -> bool:
        poly = _as_poly(self.interior)
        if poly is None:
            return False
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

    def storefront_target(self, pt) -> np.ndarray:
        """Closest point on the storefront that this person could be looking at.

        Using the nearest point on the threshold segment (rather than one fixed
        centroid) keeps the "is he facing the shop" angle sensible for people
        standing at either end of the facade.
        """
        if len(self.entrance_line) == 2:
            a = np.array(self.entrance_line[0], np.float32)
            b = np.array(self.entrance_line[1], np.float32)
            ab = b - a
            denom = float(ab @ ab)
            if denom < 1e-6:
                return a
            t = float((np.array(pt, np.float32) - a) @ ab) / denom
            t = min(max(t, 0.0), 1.0)
            return a + t * ab
        poly = _as_poly(self.interior)
        if poly is not None:
            return poly.mean(axis=0)
        return np.array(pt, np.float32)

    # ------------------------------------------------------------------- serial

    def to_dict(self) -> dict:
        d = asdict(self)
        d["staff_gallery"] = [list(map(float, e)) for e in self.staff_gallery]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        return cls(
            exterior=[list(map(int, p)) for p in d.get("exterior", [])],
            interior=[list(map(int, p)) for p in d.get("interior", [])],
            entrance_line=[list(map(int, p)) for p in d.get("entrance_line", [])],
            floor_pts=[list(map(int, p)) for p in d.get("floor_pts", [])],
            floor_w_m=float(d.get("floor_w_m", 2.0)),
            floor_h_m=float(d.get("floor_h_m", 2.0)),
            apron_hsv=d.get("apron_hsv"),
            staff_gallery=[np.array(e, np.float32) for e in d.get("staff_gallery", [])],
        )

    @property
    def staff_matrix(self) -> np.ndarray | None:
        if not self.staff_gallery:
            return None
        return np.stack([np.asarray(e, np.float32) for e in self.staff_gallery], axis=0)

    def ready(self) -> tuple[bool, str]:
        if len(self.exterior) < 3:
            return False, "exterior zone not drawn"
        if len(self.interior) < 3:
            return False, "interior zone not drawn"
        if len(self.entrance_line) != 2:
            return False, "entrance line not drawn"
        return True, "ok"


def load_scene(video_name: str) -> Scene:
    if not CONFIG_PATH.exists():
        return Scene()
    data = json.loads(CONFIG_PATH.read_text())
    if video_name not in data:
        return Scene()
    return Scene.from_dict(data[video_name])


def save_scene(video_name: str, scene: Scene) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    data[video_name] = scene.to_dict()
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
