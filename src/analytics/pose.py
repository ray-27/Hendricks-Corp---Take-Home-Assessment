"""YOLO-pose person detector plus the keypoint geometry the events need.

Default weights are YOLO11x-pose (same COCO-17 keypoint layout as YOLOv8-pose).
Nano (`yolov8n-pose.pt`) is still valid via `--pose-weights` if you need speed.

Why pose instead of a gaze/head-pose model: in entrance.mp4 bodies are ~100-210
px tall but heads are only ~20 px ear-to-ear. Gaze networks (L2CS-Net, 6DRepNet)
need a 60 px+ face, so they would return confident noise here. Shoulder and hip
keypoints are stable at this scale, so body orientation carries the signal and
the head only refines it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import MODELS_DIR, POSE_DET_CONF, POSE_IMGSZ, POSE_KP_CONF, POSE_KPT_GATE_H, POSE_MIN_H, POSE_MIN_KPTS, POSE_MIN_W, POSE_WEIGHTS_NAME  # noqa: E402

PERSON_CLASS_ID = 0

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_ANK, R_ANK = 15, 16

KP_CONF = POSE_KP_CONF  # below this a keypoint is treated as missing rather than wrong

# Shared default for every pipeline. Local file wins if present; otherwise
# Ultralytics downloads the checkpoint on first use.
DEFAULT_POSE_WEIGHTS = POSE_WEIGHTS_NAME


def default_pose_weights() -> str:
    local = MODELS_DIR / POSE_WEIGHTS_NAME
    return str(local) if local.exists() else POSE_WEIGHTS_NAME


def best_device() -> str:
    """CUDA -> MPS (Apple Silicon) -> CPU, in that order."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2, np.float32)


@dataclass
class PoseDet:
    box: tuple[int, int, int, int]
    score: float
    kpts: np.ndarray  # (17, 2) image coords
    kconf: np.ndarray  # (17,)

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    def has(self, *idx: int) -> bool:
        return all(self.kconf[i] >= KP_CONF for i in idx)

    def foot_point(self) -> np.ndarray:
        """Where the person stands. This is the point projected onto the floor."""
        if self.has(L_ANK, R_ANK):
            return ((self.kpts[L_ANK] + self.kpts[R_ANK]) * 0.5).astype(np.float32)
        if self.has(L_ANK):
            return self.kpts[L_ANK].astype(np.float32)
        if self.has(R_ANK):
            return self.kpts[R_ANK].astype(np.float32)
        x1, _y1, x2, y2 = self.box
        return np.array([(x1 + x2) * 0.5, y2], np.float32)

    def torso_quad(self) -> np.ndarray | None:
        """Shoulders-to-hips polygon: the apron region, with far less background
        than the bounding box would include."""
        if not self.has(L_SHO, R_SHO):
            return None
        if self.has(L_HIP, R_HIP):
            quad = [self.kpts[L_SHO], self.kpts[R_SHO], self.kpts[R_HIP], self.kpts[L_HIP]]
        else:
            # No hips visible: extend one shoulder-width downward as the torso.
            span = float(np.linalg.norm(self.kpts[L_SHO] - self.kpts[R_SHO]))
            drop = np.array([0.0, max(span, 8.0)], np.float32)
            quad = [
                self.kpts[L_SHO],
                self.kpts[R_SHO],
                self.kpts[R_SHO] + drop,
                self.kpts[L_SHO] + drop,
            ]
        return np.array(quad, np.int32)

    def faces_camera(self) -> bool | None:
        """A person facing the camera shows their anatomical left shoulder on the
        viewer's right, so x[L_SHO] > x[R_SHO]."""
        if self.has(L_SHO, R_SHO):
            return bool(self.kpts[L_SHO][0] > self.kpts[R_SHO][0])
        if self.has(L_HIP, R_HIP):
            return bool(self.kpts[L_HIP][0] > self.kpts[R_HIP][0])
        return None

    def facing_vector(self) -> np.ndarray:
        """Unit vector for where the torso points, in image space.

        Perpendicular to the shoulder line. The front/back ambiguity is resolved
        with faces_camera(); "toward the camera" is +y because this camera looks
        down at the walkway, so nearer ground is lower in the frame.
        """
        if self.has(L_SHO, R_SHO):
            a, b = self.kpts[L_SHO], self.kpts[R_SHO]
        elif self.has(L_HIP, R_HIP):
            a, b = self.kpts[L_HIP], self.kpts[R_HIP]
        else:
            return np.zeros(2, np.float32)
        span = _unit((b - a).astype(np.float32))
        normal = np.array([-span[1], span[0]], np.float32)
        toward = self.faces_camera()
        if toward is None:
            return np.zeros(2, np.float32)
        if (normal[1] > 0) != toward:
            normal = -normal
        return _unit(normal)

    def head_yaw(self) -> float:
        """Coarse head turn relative to the torso, in [-1, 1].

        Nose offset from the shoulder midpoint, normalised by shoulder width. At
        20 px heads this is all that survives; it is not a calibrated angle.
        """
        if not self.has(L_SHO, R_SHO, NOSE):
            return 0.0
        mid = (self.kpts[L_SHO] + self.kpts[R_SHO]) * 0.5
        span = float(np.linalg.norm(self.kpts[L_SHO] - self.kpts[R_SHO]))
        if span < 4.0:
            return 0.0
        return float(np.clip((self.kpts[NOSE][0] - mid[0]) / (0.5 * span), -1.0, 1.0))

    def attention_vector(self, max_yaw_deg: float = 60.0) -> np.ndarray:
        """Torso direction rotated by the head yaw: where attention points."""
        facing = self.facing_vector()
        if not facing.any():
            return facing
        theta = np.deg2rad(self.head_yaw() * max_yaw_deg)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        rot = np.array([[c, -s], [s, c]], np.float32)
        return _unit(rot @ facing)

    def apron_fraction(self, frame_bgr: np.ndarray, hsv_range: dict) -> float | None:
        """Fraction of torso pixels inside the sampled apron colour range."""
        quad = self.torso_quad()
        if quad is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1 = np.clip(quad.min(axis=0), [0, 0], [w - 1, h - 1])
        x2, y2 = np.clip(quad.max(axis=0) + 1, [1, 1], [w, h])
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        patch = frame_bgr[y1:y2, x1:x2]
        mask = np.zeros(patch.shape[:2], np.uint8)
        cv2.fillPoly(mask, [quad - np.array([x1, y1], np.int32)], 255)
        if int(mask.sum()) == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        lo = np.array([hsv_range["h"][0], hsv_range["s"][0], hsv_range["v"][0]], np.uint8)
        hi = np.array([hsv_range["h"][1], hsv_range["s"][1], hsv_range["v"][1]], np.uint8)
        hit = cv2.inRange(hsv, lo, hi)
        hit = cv2.bitwise_and(hit, hit, mask=mask)
        return float(np.count_nonzero(hit)) / float(np.count_nonzero(mask))

    def crop(self, frame_bgr: np.ndarray, pad_frac: float = 0.06) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.box
        px, py = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
        return frame_bgr[max(0, y1 - py) : min(h, y2 + py), max(0, x1 - px) : min(w, x2 + px)]


class PoseDetector:
    def __init__(
        self,
        weights: str | None = None,
        conf: float = POSE_DET_CONF,
        min_h: int = POSE_MIN_H,
        min_w: int = POSE_MIN_W,
        imgsz: int = POSE_IMGSZ,
        kpt_gate_h: int = POSE_KPT_GATE_H,
        min_kpts: int = POSE_MIN_KPTS,
        device: str | None = None,
    ) -> None:
        if weights is None:
            weights = default_pose_weights()
        self.model = YOLO(weights)
        self.device = device or best_device()
        try:
            self.model.to(self.device)
        except Exception as exc:  # MPS/CUDA can be present but fail to init for this op set
            print(f"{self.device} unavailable for pose model ({exc.__class__.__name__}); falling back to CPU.")
            self.device = "cpu"
        print(f"PoseDetector weights: {weights}")
        print(f"PoseDetector device: {self.device}")
        self.conf = conf
        self.min_h = min_h
        self.min_w = min_w
        # Inferring at the native 1280 matters here: the default 640 halves every
        # person, and the walkway crowd is only ~60 px tall to begin with. On a
        # sample of frames this doubled the detections and dropped the highest
        # detected foot position from y=452 to y=205.
        self.imgsz = imgsz
        # The shoe display shelves are detected as people all video long at
        # conf 0.26-0.46, and they never carry keypoints. Real people this large
        # have confident shoulders 99-100% of the time, so requiring keypoints
        # above a size gate drops the shelves without costing distant walkers,
        # which are the detections that legitimately lack keypoints.
        self.kpt_gate_h = kpt_gate_h
        self.min_kpts = min_kpts

    def detect(self, frame_bgr: np.ndarray) -> list[PoseDet]:
        res = self.model.predict(
            frame_bgr,
            conf=self.conf,
            classes=[PERSON_CLASS_ID],
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: list[PoseDet] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        kxy = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        kcf = (
            r.keypoints.conf.cpu().numpy()
            if (r.keypoints is not None and r.keypoints.conf is not None)
            else None
        )
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
            if x2 - x1 < self.min_w or y2 - y1 < self.min_h:
                continue
            kp = kxy[i] if kxy is not None and i < len(kxy) else np.zeros((17, 2), np.float32)
            kc = kcf[i] if kcf is not None and i < len(kcf) else np.zeros(17, np.float32)
            needed = self.min_kpts if (y2 - y1) >= self.kpt_gate_h else 1
            if int(np.count_nonzero(np.asarray(kc) >= KP_CONF)) < needed:
                continue
            out.append(
                PoseDet(
                    box=(x1, y1, x2, y2),
                    score=float(box.conf[0].cpu().numpy()),
                    kpts=np.asarray(kp, np.float32),
                    kconf=np.asarray(kc, np.float32),
                )
            )
        return out
