"""Detect people, embed with ReIdentificationNet, and assign persistent IDs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection, PersonDetector
from .embedder import ReIDEmbedder


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _greedy_match(cost: np.ndarray, max_cost: float) -> list[tuple[int, int, float]]:
    """Greedy assignment on a cost matrix. Returns (row, col, cost)."""
    if cost.size == 0:
        return []
    pairs: list[tuple[int, int, float]] = []
    used_r, used_c = set(), set()
    flat = [(cost[r, c], r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1])]
    flat.sort()
    for cval, r, c in flat:
        if cval > max_cost:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c, float(cval)))
    return pairs


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    score: float
    similarity: float
    hits: int = 1
    missed: int = 0
    thumbnail: np.ndarray | None = None
    color: tuple[int, int, int] = (0, 200, 255)


@dataclass
class FrameResult:
    frame_index: int
    annotated: np.ndarray
    tracks: list[Track]
    unique_ids: int


class ReIDPipeline:
    def __init__(
        self,
        match_threshold: float = 0.55,
        max_missed: int = 45,
        ema: float = 0.85,
    ) -> None:
        # Cosine similarity above this value is treated as the same person.
        # 0.55 is a starting point for TAO ReID on CCTV (different from Market-1501).
        self.match_threshold = match_threshold
        self.max_missed = max_missed
        self.ema = ema
        self.detector = PersonDetector()
        self.embedder = ReIDEmbedder()
        self.gallery: dict[int, Track] = {}
        self.next_id = 1
        self.frame_index = 0
        self._palette = _build_palette()

    def reset(self) -> None:
        self.gallery.clear()
        self.next_id = 1
        self.frame_index = 0

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        self.frame_index += 1
        detections = self.detector.detect(frame_bgr)
        crops = [d.crop(frame_bgr) for d in detections]
        embeddings = self.embedder.embed(crops) if crops else np.zeros((0, 256), dtype=np.float32)

        active = [t for t in self.gallery.values() if t.missed < self.max_missed]
        assigned_tracks: dict[int, Track] = {}
        unmatched_dets = set(range(len(detections)))

        if active and len(detections) > 0:
            sim = embeddings @ np.stack([t.embedding for t in active], axis=0).T
            ious = np.zeros((len(detections), len(active)), dtype=np.float32)
            for i, det in enumerate(detections):
                for j, tr in enumerate(active):
                    ious[i, j] = _iou(det.xyxy, tr.bbox)
            # Combined cost: appearance first, IoU as a short-term stabilizer.
            cost = 1.0 - (0.75 * sim + 0.25 * ious)
            max_cost = 1.0 - self.match_threshold
            for di, ti, cval in _greedy_match(cost, max_cost=max_cost):
                track = active[ti]
                det = detections[di]
                new_emb = embeddings[di]
                track.embedding = self.ema * track.embedding + (1.0 - self.ema) * new_emb
                n = np.linalg.norm(track.embedding)
                track.embedding = track.embedding / max(n, 1e-12)
                track.bbox = det.xyxy
                track.score = det.score
                track.similarity = float(sim[di, ti])
                track.hits += 1
                track.missed = 0
                track.thumbnail = _thumb(crops[di])
                assigned_tracks[track.track_id] = track
                unmatched_dets.discard(di)

        for di in sorted(unmatched_dets):
            det = detections[di]
            tid = self.next_id
            self.next_id += 1
            track = Track(
                track_id=tid,
                bbox=det.xyxy,
                embedding=embeddings[di],
                score=det.score,
                similarity=1.0,
                thumbnail=_thumb(crops[di]),
                color=self._palette[(tid - 1) % len(self._palette)],
            )
            self.gallery[tid] = track
            assigned_tracks[tid] = track

        for track in self.gallery.values():
            if track.track_id not in assigned_tracks:
                track.missed += 1

        live = [t for t in assigned_tracks.values()]
        annotated = self.draw(frame_bgr, live)
        unique = sum(1 for t in self.gallery.values() if t.hits >= 2 or t.missed == 0)
        return FrameResult(
            frame_index=self.frame_index,
            annotated=annotated,
            tracks=sorted(live, key=lambda t: t.track_id),
            unique_ids=unique,
        )

    def draw(self, frame_bgr: np.ndarray, tracks: list[Track]) -> np.ndarray:
        vis = frame_bgr.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = track.color
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.track_id}  {track.similarity:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        hud = [
            f"People in view: {len(tracks)}",
            f"Unique IDs: {sum(1 for t in self.gallery.values() if t.hits >= 1)}",
            f"Frame: {self.frame_index}",
            "Model: NVIDIA ReIdentificationNet ResNet50",
        ]
        y = 24
        for line in hud:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        return vis


def _thumb(crop: np.ndarray, size: tuple[int, int] = (72, 144)) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def _build_palette() -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(7)
    colors = []
    for _ in range(64):
        c = rng.integers(60, 255, size=3)
        colors.append((int(c[0]), int(c[1]), int(c[2])))
    return colors
