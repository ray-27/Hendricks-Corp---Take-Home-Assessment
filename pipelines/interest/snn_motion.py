"""Per-person spiking motion sensor (the "SNN" half of the interest layer).

This reuses the DVS-style event simulation + leaky-integrate-and-fire (LIF)
neuron idea already prototyped at the project root in `video_snn.py`, but
scopes it down from "one neuron per pixel of the whole frame" to "a small
neuron grid over one tracked person's crop". That gives every track its own
independent spiking retina and its own membrane state, which is what lets
this module report a per-person motion signal instead of one global blob.

Why this instead of just differencing bounding-box centres frame to frame:

  - It is robust to a jittery box (pose/tracker noise moves the box edges by
    a few pixels every frame; an SNN neuron with a firing threshold ignores
    sub-threshold jitter the same way it ignores real sub-threshold camera
    noise).
  - The leaky membrane (`V[t] = V[t-1]*k + s(t)`) makes it a short-term
    integrator: brief pauses (e.g. a stride's stance phase) do not
    immediately register as "stopped", but a sustained drop in limb/torso
    motion does -- which is exactly the "slowing down" behaviour the brief
    asks for, read from gait energy rather than only from centroid speed.
  - It is cheap: a small neuron grid per person, updated with one frame
    difference and one threshold compare.

Output per track per frame:
  motion_energy   fraction of neurons that fired this step, in [0, 1].
                  Higher = more relative motion inside the person's own
                  bounding box (swinging arms/legs, turning the body).
  motion_trend    recent average energy minus older average energy. Negative
                  and sustained means the person's motion is decaying, i.e.
                  slowing down / coming to a stop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULT_GRID = (14, 14)  # (rows, cols) neurons per person crop
TAU = 3.0  # membrane time constant, in frames (same default as video_snn.py)
V_TH = 0.55  # firing threshold
DIFF_THRESH = 10.0  # brightness delta (0-255) needed to register an event
TREND_WINDOW = 15  # frames compared "recent vs older" for the trend signal


@dataclass
class PersonRetina:
    """One LIF neuron pool for one tracked person."""

    grid: tuple[int, int] = DEFAULT_GRID
    tau: float = TAU
    v_th: float = V_TH
    diff_thresh: float = DIFF_THRESH
    trend_window: int = TREND_WINDOW
    k: float = field(init=False)
    V: np.ndarray = field(init=False)
    prev_gray: np.ndarray | None = None
    energy_hist: deque = field(default_factory=lambda: deque(maxlen=90))
    missed: int = 0

    def __post_init__(self) -> None:
        self.k = float(np.exp(-1.0 / max(self.tau, 1e-6)))
        self.V = np.zeros(self.grid, np.float32)

    def step(self, crop_bgr: np.ndarray | None) -> float:
        """Advance one frame; returns motion energy in [0, 1]."""
        if crop_bgr is None or crop_bgr.size == 0:
            self.missed += 1
            return self.energy_hist[-1] if self.energy_hist else 0.0
        self.missed = 0
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.grid[1], self.grid[0]), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        if self.prev_gray is None:
            self.prev_gray = small
            self.energy_hist.append(0.0)
            return 0.0

        diff = small - self.prev_gray
        spikes_in = (np.abs(diff) > self.diff_thresh).astype(np.float32)
        self.V = self.V * self.k + spikes_in
        fired = self.V >= self.v_th
        self.V[fired] = 0.0
        self.prev_gray = small

        energy = float(fired.mean())
        self.energy_hist.append(energy)
        return energy

    def trend(self, window: int | None = None) -> float:
        """Positive => speeding up, negative => slowing down."""
        hist = list(self.energy_hist)
        if len(hist) < 4:
            return 0.0
        if window is None:
            window = self.trend_window
        w = max(1, min(window, len(hist) // 2))
        recent = float(np.mean(hist[-w:]))
        older_slice = hist[-2 * w : -w] if len(hist) >= 2 * w else hist[: len(hist) - w]
        older = float(np.mean(older_slice)) if older_slice else recent
        return recent - older


class SNNMotionTracker:
    """Owns one PersonRetina per track id, keyed on the tracker's own ids."""

    def __init__(
        self,
        grid: tuple[int, int] = DEFAULT_GRID,
        tau: float = TAU,
        v_th: float = V_TH,
        diff_thresh: float = DIFF_THRESH,
        trend_window: int = TREND_WINDOW,
    ) -> None:
        self.grid = (max(6, int(grid[0])), max(6, int(grid[1])))
        self.tau = float(tau)
        self.v_th = float(v_th)
        self.diff_thresh = float(diff_thresh)
        self.trend_window = max(3, int(trend_window))
        self.retinas: dict[int, PersonRetina] = {}

    def update(self, track_id: int, crop_bgr: np.ndarray | None) -> dict:
        retina = self.retinas.setdefault(
            track_id,
            PersonRetina(
                grid=self.grid,
                tau=self.tau,
                v_th=self.v_th,
                diff_thresh=self.diff_thresh,
                trend_window=self.trend_window,
            ),
        )
        energy = retina.step(crop_bgr)
        return {"motion_energy": energy, "motion_trend": retina.trend()}

    def forget(self, track_id: int) -> None:
        self.retinas.pop(track_id, None)

    def prune(self, live_ids: set[int], max_missed: int = 60) -> None:
        """Drop retinas for tracks that have been gone a while (memory hygiene)."""
        stale = [
            tid
            for tid, r in self.retinas.items()
            if tid not in live_ids and r.missed > max_missed
        ]
        for tid in stale:
            self.retinas.pop(tid, None)
