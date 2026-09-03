"""
Spiking retina on a recorded video file.

Same pipeline as webcam_snn.py -- simulated DVS events into a per-pixel LIF
layer -- but reads frames from a video on disk instead of a live camera.
Shows the original frame and SNN panels side by side.

Pipeline, one neuron per pixel of a downsampled grid:

    video frame
        -> grayscale, downsample to a small grid
        -> temporal difference between consecutive frames
        -> threshold into ON / OFF spikes (event-camera simulation)
        -> LIF neuron per pixel
        -> population decode of spike centroid

Run (put your video in this folder, or pass an explicit path):
    python video_snn.py
    python video_snn.py --video my_clip.mp4
    python video_snn.py --max-side 720          # sharper SNN (less blocky)
    python video_snn.py --grid 640 360 --scale 2

Press 'q' to quit, 'space' to pause/resume, 'r' to restart from the beginning.

Install deps once:  pip3 install opencv-python numpy
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from collections import deque

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


# --------------------------------------------------------------------- SNN

class SpikingRetina:
    """One LIF neuron per grid cell, separate ON and OFF populations.

    V[t] = V[t-1] * k + s(t)          (leak then integrate)
    fire, V <- reset                   if V[t] >= v_th
    """

    def __init__(self, shape: tuple[int, int], tau: float, v_th: float, v_reset: float = 0.0):
        self.shape = shape
        self.tau = tau
        self.k = float(np.exp(-1.0 / tau))
        self.v_th = v_th
        self.v_reset = v_reset
        self.V_on = np.zeros(shape, dtype=np.float32)
        self.V_off = np.zeros(shape, dtype=np.float32)

    def step(self, s_on: np.ndarray, s_off: np.ndarray):
        """Advance one frame. s_on/s_off are binary spike inputs (0/1)."""
        self.V_on = self.V_on * self.k + s_on
        self.V_off = self.V_off * self.k + s_off

        fired_on = self.V_on >= self.v_th
        fired_off = self.V_off >= self.v_th
        self.V_on[fired_on] = self.v_reset
        self.V_off[fired_off] = self.v_reset
        return fired_on, fired_off

    def reset(self) -> None:
        self.V_on.fill(0.0)
        self.V_off.fill(0.0)


def events_from_frames(prev_gray: np.ndarray, gray: np.ndarray, pos_th: float, neg_th: float):
    """Simulate a DVS event camera: threshold the temporal brightness change."""
    diff = gray.astype(np.float32) - prev_gray.astype(np.float32)
    s_on = (diff > pos_th).astype(np.float32)
    s_off = (diff < -neg_th).astype(np.float32)
    return s_on, s_off, diff


def population_decode(fired_on: np.ndarray, fired_off: np.ndarray):
    """Center-of-mass over ALL spikes this frame (one point for the whole scene)."""
    activity = fired_on.astype(np.float32) + fired_off.astype(np.float32)
    total = activity.sum()
    if total < 1:
        return None
    ys, xs = np.nonzero(activity)
    w = activity[ys, xs]
    cy = float((ys * w).sum() / total)
    cx = float((xs * w).sum() / total)
    return cx, cy


def person_regions(fired_on: np.ndarray, fired_off: np.ndarray,
                   close_k: int = 11, min_area: int = 50):
    """Group per-pixel spikes into whole-person boxes.

    The retina stays one neuron per pixel on purpose (that is the DVS/LIF
    layer). A moving body therefore lights up as a *cloud* of ON/OFF spikes.
    This readout closes the gaps inside one body, then takes connected
    components so each person is one region — not thousands of patches.
    """
    activity = ((fired_on | fired_off).astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    merged = cv2.morphologyEx(activity, cv2.MORPH_CLOSE, kernel)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(merged, 8)
    regions = []
    for i in range(1, n):  # 0 is background
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        regions.append({
            "box": (int(x), int(y), int(x + bw), int(y + bh)),
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1]),
            "area": int(area),
        })
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions, merged


# --------------------------------------------------------------- rendering

def colorize_events(s_on: np.ndarray, s_off: np.ndarray, scale: int) -> np.ndarray:
    """ON = green, OFF = magenta, nothing = black."""
    h, w = s_on.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 1] = (s_on * 255).astype(np.uint8)
    img[..., 2] = (s_off * 255).astype(np.uint8)
    img[..., 0] = (s_off * 255).astype(np.uint8)
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def colorize_membrane(V_on: np.ndarray, v_th: float, scale: int) -> np.ndarray:
    """Grayscale heatmap of the analog membrane potential, before threshold."""
    norm = np.clip(V_on / max(v_th, 1e-6), 0, 1)
    img = (norm * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    h, w = V_on.shape
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def draw_strip_chart(buf: np.ndarray, history: deque, height: int) -> np.ndarray:
    """Scroll a 1px-per-frame line chart of total spike count into `buf`."""
    buf[:] = 20
    w = buf.shape[1]
    vals = list(history)[-w:]
    if len(vals) < 2:
        return buf
    peak = max(max(vals), 1)
    xs = np.linspace(w - len(vals), w - 1, len(vals)).astype(int)
    ys = (height - 6 - (np.array(vals) / peak) * (height - 16)).astype(int)
    for i in range(1, len(xs)):
        cv2.line(buf, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (60, 170, 255), 1)
    cv2.putText(buf, f"spike count / frame  (peak {peak:.0f})", (6, 14), FONT, 0.4,
                (180, 180, 180), 1, cv2.LINE_AA)
    return buf


def put_lines(img: np.ndarray, lines: list[str], x: int, y0: int, dy: int = 18,
              color=(230, 230, 230), scale: float = 0.45) -> None:
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y0 + i * dy), FONT, scale, color, 1, cv2.LINE_AA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 15), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def find_video_in_folder(folder: str) -> str | None:
    """Pick the newest video file in `folder`, if any."""
    candidates: list[str] = []
    for ext in VIDEO_EXTS:
        candidates.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        candidates.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    # de-dupe (case-insensitive FS) and prefer newest mtime
    unique = sorted(set(os.path.abspath(p) for p in candidates),
                    key=lambda p: os.path.getmtime(p), reverse=True)
    return unique[0] if unique else None


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=str, default=None,
                    help="path to video file (default: newest video in this script's folder)")
    ap.add_argument("--grid", type=int, nargs=2, default=None,
                    metavar=("W", "H"),
                    help="retina grid size (neuron count = W*H). "
                         "Default: auto from video, capped by --max-side")
    ap.add_argument("--max-side", type=int, default=480,
                    help="when --grid is omitted, longest side of the retina grid "
                         "(higher = sharper / less blocky; try 640 or 720)")
    ap.add_argument("--scale", type=int, default=None,
                    help="display upscale factor per neuron "
                         "(default: auto so each panel is ~640px wide)")
    ap.add_argument("--tau", type=float, default=3.0, help="LIF membrane time constant (frames)")
    ap.add_argument("--v-th", type=float, default=1.0, help="firing threshold")
    ap.add_argument("--pos-thresh", type=float, default=12.0,
                    help="brightness increase (0-255) needed to register an ON event")
    ap.add_argument("--neg-thresh", type=float, default=12.0,
                    help="brightness decrease (0-255) needed to register an OFF event")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (1.0 = native fps)")
    ap.add_argument("--loop", action="store_true",
                    help="restart the video when it ends")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    video_path = args.video
    if video_path is None:
        video_path = find_video_in_folder(here)
        if video_path is None:
            raise SystemExit(
                f"No video found in {here}. Drop a .mp4/.mov/.avi here, or pass "
                f"--video /path/to/file.mp4"
            )
    else:
        video_path = os.path.abspath(video_path)

    if not os.path.isfile(video_path):
        raise SystemExit(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    if args.grid is None:
        # Preserve aspect ratio; cap longest side so full-HD video stays realtime.
        long_side = max(src_w, src_h)
        if long_side > args.max_side:
            scale_down = args.max_side / long_side
            W = max(8, int(round(src_w * scale_down)))
            H = max(8, int(round(src_h * scale_down)))
        else:
            W, H = src_w, src_h
    else:
        W, H = args.grid

    if args.scale is None:
        # Keep each panel around ~640px wide so the collage still fits on screen.
        args.scale = max(1, int(round(640 / max(W, 1))))

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        src_fps = 30.0
    frame_delay_ms = max(1, int(round(1000.0 / (src_fps * max(args.speed, 1e-3)))))

    print(f"Retina grid: {W}x{H} neurons  (display scale x{args.scale})")
    retina = SpikingRetina((H, W), tau=args.tau, v_th=args.v_th)
    prev_gray = None
    spike_history: deque = deque(maxlen=400)
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0
    paused = False
    frame_idx = 0

    strip_h = 90
    print(f"Video: {video_path}")
    print(f"Source fps: {src_fps:.2f}  |  playback delay: {frame_delay_ms} ms/frame")
    print("Keys:  q = quit   space = pause/resume   r = restart")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    retina.reset()
                    prev_gray = None
                    spike_history.clear()
                    frame_idx = 0
                    continue
                print("End of video.")
                break

            frame_idx += 1
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray_full, (W, H), interpolation=cv2.INTER_AREA)

            if prev_gray is None:
                prev_gray = gray
                # still draw a blank first frame so the window appears immediately
                disp_w, disp_h = W * args.scale, H * args.scale
                cam_panel = label(cv2.resize(frame, (disp_w, disp_h)),
                                  "1. video  (this is what your retina layer sees)")
                blank = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                event_panel = label(blank.copy(), "2. spikes fired this frame  (green=ON, magenta=OFF)")
                mem_panel = label(blank.copy(), "3. membrane potential V_on  (before threshold)")
                top_row = np.hstack([cam_panel, event_panel, mem_panel])
                strip = np.zeros((strip_h, top_row.shape[1], 3), dtype=np.uint8)
                math_panel = np.zeros((130, top_row.shape[1], 3), dtype=np.uint8)
                put_lines(math_panel, ["waiting for second frame to compute temporal difference..."],
                          x=8, y0=22, dy=22, scale=0.42)
                collage = np.vstack([top_row, strip, math_panel])
                cv2.imshow("Spiking retina (video) -- q quit | space pause | r restart", collage)
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    paused = True
                if key == ord("r"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    retina.reset()
                    prev_gray = None
                    spike_history.clear()
                    frame_idx = 0
                continue

            s_on, s_off, diff = events_from_frames(prev_gray, gray, args.pos_thresh, args.neg_thresh)
            fired_on, fired_off = retina.step(s_on, s_off)
            centroid = population_decode(fired_on, fired_off)
            regions, _merged = person_regions(fired_on, fired_off)
            prev_gray = gray

            n_on = int(fired_on.sum())
            n_off = int(fired_off.sum())
            spike_history.append(n_on + n_off)

            # ---- panel 1: original video with one box per person-region
            disp_w, disp_h = W * args.scale, H * args.scale
            cam_panel = cv2.resize(frame, (disp_w, disp_h))
            if centroid is not None:
                cx, cy = centroid
                px, py = int(cx * args.scale), int(cy * args.scale)
                cv2.circle(cam_panel, (px, py), 8, (60, 170, 255), 1)
                cv2.putText(cam_panel, "global decode (all spikes)", (px + 12, py),
                            FONT, 0.35, (60, 170, 255), 1, cv2.LINE_AA)
            for i, r in enumerate(regions):
                x1, y1, x2, y2 = r["box"]
                p1 = (x1 * args.scale, y1 * args.scale)
                p2 = (x2 * args.scale, y2 * args.scale)
                cv2.rectangle(cam_panel, p1, p2, (0, 220, 255), 2)
                cv2.putText(cam_panel, f"person {i+1}", (p1[0], max(32, p1[1] - 6)),
                            FONT, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cam_panel = label(cam_panel, f"1. video  |  {len(regions)} person region(s)")

            # ---- panel 2: spike image + the same whole-person boxes
            event_panel = colorize_events(fired_on, fired_off, args.scale)
            for i, r in enumerate(regions):
                x1, y1, x2, y2 = r["box"]
                p1 = (x1 * args.scale, y1 * args.scale)
                p2 = (x2 * args.scale, y2 * args.scale)
                cv2.rectangle(event_panel, p1, p2, (255, 255, 255), 1)
            event_panel = label(event_panel, "2. spikes  (green=ON, magenta=OFF)  white box = whole person")

            # ---- panel 3: sub-threshold membrane potential
            mem_panel = colorize_membrane(retina.V_on, args.v_th, args.scale)
            mem_panel = label(mem_panel, "3. membrane potential V_on  (before threshold)")

            top_row = np.hstack([cam_panel, event_panel, mem_panel])

            # ---- strip chart
            strip = np.zeros((strip_h, top_row.shape[1], 3), dtype=np.uint8)
            draw_strip_chart(strip, spike_history, strip_h)

            # ---- live math readout
            math_h = 130
            math_panel = np.zeros((math_h, top_row.shape[1], 3), dtype=np.uint8)
            fps_n += 1
            if time.time() - fps_t0 > 0.5:
                fps = fps_n / (time.time() - fps_t0)
                fps_n = 0
                fps_t0 = time.time()
            centroid_txt = f"({centroid[0]:.1f}, {centroid[1]:.1f})" if centroid else "no motion"
            put_lines(math_panel, [
                f"encode:  s_on = 1 if (I_t - I_t-1) > {args.pos_thresh:g}   "
                f"s_off = 1 if (I_t - I_t-1) < -{args.neg_thresh:g}",
                f"LIF:     V[t] = V[t-1] * k + s(t)     k = exp(-1/tau) = {retina.k:.3f}   "
                f"(tau={args.tau:g} frames)   [one neuron per pixel — the cloud is correct]",
                f"readout: close spike gaps inside a body, then connected components  "
                f"->  {len(regions)} person region(s)   (not one patch per spike)",
                f"old decode (all spikes, one point): {centroid_txt}   |   "
                f"new decode: one box per region",
                f"frame {frame_idx:5d}  |  ON {n_on:5d}   OFF {n_off:5d}   "
                f"of {W*H} neurons   |   {fps:4.1f} fps   |   grid {W}x{H}   |   "
                f"src {src_fps:.1f} fps x{args.speed:g}",
            ], x=8, y0=22, dy=22, scale=0.42)

            collage = np.vstack([top_row, strip, math_panel])
            cv2.imshow("Spiking retina (video) -- q quit | space pause | r restart", collage)

        key = cv2.waitKey(frame_delay_ms if not paused else 30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            retina.reset()
            prev_gray = None
            spike_history.clear()
            frame_idx = 0
            paused = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
