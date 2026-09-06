"""Single place pipelines and GUIs read paths and thresholds from.

Scene JSON (drawn once in the GUIs) lives next to this file in `configs/`.
Numeric defaults below are what the scoring dataclasses and CLI flags use.
CLI flags still override a value for a single run.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent

# Scene / gallery JSON written by the annotation GUIs
BOUNDARY_JSON = CONFIG_DIR / "store_boundary_zones.json"
SHELF_FACES_JSON = CONFIG_DIR / "shelf_faces.json"
STAFF_MARKS_JSON = CONFIG_DIR / "staff_marks.json"

# Videos
VIDEOS_DIR = ROOT / "raw_videos"
ENTRANCE_VIDEO = VIDEOS_DIR / "entrance.mp4"
INTERIOR_VIDEO = VIDEOS_DIR / "interior.mp4"

# Annotated videos + CSVs
OUTPUT_DIR = ROOT / "outputs"
ENTRANCE_VIDEO_OUT = OUTPUT_DIR / "entrance_annotated.mp4"
INTERIOR_VIDEO_OUT = OUTPUT_DIR / "interior_annotated.mp4"
CSV_DIR = OUTPUT_DIR / "csv"
INTEREST_CSV_DIR = CSV_DIR / "interest"
STAFF_CSV_DIR = CSV_DIR / "staff_interaction"
SHELF_CSV_DIR = CSV_DIR / "shelf_vector_interest"
INTEREST_VIDEO_OUT = OUTPUT_DIR / "interest_annotated.mp4"
STAFF_VIDEO_OUT = OUTPUT_DIR / "staff_interaction_annotated.mp4"

# Weights: local file if present, otherwise downloadable names/URLs
MODELS_DIR = ROOT / "models"
POSE_WEIGHTS_NAME = "yolo11x-pose.pt"
REID_ONNX_PATH = MODELS_DIR / "resnet50_market1501_aicity156.onnx"

# ---------------------------------------------------------------------------
# Shared run / pose / ReID
# ---------------------------------------------------------------------------
FRAME_STRIDE = 2
POSE_IMGSZ = 1280  # native 1280; 640 halves walkway people (~60 px) and misses them
POSE_DET_CONF = 0.25
POSE_KP_CONF = 0.30  # below this a keypoint is treated as missing, not wrong
POSE_MIN_H = 32
POSE_MIN_W = 12
POSE_KPT_GATE_H = 100
POSE_MIN_KPTS = 4

TRACK_MATCH_IOU = 0.25
REID_THRESHOLD = 0.52
REID_RELINK = 0.58
REID_WEIGHT = 0.65
REID_EMB_EMA = 0.85
REID_RELINK_MAX_DIST_BH = 6.0

# ---------------------------------------------------------------------------
# Task 1 — store interest (entrance walkway)
# Distances/speeds are in body-heights (bbox height), not pixels.
# ---------------------------------------------------------------------------
INTEREST_SPEED_WINDOW_S = 0.45
INTEREST_TURN_WINDOW_S = 0.45
INTEREST_ATTEND_DEG = 68.0  # attention cone that counts as looking at the shop
INTEREST_TURN_DEG = 9.0  # angle swing toward the shop that saturates the turn cue
INTEREST_WALK_BH = 1.55  # unremarkable walking pace; slowdown cue = 0 at or above this
INTEREST_SLOW_BH = 0.55  # at/below this the speed half of slowdown saturates
INTEREST_MOTION_TREND_REF = 0.035  # SNN energy drop that saturates the trend half
INTEREST_APPROACH_REF_BH = 0.25  # closing speed that saturates the approach cue
INTEREST_W_ORIENT = 0.32
INTEREST_W_TURN = 0.18
INTEREST_W_SLOW = 0.20
INTEREST_W_APPROACH = 0.30
INTEREST_SCORE_THRESHOLD = 0.47
INTEREST_SUSTAIN_S = 0.45  # EMA score must hold above threshold this long
INTEREST_EMA = 0.45
INTEREST_ENTERED_DWELL_S = 0.25  # continuous time inside the shop polygon
INTEREST_ENTERED_BACKFILL_SCORE = 0.38
INTEREST_MIN_HITS = 4
INTEREST_MAX_MISSED = 45  # keep id while they cross the doorway
INTEREST_TRACK_MIN_HITS = 3

# Per-person LIF motion sensor (SNN)
SNN_GRID = (14, 14)
SNN_DIFF_THRESH = 10.0  # brightness delta (0–255) to register an event
SNN_VTH = 0.55  # firing threshold
SNN_TAU = 3.0  # membrane time constant, frames
SNN_TREND_WINDOW = 15

# ---------------------------------------------------------------------------
# Task 2 — per-shelf interest (interior, explicit face geometry)
# ---------------------------------------------------------------------------
SHELF_FACE_DEG = 55.0  # max angle between person facing vec and -normal
SHELF_ENGAGE_S = 1.2  # sustained engagement to open an event
SHELF_GAP_CLOSE_S = 1.0  # tolerated glance-away before closing
SHELF_COOLDOWN_S = 5.0  # per (person, shelf) cooldown after a close
SHELF_MIN_EVENT_S = 0.8  # closed event shorter than this is jitter, not counted
SHELF_MIN_HITS = 4
SHELF_MAX_MISSED = 90
SHELF_RETIRED_TTL_FRAMES = 450  # keep id while walking between shelves

# ---------------------------------------------------------------------------
# Task 3 — staff–customer interaction (entrance)
# ---------------------------------------------------------------------------
STAFF_ROLE_SIM_THRESHOLD = 0.62  # cosine vs enrolled gallery to count as a staff vote
STAFF_ROLE_MIN_VOTES = 3
STAFF_ROLE_MIN_RATIO = 0.55
STAFF_ROLE_MIN_CHECKS = 5

STAFF_NEAR_BH = 1.45  # conversational distance; beyond this, not engaged
STAFF_VERY_NEAR_BH = 0.75  # this close, skip the facing requirement
STAFF_FACE_DEG = 75.0  # facing-cone half-angle toward the other person
STAFF_W_PROX = 0.55
STAFF_W_FACE = 0.45
STAFF_SCORE_THRESHOLD = 0.50
STAFF_OPEN_S = 1.00  # sustained engagement to open a session
STAFF_GAP_CLOSE_S = 1.50
STAFF_COOLDOWN_S = 3.00  # later return to the same staff = a new session
STAFF_MIN_EVENT_S = 1.20
STAFF_MIN_HITS = 4
STAFF_MAX_MISSED = 45
STAFF_RETIRED_TTL_FRAMES = 90  # brief occlusion only; re-entry may be a new instance
