"""Staff-marking config: a small ReID embedding gallery, enrolled once by
clicking each staff member's body in `staff_gui.py`.

Saved to `configs/staff_marks.json`, keyed by video filename, e.g.:

    {
      "entrance.mp4": {
        "gallery": [[...], [...]]
      }
    }

Why ReID-gallery matching instead of apron colour or a VLM (see
`pipelines/staff_interaction/README.md` for the full history)
------------------------------------------------------------------------
Two earlier approaches were tried and both failed on this footage:

  1. Apron-colour (HSV) thresholding -- aprons, rugs, wood shelving and
     shadows can land in the same hue/saturation/value neighbourhood, so a
     window loose enough to survive lighting was also loose enough to
     match half the store.
  2. A VLM (Qwen2-VL-2B) asked "is this an apron?" per crop -- prompt
     engineering could push false positives down or false negatives down,
     but not both at once: tightening the prompt to stop matching
     customers' bags/jackets also started rejecting real staff whose
     uniform did not look exactly like the described apron, and vice
     versa. It is also the slowest part of the pipeline by a wide margin.

A ReID gallery sidesteps both failure modes. It does not classify by
colour (immune to the apron-vs-rug problem) or by asking a general-purpose
model to recognise "an apron" (immune to prompt-wording sensitivity) --
it simply asks "does this track's appearance embedding look like one of
the specific people I was shown are staff?", which is a much narrower and
better-conditioned question. It is also essentially free at inference
time: the tracker already computes an appearance embedding for every
track every frame (for ReID-assisted tracking itself), so role
classification is one cosine similarity against a handful of gallery
vectors, not a model call.

The cost is a one-time manual step (`staff_gui.py`): clicking each visible
staff member once (ideally a couple of times, from different frames/poses,
for a more robust gallery). This is a deliberate trade -- a few seconds of
setup for a deterministic, fast, tunable classifier -- and is explicitly
allowed by the brief ("the solution may use visual appearance, behavior,
spatial context, or any other reasonable approach").
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs.paths import STAFF_MARKS_JSON as CONFIG_PATH  # noqa: E402


@dataclass
class StaffMarks:
    gallery: list = field(default_factory=list)  # list of float embedding vectors (unit-normalised)

    def has_gallery(self) -> bool:
        return len(self.gallery) > 0

    def ready(self) -> tuple[bool, str]:
        if not self.has_gallery():
            return False, "no staff enrolled"
        return True, "ok"

    def gallery_matrix(self) -> np.ndarray | None:
        if not self.gallery:
            return None
        return np.stack([np.asarray(e, np.float32) for e in self.gallery], axis=0)

    def to_dict(self) -> dict:
        return {"gallery": [list(map(float, e)) for e in self.gallery]}

    @classmethod
    def from_dict(cls, d: dict) -> "StaffMarks":
        return cls(gallery=[list(map(float, e)) for e in d.get("gallery", [])])


def load_staff_marks(video_name: str) -> StaffMarks:
    if not CONFIG_PATH.exists():
        return StaffMarks()
    data = json.loads(CONFIG_PATH.read_text())
    if video_name not in data:
        return StaffMarks()
    return StaffMarks.from_dict(data[video_name])


def save_staff_marks(video_name: str, marks: StaffMarks) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    data[video_name] = marks.to_dict()
    CONFIG_PATH.write_text(json.dumps(data, indent=2))
