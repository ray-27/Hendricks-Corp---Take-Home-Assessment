"""Qwen2-VL-2B-Instruct wrapper: the two semantic judgments this pipeline
needs a real vision-language model for, everything else stays rule-based.

  ask_role(crop)        -- "is this person wearing a staff apron?"
                           single-person *torso* crop (see `scoring.py`'s
                           `_torso_crop`), queried a handful of times per
                           track (sticky majority vote), not every frame.
                           The prompt explicitly describes what an apron is
                           and is not (a bag, jacket, or scarf is not an
                           apron) and is biased to answer NO when unsure,
                           since a false positive here is sticky and cannot
                           self-correct -- see `RoleParams` in `scoring.py`.
  ask_interaction(crop) -- "are these two people actively interacting?"
                           pair crop, queried only when the pair is already
                           within conversational distance, throttled by a
                           per-pair cooldown (see `scoring.py`) -- a full
                           generative VLM call is ~0.5-2s on Apple Silicon
                           MPS, so it is never run per-frame.

Both methods return `True` / `False` / `None` -- `None` means the model's
answer could not be parsed into yes/no (should be retried, not treated as
either answer) rather than silently defaulting to one side.

Kept as its own module so the rest of this pipeline (tracker, session state
machine, CLI) does not need to know anything about prompt text or model
internals -- swapping in a different VLM later only touches this file.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image

ROLE_PROMPT_BASE = (
    "You are looking at a single cropped photo, focused on one person's torso, "
    "inside a retail store. Staff members wear a distinct WORK GARMENT layer "
    "over their regular clothes -- this could be a full bib apron, a "
    "waist/half apron, a tabard, or a fitted vest/top -- usually a single "
    "solid colour that is different from ordinary streetwear, and very often "
    "printed with a small store logo or name on the chest. It does not need "
    "to be tied at the waist to count; a snug branded top or vest counts just "
    "as much as a traditional tied apron.\n\n"
    "Customers do NOT wear this. A customer may instead be carrying a "
    "handbag, tote bag, or shopping bag in front of their torso, or wearing "
    "an ordinary jacket, cardigan, or scarf with no logo -- these are not "
    "staff garments, and seeing only one of these (with no branded garment "
    "underneath or around it) should be answered NO.{extra}\n\n"
    "Look carefully at the colour and texture of what is actually covering "
    "the torso, and at whether there is a logo/text on it, before deciding. "
    "If you can see a distinct, differently-coloured or logo'd work garment "
    "on the torso, answer YES even if the framing is imperfect (bent over, "
    "partially turned, or a bit blurry). Only answer NO if the torso clearly "
    "shows ordinary street clothing, or is covered only by a bag/jacket with "
    "no sign of a branded garment underneath. Answer with exactly one word: "
    "YES or NO."
)

INTERACTION_PROMPT = (
    "You are looking at a cropped photo containing two people inside a "
    "retail store; one may be a staff member and one may be a customer. "
    "Are they actively interacting with each other right now -- for "
    "example talking face to face, one helping or handing something to "
    "the other, or looking at the same item together? People who are "
    "merely standing near each other, walking past, waiting in a queue, "
    "or not paying attention to each other do NOT count as interacting. "
    "Answer with exactly one word: YES or NO."
)

_YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def _best_device() -> str:
    """CUDA -> MPS (Apple Silicon) -> CPU, same convention as the rest of
    this repo (`analytics.pose.best_device`, `reid.embedder.pick_runtime`)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _parse_yes_no(text: str) -> bool | None:
    m = _YES_NO_RE.search(text or "")
    if m is None:
        return None
    return m.group(1).upper() == "YES"


def _to_pil(crop_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


class VLMJudge:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 8,
        max_side: int = 448,
        role_hint: str = "",
    ) -> None:
        """`max_side` downsizes crops before they hit the model -- Qwen2-VL's
        cost scales with image tokens, and a yes/no apron/interaction call
        does not need full resolution.

        `role_hint`: optional free-text description of what staff aprons
        actually look like *in this specific video* (e.g. "a dark maroon
        fabric apron tied at the waist"). The base prompt already describes
        aprons generically and explicitly rules out bags/jackets, but a
        concrete description of the real uniform, when you know it, helps
        the model ground the answer in this footage instead of a generic
        prior -- pass it via `--staff-description` on the CLI. Optional;
        the pipeline runs with no setup step either way.
        """
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = device or _best_device()
        self.max_new_tokens = max_new_tokens
        self.max_side = max_side
        extra = f" In this specific store, staff aprons look like: {role_hint.strip()}." if role_hint.strip() else ""
        self.role_prompt = ROLE_PROMPT_BASE.format(extra=extra)

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        print(f"Loading {model_id} on {self.device} ({dtype})…")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("VLMJudge ready.")

    def _resize(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.max_side / max(w, h)
        if scale >= 1.0:
            return img
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    def _ask(self, crop_bgr: np.ndarray, prompt: str) -> str:
        import torch

        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        img = self._resize(_to_pil(crop_bgr))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    def ask_role(self, crop_bgr: np.ndarray) -> bool | None:
        return _parse_yes_no(self._ask(crop_bgr, self.role_prompt))

    def ask_interaction(self, crop_bgr: np.ndarray) -> bool | None:
        return _parse_yes_no(self._ask(crop_bgr, INTERACTION_PROMPT))
