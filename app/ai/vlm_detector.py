"""
ClaudeVLMDetector  –  zero-training injury detection via Claude vision API.

Drop-in replacement for YOLODetector.  Requires:
  pip install anthropic

Environment variables (same .env file as the rest of the system):
  FIRST_AID_MODEL_MODE=claude          # activate this detector
  ANTHROPIC_API_KEY=sk-ant-...         # your Anthropic key
  FIRST_AID_CONFIDENCE_THRESHOLD=0.70  # optional, same as before
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re

import cv2
import numpy as np

from .types import Detection
from .detector import BaseDetector, DetectorError, VALID_CLASSES

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt sent to Claude for every image
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a first-aid image analysis assistant.
Your job is to identify visible injuries in photographs.
You must return ONLY a valid JSON array — no explanation, no markdown, no backticks.

Each element in the array must be an object with exactly these keys:
  "label"      : one of: cut, abrasion, bruise, burn, swelling, other, unknown
  "confidence" : a float 0.0–1.0 representing how certain you are
  "bbox"       : [x1, y1, x2, y2] pixel coordinates of the injury region
                 where (0,0) is the top-left corner of the image

Rules:
- If NO injury is visible, return an empty array: []
- If the image is blurry, too dark, or not a medical image, return []
- Never guess.  Only report injuries you can actually see.
- bbox must fit within the image dimensions provided.
- confidence should reflect real visual evidence.
"""

_USER_TEMPLATE = (
    "Image dimensions: {width}px wide × {height}px tall.\n"
    "Identify all visible injuries and return the JSON array."
)


def _encode_image(image: np.ndarray) -> str:
    """Encode an OpenCV BGR image as base64 JPEG."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise DetectorError("Failed to encode image to JPEG for VLM.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _parse_response(text: str, img_w: int, img_h: int) -> list[Detection]:
    """Parse Claude's JSON response into Detection objects."""
    # Strip any accidental markdown fences
    text = re.sub(r"```[a-z]*", "", text).strip().strip("`").strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("VLM response is not valid JSON: %s | raw=%r", exc, text[:300])
        return []

    if not isinstance(raw, list):
        logger.warning("VLM response is not a JSON list: %r", text[:300])
        return []

    detections: list[Detection] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            label = str(item.get("label", "unknown")).strip().lower()
            if label not in VALID_CLASSES:
                label = "other"

            conf = float(item.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))

            raw_bbox = item.get("bbox", [])
            if len(raw_bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
            x1 = _clamp(x1, 0, img_w - 1)
            y1 = _clamp(y1, 0, img_h - 1)
            x2 = _clamp(x2, x1 + 1, img_w)
            y2 = _clamp(y2, y1 + 1, img_h)

            detections.append(Detection(label, conf, (x1, y1, x2, y2)))
        except (KeyError, TypeError, ValueError):
            continue

    return detections


# ──────────────────────────────────────────────────────────────────────────────
# Main detector class
# ──────────────────────────────────────────────────────────────────────────────
class ClaudeVLMDetector(BaseDetector):
    """
    Sends the image to Claude claude-sonnet-4-6 vision and parses structured
    injury detections from the JSON response.

    No model training required.  Accuracy is driven by Claude's built-in
    medical vision knowledge.
    """

    name = "claude_vlm"

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise DetectorError(
                "ANTHROPIC_API_KEY is not set.  "
                "Add it to your .env file to use the Claude VLM detector."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise DetectorError(
                "anthropic package is not installed.  Run: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info("ClaudeVLMDetector initialised (model: claude-sonnet-4-6)")

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]

        try:
            b64 = _encode_image(image)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"Image encoding failed: {exc}") from exc

        try:
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": _USER_TEMPLATE.format(width=w, height=h),
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:
            raise DetectorError(f"Claude API call failed: {exc}") from exc

        raw_text = "".join(
            block.text for block in msg.content if hasattr(block, "text")
        )
        logger.debug("VLM raw response: %r", raw_text[:500])

        return _parse_response(raw_text, w, h)
