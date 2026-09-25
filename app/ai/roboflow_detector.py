"""
RoboflowDetector  —  uses the Roboflow Inference API (100% free, no GPU needed).

How it works:
  1. You go to universe.roboflow.com, fork the public firstaid_4class-2 dataset,
     and click "Train" — Roboflow trains on their servers for free.
  2. After training, your model gets a Model ID like  "firstaid_4class-2/2"
  3. This detector calls that hosted model via HTTP — no best.pt download needed.

Required env vars (.env):
  FIRST_AID_MODEL_MODE=roboflow
  ROBOFLOW_API_KEY=your_key_here          # from app.roboflow.com → Settings → API Keys
  ROBOFLOW_MODEL_ID=firstaid_4class-2/2   # workspace/project/version

Optional:
  FIRST_AID_CONFIDENCE_THRESHOLD=0.50     # passed to the API as confidence
"""
from __future__ import annotations

import base64
import logging
import os

import cv2
import numpy as np

from .types import Detection
from .detector import BaseDetector, DetectorError, VALID_CLASSES

logger = logging.getLogger(__name__)

# Map Roboflow dataset class names → your system's VALID_CLASSES
_LABEL_MAP: dict[str, str] = {
    "cut_open_wound": "cut",
    "cut":            "cut",
    "burn_mild":      "burn",
    "burn_severe":    "burn",
    "burn":           "burn",
    "bruise":         "bruise",
    "abrasion":       "abrasion",
    "swelling":       "swelling",
    "laceration":     "cut",
    "wound":          "cut",
    "injury":         "other",
}


def _encode_image(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise DetectorError("Failed to encode image to JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class RoboflowDetector(BaseDetector):
    """
    Calls the Roboflow hosted inference endpoint.
    Free tier: unlimited calls, ~0.3–1 s per image over the network.
    No GPU, no best.pt, no ultralytics install needed.
    """

    name = "roboflow"

    def __init__(self, api_key: str, model_id: str, confidence: float = 0.45) -> None:
        if not api_key:
            raise DetectorError(
                "ROBOFLOW_API_KEY is not set. "
                "Get it from app.roboflow.com → Settings → API Keys."
            )
        if not model_id:
            raise DetectorError(
                "ROBOFLOW_MODEL_ID is not set. "
                "Set it to  workspace-slug/project-slug/version  e.g. firstaid_4class-2/2"
            )
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise DetectorError(
                "inference-sdk is not installed. Run:  pip install inference-sdk"
            ) from exc

        self._client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        self._model_id  = model_id
        self._confidence = confidence
        logger.info("RoboflowDetector ready — model: %s", model_id)

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]

        try:
            result = self._client.infer(image, model_id=self._model_id)
        except Exception as exc:
            raise DetectorError(f"Roboflow API call failed: {exc}") from exc

        predictions = result.get("predictions", []) if isinstance(result, dict) else []
        detections: list[Detection] = []

        for pred in predictions:
            try:
                conf = float(pred.get("confidence", 0.0))
                if conf < self._confidence:
                    continue

                raw_label = str(pred.get("class", "unknown")).strip().lower()
                label = _LABEL_MAP.get(raw_label, raw_label)
                if label not in VALID_CLASSES:
                    label = "other"

                # Roboflow returns centre + width/height
                cx = float(pred["x"])
                cy = float(pred["y"])
                bw = float(pred["width"])
                bh = float(pred["height"])

                x1 = max(0, int(cx - bw / 2))
                y1 = max(0, int(cy - bh / 2))
                x2 = min(w, int(cx + bw / 2))
                y2 = min(h, int(cy + bh / 2))

                detections.append(Detection(label, conf, (x1, y1, x2, y2)))
            except (KeyError, TypeError, ValueError):
                continue

        return detections
