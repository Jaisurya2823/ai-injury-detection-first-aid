"""
RoboflowWorkflowDetector
========================
Calls the hosted Roboflow Workflow via inference-sdk 1.7+.

  Workspace : jaisurya-i4vpr
  Workflow  : firstaid4class-2-vfirstaid4class-2-v45sb-1-yolo26m-t1-logic
  Endpoint  : https://serverless.roboflow.com

inference-sdk 1.7 returns TYPED OBJECTS, not plain dicts.
The response looks like:

  [  # list, one entry per image
    {
      "predictions": <ObjectDetectionInferenceResponse object>
        .predictions = [
          <ObjectDetectionPrediction>
            .x, .y, .width, .height, .confidence, .class_name (or .class_)
        ]
    }
  ]

_unwrap_output() handles all known shapes:
  - typed SDK objects  (inference-sdk 1.7)
  - plain dicts        (older SDK or direct REST call)
  - nested dicts       (workflow wraps output in named key)
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import cv2
import numpy as np

from .types import Detection
from .detector import BaseDetector, DetectorError, VALID_CLASSES

logger = logging.getLogger(__name__)

_WORKSPACE   = "jaisurya-i4vpr"
_WORKFLOW_ID = "firstaid4class-2-vfirstaid4class-2-v45sb-1-yolo26m-t1-logic"
_API_URL     = "https://serverless.roboflow.com"
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0

_LABEL_MAP: dict[str, str] = {
    "cut_open_wound": "cut",
    "cut":            "cut",
    "laceration":     "cut",
    "wound":          "cut",
    "burn_mild":      "burn",
    "burn_severe":    "burn",
    "burn":           "burn",
    "bruise":         "bruise",
    "abrasion":       "abrasion",
    "swelling":       "swelling",
    "injury":         "other",
}


# ── Utility: safely read an attribute or dict key ─────────────────────────────

def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Read attr or dict key from obj, trying each key in order."""
    for k in keys:
        try:
            if isinstance(obj, dict):
                if k in obj:
                    return obj[k]
            else:
                v = getattr(obj, k, _SENTINEL)
                if v is not _SENTINEL:
                    return v
        except Exception:
            pass
    return default

_SENTINEL = object()


def _looks_like_pred(obj: Any) -> bool:
    """True if obj looks like a single YOLO prediction (has confidence + coords)."""
    return (
        _get(obj, "confidence") is not None
        and (_get(obj, "x") is not None or _get(obj, "bbox") is not None)
    )


def _to_pred_list(val: Any, depth: int = 0) -> list[Any] | None:
    """
    Recursively extract the prediction list from whatever the SDK returned.
    Handles:
      - A list of prediction objects / dicts                → return it directly
      - An ObjectDetectionInferenceResponse with .predictions
      - A dict with "predictions" key (plain or nested)
    """
    if depth > 4:
        return None

    # Already a list of predictions
    if isinstance(val, list):
        if val and _looks_like_pred(val[0]):
            return val
        if not val:
            return val   # empty list is valid (no detections)
        return None

    # SDK typed object or dict — look for .predictions / ["predictions"]
    inner = _get(val, "predictions")
    if inner is not None:
        result = _to_pred_list(inner, depth + 1)
        if result is not None:
            return result

    # Try every attribute / key for a nested structure
    items = val.items() if isinstance(val, dict) else (
        ((k, getattr(val, k)) for k in getattr(val, "__dict__", {}).keys())
    )
    for _k, v in items:
        if _k == "predictions":
            continue  # already tried above
        result = _to_pred_list(v, depth + 1)
        if result is not None:
            return result

    return None


def _extract_predictions(output: Any) -> list[Any]:
    """
    Entry point: given results[0] (a dict whose values may be typed objects),
    return the flat list of prediction objects/dicts.
    """
    if output is None:
        return []

    # Log top-level keys for diagnostics
    if isinstance(output, dict):
        logger.debug("Workflow output keys: %s", list(output.keys()))
        for k, v in output.items():
            logger.debug("  [%s] type=%s", k, type(v).__name__)
            inner = _get(v, "predictions")
            if inner is not None:
                logger.debug("    .predictions type=%s", type(inner).__name__)

    result = _to_pred_list(output)
    if result is None:
        # Last resort: dump what we have so the debug env var helps
        if os.getenv("ROBOFLOW_WORKFLOW_DEBUG"):
            try:
                import json
                logger.warning("RAW OUTPUT: %s",
                               json.dumps(output, default=str)[:3000])
            except Exception:
                logger.warning("RAW OUTPUT (repr): %s", repr(output)[:3000])

        logger.warning(
            "Could not extract predictions from workflow output. "
            "Set ROBOFLOW_WORKFLOW_DEBUG=1 and run scripts/debug_workflow.py "
            "to inspect the raw response structure."
        )
        return []

    logger.debug("_extract_predictions: found %d raw predictions", len(result))
    return result


# ── Convert a single prediction object → our Detection type ──────────────────

def _pred_to_detection(pred: Any, img_w: int, img_h: int, min_confidence: float) -> Detection | None:
    """Convert one SDK prediction object or dict to a Detection, or None to skip."""
    try:
        conf = float(_get(pred, "confidence", default=0.0))
        if conf < min_confidence:
            return None

        # class name: SDK uses .class_name, older code .class_, dicts use "class"
        raw_label = str(
            _get(pred, "class_name", "class_", "class", default="unknown")
        ).strip().lower()
        label = _LABEL_MAP.get(raw_label, raw_label)
        if label not in VALID_CLASSES:
            label = "other"

        # Coordinates: Roboflow centre-xywh
        cx = float(_get(pred, "x", default=0))
        cy = float(_get(pred, "y", default=0))
        bw = float(_get(pred, "width", default=0))
        bh = float(_get(pred, "height", default=0))

        x1 = max(0,     int(cx - bw / 2))
        y1 = max(0,     int(cy - bh / 2))
        x2 = min(img_w, int(cx + bw / 2))
        y2 = min(img_h, int(cy + bh / 2))

        logger.debug("  → %s conf=%.3f bbox=(%d,%d,%d,%d)", label, conf, x1, y1, x2, y2)
        return Detection(label, conf, (x1, y1, x2, y2))
    except Exception as exc:
        logger.debug("  Skipping malformed prediction %r: %s", pred, exc)
        return None


def _parse_detections(
    raw_preds: list[Any],
    img_w: int,
    img_h: int,
    min_confidence: float,
) -> list[Detection]:
    out = []
    for pred in raw_preds:
        det = _pred_to_detection(pred, img_w, img_h, min_confidence)
        if det is not None:
            out.append(det)
    return out


# ── Image encoding ─────────────────────────────────────────────────────────────

def _ndarray_to_base64_jpeg(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise DetectorError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Main detector ──────────────────────────────────────────────────────────────

class RoboflowWorkflowDetector(BaseDetector):
    """
    Calls the Roboflow hosted Workflow via inference-sdk 1.7+.
    Handles typed SDK objects returned by run_workflow() in addition to plain dicts.
    """

    name = "roboflow_workflow"

    def __init__(self, api_key: str, confidence: float = 0.40) -> None:
        if not api_key:
            raise DetectorError(
                "ROBOFLOW_API_KEY is not set. "
                "Get your key from app.roboflow.com → Settings → API Keys."
            )
        try:
            from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        except ImportError as exc:
            raise DetectorError("Run: pip install inference-sdk") from exc

        self._client = InferenceHTTPClient(api_url=_API_URL, api_key=api_key)
        self._client.configure(InferenceConfiguration(api_key_transport="header"))
        self._confidence = confidence
        logger.info("RoboflowWorkflowDetector ready  min_conf=%.2f", confidence)

    def predict(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        b64 = _ndarray_to_base64_jpeg(image)
        raw = self._call_with_retry(b64)
        preds = _extract_predictions(raw)
        logger.info("Workflow: %d raw predictions", len(preds))
        detections = _parse_detections(preds, w, h, self._confidence)
        logger.info("Workflow: %d detections above threshold %.2f", len(detections), self._confidence)
        return detections

    def _call_with_retry(self, b64: str) -> Any:
        last_exc: Exception | None = None
        delay = _RETRY_BACKOFF
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                results = self._client.run_workflow(
                    workspace_name=_WORKSPACE,
                    workflow_id=_WORKFLOW_ID,
                    images={"image": b64},
                )
                logger.debug("run_workflow returned type=%s len=%s",
                             type(results).__name__,
                             len(results) if isinstance(results, list) else "?")
                if not results:
                    logger.warning("Empty results list on attempt %d", attempt)
                    return {}
                return results[0]
            except Exception as exc:
                last_exc = exc
                logger.warning("Attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
        raise DetectorError(
            f"Workflow failed after {_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc
