from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .types import Detection

VALID_CLASSES = {"cut", "abrasion", "bruise", "burn", "swelling", "other", "unknown"}

class DetectorError(RuntimeError):
    pass

class BaseDetector:
    name = "base"
    def predict(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError

class NullDetector(BaseDetector):
    name = "null"
    def predict(self, image: np.ndarray) -> list[Detection]:
        return []

class YOLODetector(BaseDetector):
    name = "yolo"
    def __init__(self, weights: Path):
        if not weights.exists():
            raise DetectorError(f"YOLO weights not found: {weights}")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorError("ultralytics is not installed. Install it and provide trained injury weights.") from e
        try:
            self.model = YOLO(str(weights))
        except Exception as e:
            raise DetectorError(f"Could not load YOLO weights: {e}") from e

    def predict(self, image: np.ndarray) -> list[Detection]:
        try:
            results = self.model.predict(source=image, verbose=False)
        except Exception as e:
            raise DetectorError(f"YOLO inference failed: {e}") from e
        out: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    cls = int(box.cls.item())
                    conf = max(0.0, min(1.0, float(box.conf.item())))
                    label = str(names.get(cls, "unknown")).strip().lower()
                    if label not in VALID_CLASSES:
                        label = "other"
                    xyxy = box.xyxy[0].tolist()
                    coords = tuple(int(round(max(0, x))) for x in xyxy)
                    if len(coords) == 4:
                        out.append(Detection(label, conf, coords))
                except Exception:
                    continue
        return out

class DemoDetector(BaseDetector):
    """Explicit demonstration detector. It is NOT a medical model."""
    name = "demo"
    def predict(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        # Detect a deliberately synthetic red marker, useful only for UI demonstrations.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 100, 80), (179, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area > max(500, int(w * h * 0.01)):
                best = (x, y, bw, bh, area)
                break
        if best:
            x, y, bw, bh, area = best
            # Intentionally labeled "other" because demo colour detection is not injury diagnosis.
            return [Detection("other", 0.80, (int(x), int(y), int(x + bw), int(y + bh)))]
        return []
