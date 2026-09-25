from __future__ import annotations
import cv2
import numpy as np
from .types import QualityReport


class QualityGate:
    def __init__(self, min_width=160, min_height=160, blur_threshold=45.0, min_brightness=25.0, max_brightness=245.0):
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def check_image(self, image: np.ndarray) -> QualityReport:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return QualityReport(False, 0.0, 0.0, 0, 0, ["empty_image"], 1.0)
        if image.ndim not in (2, 3):
            return QualityReport(False, 0.0, 0.0, 0, 0, ["invalid_image_shape"], 1.0)
        h, w = image.shape[:2]
        reasons: list[str] = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        if w < self.min_width or h < self.min_height:
            reasons.append("resolution_too_low")
        if blur < self.blur_threshold:
            reasons.append("image_too_blurry")
        if brightness < self.min_brightness:
            reasons.append("image_too_dark")
        if brightness > self.max_brightness:
            reasons.append("image_too_bright")
        return QualityReport(not reasons, blur, brightness, w, h, reasons, 0.0)

    def check_file(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return self.check_image(image)
