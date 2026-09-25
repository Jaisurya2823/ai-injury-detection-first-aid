from __future__ import annotations
import numpy as np
import cv2
from .types import Detection, SegmentationResult

class BoxSegmenter:
    """Safe baseline segmenter: represents the detector ROI as an affected-region mask.
    Replace with a trained segmentation model when weights/data are available."""
    def segment(self, image: np.ndarray, detection: Detection) -> SegmentationResult:
        h,w=image.shape[:2]
        x1,y1,x2,y2=detection.bbox
        x1=max(0,min(w,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
        area=max(0,x2-x1)*max(0,y2-y1)
        ratio=area/float(max(1,w*h))
        return SegmentationResult(ratio)
