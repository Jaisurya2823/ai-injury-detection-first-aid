from __future__ import annotations
import cv2
import numpy as np
from .types import Detection, SegmentationResult

def extract_features(image: np.ndarray, detections: list[Detection], seg: SegmentationResult | None) -> dict:
    if image is None or image.size==0:
        return {"labels":[],"location":None,"redness":0.0,"brightness":0.0,"extent":0.0}
    labels=[d.label for d in detections]
    location="unknown"
    if detections:
        h,w=image.shape[:2]
        x1,y1,x2,y2=detections[0].bbox
        cx=(x1+x2)/2; cy=(y1+y2)/2
        location=("upper" if cy<h/3 else "lower" if cy>2*h/3 else "middle")+"-"+("left" if cx<w/3 else "right" if cx>2*w/3 else "center")
    b,g,r=cv2.split(image)
    redness=float(np.mean(np.clip((r.astype(np.float32)-((g.astype(np.float32)+b.astype(np.float32))/2))/255.0,0,1)))
    brightness=float(np.mean(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)))
    return {"labels":labels,"location":location,"redness":redness,"brightness":brightness,"extent":seg.mask_area_ratio if seg else 0.0}
