from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Region:
    bbox: tuple[int,int,int,int]
    score: float

class HumanBodyLocalizer:
    """Model boundary for human/body-region localization.
    The default implementation passes the full frame through safely.
    Replace it with a person detector when a trained model is available.
    """
    def localize(self, image: np.ndarray) -> list[Region]:
        h,w=image.shape[:2]
        return [Region((0,0,w,h),1.0)] if image is not None and image.size else []
