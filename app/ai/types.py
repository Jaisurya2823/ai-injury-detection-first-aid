from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

@dataclass
class SegmentationResult:
    mask_area_ratio: float
    mask_path: str | None = None

@dataclass
class QualityReport:
    passed: bool
    blur_score: float
    brightness: float
    width: int
    height: int
    reasons: list[str] = field(default_factory=list)
    ood_hint: float = 0.0

@dataclass
class ReliabilityReport:
    reliable: bool
    confidence: float
    ood_score: float
    disagreement: float
    reasons: list[str] = field(default_factory=list)

@dataclass
class AnalysisResult:
    labels: list[str]
    confidence: float
    ood_score: float
    severity: str
    location: str | None
    status: str
    warnings: list[str] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    mask_area_ratio: float = 0.0
    disagreement: float = 0.0
    evidence: list[dict[str, Any]] = field(default_factory=list)
