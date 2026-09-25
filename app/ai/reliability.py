from __future__ import annotations
from .types import Detection, ReliabilityReport


class ReliabilityEngine:
    def __init__(self, confidence_threshold: float = 0.70, ood_threshold: float = 0.55):
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.ood_threshold = max(0.0, min(1.0, float(ood_threshold)))

    def evaluate(self, detections: list[Detection]) -> ReliabilityReport:
        if not detections:
            return ReliabilityReport(False, 0.0, 1.0, 1.0, ["no_injury_detection"])
        confidences = [max(0.0, min(1.0, d.confidence)) for d in detections]
        labels = {d.label for d in detections}
        confidence = max(confidences)
        disagreement = 0.0 if len(labels) <= 1 else min(1.0, (len(labels) - 1) / max(1, len(detections)))
        ood = 1.0 - confidence
        reasons: list[str] = []
        if confidence < self.confidence_threshold:
            reasons.append("confidence_below_threshold")
        if ood > self.ood_threshold:
            reasons.append("ood_risk_high")
        if disagreement > 0.5:
            reasons.append("model_disagreement")
        return ReliabilityReport(not reasons, confidence, ood, disagreement, reasons)
