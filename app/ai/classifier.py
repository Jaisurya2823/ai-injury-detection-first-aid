from __future__ import annotations
from dataclasses import dataclass
from .types import Detection

@dataclass(frozen=True)
class Classification:
    labels: list[str]
    confidence: float

class MultiLabelClassifier:
    """Aggregates detector outputs into a multi-label result.
    A learned classifier can replace this adapter without changing the pipeline contract."""
    def classify(self, detections: list[Detection]) -> Classification:
        if not detections: return Classification(["unknown"],0.0)
        by_label={}
        for d in detections: by_label[d.label]=max(by_label.get(d.label,0.0),d.confidence)
        labels=sorted(by_label, key=by_label.get, reverse=True)
        return Classification(labels, max(by_label.values()))
