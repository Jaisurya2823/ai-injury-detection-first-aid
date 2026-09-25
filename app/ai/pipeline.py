from __future__ import annotations
from pathlib import Path
import cv2
from .types import AnalysisResult
from .quality import QualityGate
from .detector import BaseDetector, DetectorError
from .segmentation import BoxSegmenter
from .features import extract_features
from .reliability import ReliabilityEngine
from .localization import HumanBodyLocalizer
from .classifier import MultiLabelClassifier
from .risk import RiskSeverityEngine


class InjuryPipeline:
    def __init__(self, detector: BaseDetector, quality: QualityGate, segmenter: BoxSegmenter,
                 reliability: ReliabilityEngine, localizer: HumanBodyLocalizer | None = None,
                 classifier: MultiLabelClassifier | None = None, risk_engine: RiskSeverityEngine | None = None):
        self.detector = detector
        self.quality = quality
        self.segmenter = segmenter
        self.reliability = reliability
        self.localizer = localizer or HumanBodyLocalizer()
        self.classifier = classifier or MultiLabelClassifier()
        self.risk_engine = risk_engine or RiskSeverityEngine()

    def _safe_predict(self, image):
        try:
            return self.detector.predict(image), None
        except DetectorError as exc:
            return [], f"detector_error:{exc}"
        except Exception as exc:
            return [], f"detector_unexpected_error:{type(exc).__name__}"

    def analyze_image(self, path: str | Path, context: dict | None = None) -> AnalysisResult:
        context = context or {}
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        q = self.quality.check_image(image)
        if not q.passed:
            return AnalysisResult([], 0.0, 1.0, "insufficient_evidence", None, "QUALITY_REJECTED", q.reasons)
        regions = self.localizer.localize(image)
        if not regions:
            return AnalysisResult([], 0.0, 1.0, "insufficient_evidence", None, "ABSTAIN", ["human_region_not_found"])
        detections, err = self._safe_predict(image)
        if err:
            return AnalysisResult([], 0.0, 1.0, "insufficient_evidence", None, "ABSTAIN", [err])
        rel = self.reliability.evaluate(detections)
        if not rel.reliable:
            labels = sorted({d.label for d in detections}) or ["unknown"]
            return AnalysisResult(labels, rel.confidence, rel.ood_score, "insufficient_evidence", None, "ABSTAIN", rel.reasons, detections, 0.0, rel.disagreement)
        cls = self.classifier.classify(detections)
        first = max(detections, key=lambda d: d.confidence)
        seg = self.segmenter.segment(image, first)
        features = extract_features(image, detections, seg)
        severity = self.risk_engine.estimate(cls.labels, cls.confidence, seg.mask_area_ratio, context)
        return AnalysisResult(cls.labels, cls.confidence, rel.ood_score, severity, features["location"], "OK", [], detections, seg.mask_area_ratio)

    def analyze_video(self, path: str | Path, sample_every_n_frames: int = 10) -> AnalysisResult:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return AnalysisResult([], 0.0, 1.0, "insufficient_evidence", None, "VIDEO_REJECTED", ["video_decode_failed"])
        all_detections = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % max(1, sample_every_n_frames) == 0:
                    q = self.quality.check_image(frame)
                    if q.passed:
                        detections, err = self._safe_predict(frame)
                        if err:
                            return AnalysisResult([], 0.0, 1.0, "insufficient_evidence", None, "ABSTAIN", [err])
                        all_detections.extend(detections)
                idx += 1
                if idx >= 900:
                    break
        finally:
            cap.release()
        rel = self.reliability.evaluate(all_detections)
        if not rel.reliable:
            labels = sorted({d.label for d in all_detections}) or ["unknown"]
            return AnalysisResult(labels, rel.confidence, rel.ood_score, "insufficient_evidence", None, "ABSTAIN", rel.reasons, all_detections, 0.0, rel.disagreement)
        cls = self.classifier.classify(all_detections)
        severity = self.risk_engine.estimate(cls.labels, cls.confidence, 0.0, {})
        return AnalysisResult(cls.labels, cls.confidence, rel.ood_score, severity, None, "OK", [], all_detections, 0.0, rel.disagreement)
