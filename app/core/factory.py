from __future__ import annotations
import os
from pathlib import Path

from app.config import Settings
from app.db import Database
from app.media import MediaStore
from app.ai.quality import QualityGate
from app.ai.detector import NullDetector, YOLODetector, DemoDetector, DetectorError
from app.ai.segmentation import BoxSegmenter
from app.ai.reliability import ReliabilityEngine
from app.ai.pipeline import InjuryPipeline
from app.ai.localization import HumanBodyLocalizer
from app.ai.classifier import MultiLabelClassifier
from app.ai.risk import RiskSeverityEngine
from app.safety import SafetyEngine
from app.rag import LocalRAG, GuidanceEngine
from app.core.service import FirstAidService


def _build_detector(settings: Settings):
    """
    Modes (set via FIRST_AID_MODEL_MODE):
      workflow  → RoboflowWorkflowDetector (hosted YOLO26m workflow — recommended)
      roboflow  → RoboflowDetector         (hosted model API, older approach)
      claude    → ClaudeVLMDetector        (Claude vision API)
      yolo      → YOLODetector             (local best.pt, GPU optional)
      demo      → DemoDetector             (red marker only)
      null      → NullDetector
    """
    mode = (settings.model_mode or "null").lower()

    if mode == "workflow":
        try:
            from app.ai.workflow_detector import RoboflowWorkflowDetector
            return RoboflowWorkflowDetector(
                api_key=os.getenv("ROBOFLOW_API_KEY", ""),
                confidence=settings.confidence_threshold,
            )
        except DetectorError as exc:
            import logging
            logging.getLogger(__name__).error(
                "RoboflowWorkflowDetector failed (%s); falling back to NullDetector.", exc
            )
            return NullDetector()

    if mode == "roboflow":
        try:
            from app.ai.roboflow_detector import RoboflowDetector
            return RoboflowDetector(
                api_key=os.getenv("ROBOFLOW_API_KEY", ""),
                model_id=os.getenv("ROBOFLOW_MODEL_ID", ""),
                confidence=settings.confidence_threshold,
            )
        except DetectorError as exc:
            import logging
            logging.getLogger(__name__).error(
                "RoboflowDetector failed (%s); falling back to NullDetector.", exc
            )
            return NullDetector()

    if mode == "claude":
        try:
            from app.ai.vlm_detector import ClaudeVLMDetector
            return ClaudeVLMDetector()
        except DetectorError as exc:
            import logging
            logging.getLogger(__name__).error(
                "ClaudeVLMDetector failed (%s); falling back to NullDetector.", exc
            )
            return NullDetector()

    if mode == "yolo":
        try:
            return YOLODetector(settings.yolo_weights)
        except DetectorError:
            return NullDetector()

    if mode == "demo":
        return DemoDetector()

    return NullDetector()


def build_service(settings: Settings) -> FirstAidService:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    media = MediaStore(settings, db)
    detector = _build_detector(settings)
    pipeline = InjuryPipeline(
        detector=detector,
        quality=QualityGate(),
        segmenter=BoxSegmenter(),
        reliability=ReliabilityEngine(settings.confidence_threshold, settings.ood_threshold),
        localizer=HumanBodyLocalizer(),
        classifier=MultiLabelClassifier(),
        risk_engine=RiskSeverityEngine(),
    )
    rag_path = Path(__file__).resolve().parents[2] / "knowledge" / "sources.json"
    rag = LocalRAG(rag_path)
    db.save_knowledge_documents(rag.docs)
    return FirstAidService(settings, pipeline, db, media, SafetyEngine(), GuidanceEngine(rag))
