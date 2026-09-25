from __future__ import annotations

import json
import uuid
from typing import BinaryIO

from app.ai.types import AnalysisResult
from app.config import Settings
from app.db import Database
from app.db.database import utc_now
from app.media import MediaStore
from app.rag import GuidanceEngine
from app.safety import SafetyEngine


class FirstAidService:
    def __init__(self, settings: Settings, pipeline, db: Database, media: MediaStore,
                 safety: SafetyEngine, guidance: GuidanceEngine):
        self.settings = settings
        self.pipeline = pipeline
        self.db = db
        self.media = media
        self.safety = safety
        self.guidance = guidance

    def new_session(self, user_id: str = "anonymous") -> str:
        user_id = (user_id or "anonymous").strip()[:100]
        sid = str(uuid.uuid4())
        self.db.create_session(sid, user_id)
        return sid

    @staticmethod
    def _analysis_dict(analysis: AnalysisResult) -> dict:
        return {
            "labels": analysis.labels,
            "confidence": analysis.confidence,
            "ood_score": analysis.ood_score,
            "severity": analysis.severity,
            "location": analysis.location,
            "status": analysis.status,
            "warnings": analysis.warnings,
            "mask_area_ratio": analysis.mask_area_ratio,
            "disagreement": analysis.disagreement,
            "detections": [
                {"label": d.label, "confidence": d.confidence, "bbox": [int(v) for v in d.bbox]}
                for d in analysis.detections
            ],
        }

    def process_upload(self, source: bytes | BinaryIO, mime_type: str, session_id: str, context: dict | None = None) -> dict:
        if not self.db.session_exists(session_id):
            raise ValueError("Session does not exist")
        context = context or {}
        media = self.media.store(source, session_id, mime_type)
        if media.media_type == "image":
            analysis = self.pipeline.analyze_image(media.path, context)
        else:
            analysis = self.pipeline.analyze_video(media.path)
        decision = self.safety.decide(analysis.labels, analysis.confidence, analysis.severity, context)
        guidance = self.guidance.generate(analysis.labels, decision, context)
        aid = str(uuid.uuid4())
        analysis_row = {
            "analysis_id": aid,
            "media_id": media.media_id,
            "injury_labels": json.dumps(analysis.labels),
            "confidence": float(analysis.confidence),
            "ood_score": float(analysis.ood_score),
            "disagreement": float(getattr(analysis, "disagreement", 0.0) or 0.0),
            "severity": analysis.severity,
            "location": analysis.location,
            "analysis_status": analysis.status,
            "warnings": json.dumps(analysis.warnings),
            "created_at": utc_now(),
        }
        context_row = {
            "context_id": str(uuid.uuid4()),
            "analysis_id": aid,
            "mechanism": context.get("mechanism"),
            "time_since_injury": context.get("time_since_injury"),
            "bleeding_status": context.get("bleeding_status"),
            "pain_level": context.get("pain_level"),
            "movement_limitation": context.get("movement_limitation"),
            "red_flags": json.dumps(context.get("red_flags", [])),
        }
        guidance_row = {
            "guidance_id": str(uuid.uuid4()),
            "analysis_id": aid,
            "risk_level": guidance["risk_level"],
            "action": " ".join(guidance["steps"]),
            "warning": guidance["warning"],
            "source_ids": json.dumps([e["document_id"] for e in guidance["evidence"]]),
            "created_at": utc_now(),
        }
        try:
            self.db.create_analysis_bundle(analysis_row, context_row, guidance_row)
        except Exception:
            # Prevent a partially persisted result from being left behind if DB persistence fails.
            try:
                with self.db.connect() as c:
                    c.execute("DELETE FROM media WHERE media_id=?", (media.media_id,))
            finally:
                media.path.unlink(missing_ok=True)
            raise
        return {
            "analysis_id": aid,
            "media_id": media.media_id,
            "media_type": media.media_type,
            "stored_path": str(media.path),
            "original_size": media.original_size,
            "compressed_size": media.compressed_size,
            "deduplicated": media.deduplicated,
            "analysis": self._analysis_dict(analysis),
            "safety": decision.__dict__,
            "guidance": guidance,
        }

    def history(self, session_id: str, limit: int = 20) -> list[dict]:
        if not self.db.session_exists(session_id):
            raise ValueError("Session does not exist")
        rows = self.db.recent_history(session_id, limit)
        out = []
        for row in rows:
            row = dict(row)
            for key in ("injury_labels", "warnings"):
                try:
                    row[key] = json.loads(row[key])
                except Exception:
                    row[key] = []
            out.append(row)
        return out

    def cleanup(self) -> int:
        return self.media.cleanup()

    def storage_stats(self) -> dict[str, int]:
        stats = self.db.storage_stats()
        stats["compression_saved_bytes"] = max(0, stats["original_bytes"] - stats["compressed_bytes"])
        stats["compression_ratio"] = round(stats["compressed_bytes"] / stats["original_bytes"], 4) if stats["original_bytes"] else 0.0
        return stats
