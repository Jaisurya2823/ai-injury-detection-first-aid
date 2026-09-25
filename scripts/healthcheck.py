from __future__ import annotations
import json
import shutil
from pathlib import Path
from app.config import Settings
from app.core.factory import build_service

s = Settings.from_env()
s.ensure_dirs()
svc = build_service(s)
print(json.dumps({
    "status": "ok",
    "db": str(s.db_path),
    "model_mode": s.model_mode,
    "detector": getattr(svc.pipeline.detector, "name", svc.pipeline.detector.__class__.__name__),
    "knowledge_docs": svc.db.count("knowledge_documents"),
    "ffmpeg": shutil.which("ffmpeg") or None,
    "data_dir": str(s.data_dir),
}, indent=2))
