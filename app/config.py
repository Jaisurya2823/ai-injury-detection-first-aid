from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


# ── .env loader ───────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """
    Find and load .env from the first of several candidate locations.

    Search order (stops at first .env found):
      1. Directory of the script that started the process  (sys.argv[0] parent)
      2. Current working directory
      3. This file's grandparent  (project_root/app/config.py → project_root)
      4. Any parent of this file up to filesystem root

    Uses python-dotenv when installed; otherwise uses a built-in line parser.
    Never overwrites env vars that are already set in the shell.
    """
    candidates: list[Path] = []

    # 1. Script directory (where run_streamlit.py / run.py lives)
    if sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]).resolve().parent / ".env")

    # 2. CWD
    candidates.append(Path.cwd() / ".env")

    # 3. This file's grandparent (project_root/app/config.py → project_root/.env)
    candidates.append(Path(__file__).resolve().parent.parent / ".env")

    # 4. Walk up from this file
    p = Path(__file__).resolve().parent
    for _ in range(6):
        p = p.parent
        candidates.append(p / ".env")

    env_path: Path | None = None
    for c in candidates:
        if c.exists() and c.is_file():
            env_path = c
            break

    if env_path is None:
        return  # no .env found — rely on shell environment

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    # Built-in fallback parser (no variable substitution, handles quoted values)
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # strip inline comments and quotes
            val = val.split("#")[0].strip().strip('"').strip("'").strip()
            if key and key not in os.environ:
                os.environ[key] = val


# Run once at import time — before any os.getenv() call.
_load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ── Settings ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    yolo_weights: Path
    model_mode: str
    image_max_side: int
    image_quality: int
    video_max_side: int
    video_fps: int
    video_crf: int
    max_upload_mb: int
    retention_hours: int
    confidence_threshold: float
    ood_threshold: float

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("FIRST_AID_DATA_DIR", "data")).resolve()
        db_path  = Path(os.getenv("FIRST_AID_DB_PATH",
                                  str(data_dir / "first_aid.sqlite3"))).resolve()
        mode = os.getenv("FIRST_AID_MODEL_MODE", "workflow").strip().lower()
        return cls(
            data_dir=data_dir,
            db_path=db_path,
            yolo_weights=Path(os.getenv("FIRST_AID_YOLO_WEIGHTS",
                                        "models/best.pt")).resolve(),
            model_mode=mode,
            image_max_side=_int("FIRST_AID_IMAGE_MAX_SIDE", 1600),
            image_quality=max(40, min(95, _int("FIRST_AID_IMAGE_QUALITY", 82))),
            video_max_side=_int("FIRST_AID_VIDEO_MAX_SIDE", 1280),
            video_fps=max(1, _int("FIRST_AID_VIDEO_FPS", 8)),
            video_crf=max(18, min(40, _int("FIRST_AID_VIDEO_CRF", 30))),
            max_upload_mb=max(1, _int("FIRST_AID_MAX_UPLOAD_MB", 100)),
            retention_hours=max(1, _int("FIRST_AID_RETENTION_HOURS", 24)),
            # Default thresholds low enough for YOLO26m outputs (0.40–0.85 range)
            confidence_threshold=max(0.0, min(1.0,
                _float("FIRST_AID_CONFIDENCE_THRESHOLD", 0.40))),
            ood_threshold=max(0.0, min(1.0,
                _float("FIRST_AID_OOD_THRESHOLD", 0.60))),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "media" / "images").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "media" / "videos").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "tmp").mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
