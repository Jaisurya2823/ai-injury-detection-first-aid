from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

import cv2
from PIL import Image

from app.config import Settings
from app.db.database import Database

IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif", "image/tiff"}
VIDEO_MIMES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/matroska", "video/x-matroska"}


class MediaError(ValueError):
    pass


@dataclass
class StoredMedia:
    media_id: str
    media_type: str
    path: Path
    mime_type: str
    original_size: int
    compressed_size: int
    width: int | None
    height: int | None
    duration: float | None
    sha256: str
    deduplicated: bool = False


class MediaStore:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.settings.ensure_dirs()

    def _write_temp(self, source: bytes | BinaryIO, suffix: str = ".upload") -> tuple[Path, int]:
        fd, tmp_name = tempfile.mkstemp(prefix="upload_", suffix=suffix, dir=self.settings.data_dir / "tmp")
        size = 0
        try:
            with open(fd, "wb", closefd=True) as out:
                if isinstance(source, (bytes, bytearray, memoryview)):
                    payload = bytes(source)
                    out.write(payload)
                    size = len(payload)
                else:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        if not isinstance(chunk, (bytes, bytearray)):
                            raise MediaError("Upload stream returned non-byte data")
                        out.write(chunk)
                        size += len(chunk)
            return Path(tmp_name), size
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def store(self, source: bytes | BinaryIO, session_id: str, mime_type: str) -> StoredMedia:
        if not self.db.session_exists(session_id):
            raise MediaError("Unknown session_id")
        mime_type = (mime_type or "").split(";", 1)[0].strip().lower()
        if mime_type not in IMAGE_MIMES | VIDEO_MIMES:
            raise MediaError(f"Unsupported MIME type: {mime_type or 'missing'}")
        temp, original_size = self._write_temp(source)
        try:
            if original_size <= 0:
                raise MediaError("Upload is empty")
            if original_size > self.settings.max_upload_mb * 1024 * 1024:
                raise MediaError("Upload exceeds configured maximum size")
            original_hash = sha256_file(temp)
            existing = self.db.find_media_by_hash(original_hash, session_id)
            if existing and Path(existing["file_path"]).exists():
                return StoredMedia(
                    existing["media_id"], existing["media_type"], Path(existing["file_path"]), existing["mime_type"],
                    original_size, existing["compressed_size"], existing["width"], existing["height"], existing["duration"],
                    original_hash, True,
                )
            if mime_type in IMAGE_MIMES:
                return self._store_image(temp, session_id, original_size, original_hash)
            return self._store_video(temp, session_id, original_size, original_hash)
        finally:
            temp.unlink(missing_ok=True)

    def _store_image(self, temp: Path, session_id: str, original_size: int, sha: str) -> StoredMedia:
        media_id = str(uuid.uuid4())
        out = self.settings.data_dir / "media" / "images" / f"{media_id}.webp"
        try:
            with Image.open(temp) as src:
                src.load()
                if getattr(src, "is_animated", False):
                    src.seek(0)
                im = src.convert("RGB")
                im.thumbnail((self.settings.image_max_side, self.settings.image_max_side), Image.Resampling.LANCZOS)
                im.save(out, format="WEBP", quality=self.settings.image_quality, method=6, lossless=False)
                width, height = im.size
        except Exception as exc:
            out.unlink(missing_ok=True)
            raise MediaError("Invalid or unreadable image upload") from exc
        return self._record_media(media_id, session_id, "image", out, "image/webp", original_size, sha, width, height, None)

    def _store_video(self, temp: Path, session_id: str, original_size: int, sha: str) -> StoredMedia:
        media_id = str(uuid.uuid4())
        out = self.settings.data_dir / "media" / "videos" / f"{media_id}.mp4"
        duration, width, height = self._probe_video(temp)
        self._transcode_video(temp, out)
        return self._record_media(media_id, session_id, "video", out, "video/mp4", original_size, sha, width, height, duration)

    def _record_media(self, media_id: str, session_id: str, media_type: str, out: Path, mime: str,
                      original_size: int, sha: str, width: int | None, height: int | None, duration: float | None) -> StoredMedia:
        try:
            size = out.stat().st_size
            if size <= 0:
                raise MediaError("Compressed media is empty")
            now = datetime.now(timezone.utc)
            row = {
                "media_id": media_id, "session_id": session_id, "media_type": media_type,
                "file_path": str(out), "mime_type": mime, "original_size": original_size,
                "compressed_size": size, "width": width, "height": height, "duration": duration,
                "sha256": sha, "created_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=self.settings.retention_hours)).isoformat(),
                "processing_status": "STORED",
            }
            self.db.create_media(row)
        except Exception:
            out.unlink(missing_ok=True)
            raise
        return StoredMedia(media_id, media_type, out, mime, original_size, size, width, height, duration, sha)

    def _probe_video(self, path: Path) -> tuple[float | None, int | None, int | None]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise MediaError("Video could not be decoded")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
            if width is None or height is None:
                raise MediaError("Video dimensions could not be determined")
            duration = (frames / fps) if fps > 0 and frames > 0 else None
            ok, frame = cap.read()
            if not ok or frame is None:
                raise MediaError("Video contains no readable frames")
            return duration, width, height
        finally:
            cap.release()

    def _transcode_video(self, src: Path, dst: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise MediaError("FFmpeg is required to store videos as compressed MP4/H.264")
        scale = max(320, int(self.settings.video_max_side))
        fps = max(1, int(self.settings.video_fps))
        crf = max(18, min(40, int(self.settings.video_crf)))
        vf = f"scale='if(gt(iw,ih),min({scale},iw),-2)':'if(gt(iw,ih),-2,min({scale},ih))'"
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
               "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired as exc:
            dst.unlink(missing_ok=True)
            raise MediaError("Video compression timed out") from exc
        if proc.returncode != 0 or not dst.exists():
            dst.unlink(missing_ok=True)
            raise MediaError(f"Video compression failed: {proc.stderr[-500:]}")

    def cleanup(self) -> int:
        removed = 0
        for p in self.db.cleanup_expired_media():
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except OSError:
                continue
        tmp_dir = self.settings.data_dir / "tmp"
        for p in tmp_dir.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
                    removed += 1
            except OSError:
                continue
        return removed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
