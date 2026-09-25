from __future__ import annotations
from pathlib import Path
from io import BytesIO
import cv2, numpy as np
import pytest
from PIL import Image
from app.config import Settings
from app.db import Database
from app.media import MediaStore, MediaError


def settings(tmp_path, max_mb=5):
    return Settings(data_dir=tmp_path/"data",db_path=tmp_path/"data"/"db.sqlite3",yolo_weights=tmp_path/"best.pt",model_mode="null",image_max_side=800,image_quality=80,video_max_side=640,video_fps=8,video_crf=30,max_upload_mb=max_mb,retention_hours=1,confidence_threshold=.7,ood_threshold=.55)

def init(tmp_path):
    s=settings(tmp_path); db=Database(s.db_path); store=MediaStore(s,db); db.create_session("session","user"); return s,db,store

def test_corrupt_image_is_rejected_cleanly(tmp_path):
    _,db,store=init(tmp_path)
    with pytest.raises(MediaError): store.store(b"not-an-image", "session", "image/jpeg")
    assert db.count("media")==0

def test_oversized_upload_is_rejected(tmp_path):
    _,db,store=init(tmp_path)
    with pytest.raises(MediaError): store.store(b"x"*(6*1024*1024), "session", "image/jpeg")
    assert db.count("media")==0

def test_invalid_video_is_rejected_cleanly(tmp_path):
    _,db,store=init(tmp_path)
    with pytest.raises(MediaError): store.store(b"not-a-video", "session", "video/mp4")
    assert db.count("media")==0

def test_cleanup_removes_expired_media(tmp_path):
    s,db,store=init(tmp_path)
    im=Image.fromarray(np.full((300,300,3),200,np.uint8)); b=BytesIO(); im.save(b,"JPEG")
    # Use zero-ish retention by directly making the DB expiry old after storage.
    m=store.store(b.getvalue(),"session","image/jpeg")
    with db.connect() as c:
        c.execute("UPDATE media SET expires_at='2000-01-01T00:00:00+00:00' WHERE media_id=?",(m.media_id,))
    assert m.path.exists(); store.cleanup(); assert not m.path.exists(); assert db.count("media")==0

def test_missing_ffmpeg_fails_non_mp4(monkeypatch,tmp_path):
    s,db,store=init(tmp_path)
    monkeypatch.setattr("app.media.storage.shutil.which", lambda name: None)
    with pytest.raises(MediaError): store._transcode_video(tmp_path/"input.mov", tmp_path/"out.mp4")

def test_session_isolation_for_duplicate_hash(tmp_path):
    s=settings(tmp_path); db=Database(s.db_path); store=MediaStore(s,db)
    db.create_session("s1","u1"); db.create_session("s2","u2")
    im=Image.fromarray(np.full((300,300,3),150,np.uint8)); b=BytesIO(); im.save(b,"JPEG",quality=95); raw=b.getvalue()
    a=store.store(raw,"s1","image/jpeg"); c=store.store(raw,"s2","image/jpeg")
    assert a.media_id != c.media_id
    assert db.count("media") == 2


def test_temp_directory_is_clean_after_store(tmp_path):
    s,db,store=init(tmp_path)
    im=Image.fromarray(np.full((300,300,3),150,np.uint8)); b=BytesIO(); im.save(b,"JPEG")
    store.store(b.getvalue(),"session","image/jpeg")
    assert not list((s.data_dir / "tmp").glob("*"))

def test_legacy_database_migrates(tmp_path):
    import sqlite3
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
    PRAGMA foreign_keys=OFF;
    CREATE TABLE users(user_id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
    CREATE TABLE sessions(session_id TEXT PRIMARY KEY, user_id TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE media(media_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, media_type TEXT NOT NULL, file_path TEXT NOT NULL,
                       mime_type TEXT NOT NULL, original_size INTEGER NOT NULL, compressed_size INTEGER NOT NULL,
                       width INTEGER, height INTEGER, duration REAL, sha256 TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
                       expires_at TEXT NOT NULL, processing_status TEXT NOT NULL);
    CREATE TABLE analysis_results(analysis_id TEXT PRIMARY KEY, media_id TEXT NOT NULL, injury_labels TEXT NOT NULL,
                       confidence REAL NOT NULL, ood_score REAL NOT NULL, severity TEXT NOT NULL, location TEXT,
                       analysis_status TEXT NOT NULL, warnings TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE context_answers(context_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, mechanism TEXT,
                       time_since_injury TEXT, bleeding_status TEXT, pain_level TEXT, movement_limitation TEXT, red_flags TEXT NOT NULL);
    CREATE TABLE guidance(guidance_id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, risk_level TEXT NOT NULL,
                       action TEXT NOT NULL, warning TEXT NOT NULL, source_ids TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE knowledge_documents(document_id TEXT PRIMARY KEY,title TEXT NOT NULL,source TEXT NOT NULL,version TEXT NOT NULL,updated_at TEXT NOT NULL);
    """)
    conn.commit(); conn.close()
    db = Database(db_path)
    with db.connect() as c:
        assert "disagreement" in {r[1] for r in c.execute("PRAGMA table_info(analysis_results)").fetchall()}
        sql = c.execute("SELECT sql FROM sqlite_master WHERE name='media'").fetchone()[0]
        assert "UNIQUE" not in sql.upper()
