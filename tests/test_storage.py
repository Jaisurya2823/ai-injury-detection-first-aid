from __future__ import annotations
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from app.config import Settings
from app.db import Database
from app.media import MediaStore, MediaError


def make_settings(tmp_path):
    return Settings(data_dir=tmp_path/"data",db_path=tmp_path/"data"/"db.sqlite3",yolo_weights=tmp_path/"best.pt",model_mode="null",image_max_side=800,image_quality=80,video_max_side=640,video_fps=8,video_crf=30,max_upload_mb=5,retention_hours=1,confidence_threshold=.7,ood_threshold=.55)

def make_jpeg():
    img=Image.fromarray(np.full((400,400,3), [120,80,80], dtype=np.uint8))
    b=BytesIO(); img.save(b,"JPEG",quality=95); return b.getvalue()

def test_image_is_compressed_and_recorded(tmp_path):
    s=make_settings(tmp_path); db=Database(s.db_path); store=MediaStore(s,db); db.create_session("s","u")
    m=store.store(make_jpeg(),"s","image/jpeg")
    assert m.path.exists() and m.path.suffix==".webp"
    assert m.compressed_size < m.original_size
    assert db.count("media")==1

def test_duplicate_upload_reuses_file(tmp_path):
    s=make_settings(tmp_path); db=Database(s.db_path); store=MediaStore(s,db); db.create_session("s","u")
    raw=make_jpeg(); a=store.store(raw,"s","image/jpeg"); b=store.store(raw,"s","image/jpeg")
    assert b.deduplicated is True
    assert a.path==b.path
    assert db.count("media")==1

def test_rejects_unsupported_type(tmp_path):
    s=make_settings(tmp_path); db=Database(s.db_path); store=MediaStore(s,db); db.create_session("s","u")
    try: store.store(b"abc","s","application/pdf")
    except MediaError: return
    assert False, "expected MediaError"

def test_video_is_transcoded_to_mp4(tmp_path):
    import subprocess
    import cv2, numpy as np
    s=make_settings(tmp_path); db=Database(s.db_path); store=MediaStore(s,db); db.create_session("s","u")
    raw=tmp_path/"src.mp4"; out=tmp_path/"src_processed.mp4"
    # Generate a tiny valid MP4 with ffmpeg, avoiding codec assumptions in OpenCV.
    subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","lavfi","-i","color=c=gray:s=320x240:r=12:d=1",str(raw)],check=True)
    m=store.store(raw.read_bytes(),"s","video/mp4")
    assert m.path.exists() and m.path.suffix==".mp4"
    assert m.compressed_size > 0 and m.width == 320 and m.height == 240
    assert db.count("media")==1
