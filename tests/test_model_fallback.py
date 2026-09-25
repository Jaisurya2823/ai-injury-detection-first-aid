from pathlib import Path
from app.config import Settings
from app.core.factory import build_service

def test_missing_yolo_weights_falls_back_to_safe_null(tmp_path):
    s=Settings(data_dir=tmp_path/'data',db_path=tmp_path/'data'/'db.sqlite3',yolo_weights=tmp_path/'missing.pt',model_mode='yolo',image_max_side=800,image_quality=80,video_max_side=640,video_fps=8,video_crf=30,max_upload_mb=5,retention_hours=1,confidence_threshold=.7,ood_threshold=.55)
    svc=build_service(s)
    assert svc.pipeline.detector.__class__.__name__=='NullDetector'
