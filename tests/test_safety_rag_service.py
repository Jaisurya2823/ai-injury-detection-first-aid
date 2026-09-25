from __future__ import annotations
from pathlib import Path
from app.safety import SafetyEngine
from app.rag import LocalRAG, GuidanceEngine
from app.config import Settings
from app.db import Database
from app.media import MediaStore
from app.ai.pipeline import InjuryPipeline
from app.ai.quality import QualityGate
from app.ai.detector import NullDetector
from app.ai.segmentation import BoxSegmenter
from app.ai.reliability import ReliabilityEngine
from app.core.service import FirstAidService
from io import BytesIO
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def test_red_flag_escalates():
    d=SafetyEngine().decide(["cut"],.95,"mild",{"red_flags":["continuous_bleeding"]})
    assert d.risk_level=="emergency_or_urgent"

def test_low_confidence_abstains():
    d=SafetyEngine().decide(["cut"],.45,"mild",{})
    assert d.risk_level=="insufficient_evidence"

def test_rag_retrieves_burn_source():
    rag=LocalRAG(ROOT/"knowledge"/"sources.json")
    ev=rag.retrieve("burn cool running water",3)
    assert ev and ev[0].document_id=="who-burns-2023"

def test_service_end_to_end(tmp_path):
    settings=Settings(data_dir=tmp_path/"data",db_path=tmp_path/"data"/"db.sqlite3",yolo_weights=tmp_path/"best.pt",model_mode="null",image_max_side=800,image_quality=80,video_max_side=640,video_fps=8,video_crf=30,max_upload_mb=5,retention_hours=1,confidence_threshold=.7,ood_threshold=.55)
    db=Database(settings.db_path); media=MediaStore(settings,db)
    pipeline=InjuryPipeline(NullDetector(),QualityGate(min_width=10,min_height=10,blur_threshold=1),BoxSegmenter(),ReliabilityEngine())
    rag=LocalRAG(ROOT/"knowledge"/"sources.json")
    svc=FirstAidService(settings,pipeline,db,media,SafetyEngine(),GuidanceEngine(rag))
    sid=svc.new_session(); im=Image.fromarray(np.full((300,300,3),180,np.uint8)); im=Image.fromarray(np.array(im)); im.save  # keep image object stable
    from PIL import ImageDraw
    draw=ImageDraw.Draw(im); draw.rectangle((20,20,280,280), outline=(30,30,30), width=5)
    b=BytesIO(); im.save(b,"JPEG")
    out=svc.process_upload(b.getvalue(),"image/jpeg",sid,{})
    assert out["analysis"]["status"]=="ABSTAIN"
    assert db.count("analysis_results")==1 and db.count("guidance")==1

def test_unknown_or_other_does_not_get_specific_guidance():
    d = SafetyEngine().decide(["other"], .90, "mild", {})
    assert d.risk_level == "insufficient_evidence"
