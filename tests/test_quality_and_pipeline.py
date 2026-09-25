from __future__ import annotations
import numpy as np, cv2
from app.ai.quality import QualityGate
from app.ai.detector import NullDetector, DemoDetector
from app.ai.pipeline import InjuryPipeline
from app.ai.segmentation import BoxSegmenter
from app.ai.reliability import ReliabilityEngine
from pathlib import Path

def test_quality_rejects_blurry():
    q=QualityGate(min_width=10,min_height=10,blur_threshold=45)
    img=np.zeros((200,200,3),np.uint8)
    r=q.check_image(img)
    assert not r.passed
    assert "image_too_dark" in r.reasons

def test_null_detector_abstains(tmp_path):
    img=np.full((300,300,3),180,np.uint8)
    cv2.rectangle(img,(20,20),(280,280),(120,120,120),4)
    p=tmp_path/"x.jpg"; cv2.imwrite(str(p),img)
    pipeline=InjuryPipeline(NullDetector(),QualityGate(min_width=10,min_height=10,blur_threshold=1),BoxSegmenter(),ReliabilityEngine())
    r=pipeline.analyze_image(p)
    assert r.status=="ABSTAIN"
    assert r.severity=="insufficient_evidence"

def test_demo_detector_is_explicitly_non_medical(tmp_path):
    img=np.full((300,300,3),180,np.uint8); cv2.line(img,(0,0),(299,299),(0,0,255),5)
    p=tmp_path/"x.jpg"; cv2.imwrite(str(p),img)
    pipeline=InjuryPipeline(DemoDetector(),QualityGate(min_width=10,min_height=10,blur_threshold=1),BoxSegmenter(),ReliabilityEngine())
    r=pipeline.analyze_image(p)
    assert r.status=="OK"
    assert r.labels==["other"]
