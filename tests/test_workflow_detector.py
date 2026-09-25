"""
Tests for RoboflowWorkflowDetector.

Unit tests run offline (no network, no API key needed).
The optional smoke test at the bottom requires ROBOFLOW_API_KEY to be set.
"""
from __future__ import annotations

import os
import types
import numpy as np
import pytest

from app.ai.workflow_detector import (
    _extract_predictions,
    _parse_detections,
    RoboflowWorkflowDetector,
    _WORKSPACE,
    _WORKFLOW_ID,
)
from app.ai.detector import DetectorError, VALID_CLASSES
from app.ai.types import Detection


# ── _extract_predictions ──────────────────────────────────────────────────────

def test_extract_direct_predictions_key():
    output = {"predictions": [{"confidence": 0.9, "class": "cut"}], "image": {}}
    assert len(_extract_predictions(output)) == 1


def test_extract_nested_predictions_key():
    output = {"detections": {"predictions": [{"confidence": 0.7}]}}
    assert len(_extract_predictions(output)) == 1


def test_extract_list_of_confidence_dicts():
    output = {"results": [{"confidence": 0.8, "class": "bruise"}]}
    assert len(_extract_predictions(output)) == 1


def test_extract_empty_output():
    assert _extract_predictions({}) == []
    assert _extract_predictions({"image": {"width": 640}}) == []


# ── _parse_detections ─────────────────────────────────────────────────────────

def _make_pred(label, conf, x=100, y=100, w=50, h=50):
    return {"class": label, "confidence": conf, "x": x, "y": y, "width": w, "height": h}


def test_parse_maps_cut_open_wound():
    preds = [_make_pred("cut_open_wound", 0.85)]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    assert len(dets) == 1
    assert dets[0].label == "cut"


def test_parse_maps_burn_mild():
    preds = [_make_pred("burn_mild", 0.80)]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    assert dets[0].label == "burn"


def test_parse_unknown_class_becomes_other():
    preds = [_make_pred("mystery_injury", 0.75)]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    assert dets[0].label == "other"


def test_parse_all_labels_in_valid_classes():
    labels = ["cut_open_wound", "burn_mild", "bruise", "abrasion", "swelling"]
    preds = [_make_pred(l, 0.9) for l in labels]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    assert all(d.label in VALID_CLASSES for d in dets)


def test_parse_filters_low_confidence():
    preds = [_make_pred("bruise", 0.30), _make_pred("cut", 0.80)]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    assert len(dets) == 1
    assert dets[0].label == "cut"


def test_parse_bbox_clamped_to_image():
    # x=10, y=10, w=2000 → x2 should clamp to img_w
    preds = [_make_pred("bruise", 0.9, x=10, y=10, w=2000, h=2000)]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    x1, y1, x2, y2 = dets[0].bbox
    assert x2 <= 640
    assert y2 <= 480


def test_parse_malformed_pred_skipped():
    preds = [{"confidence": 0.9}, _make_pred("bruise", 0.9)]
    dets = _parse_detections(preds, 640, 480, min_confidence=0.5)
    assert len(dets) == 1   # malformed entry skipped, good one kept


def test_parse_empty():
    assert _parse_detections([], 640, 480, 0.5) == []


# ── RoboflowWorkflowDetector init ─────────────────────────────────────────────

def test_detector_raises_without_api_key():
    with pytest.raises(DetectorError, match="ROBOFLOW_API_KEY"):
        RoboflowWorkflowDetector(api_key="", confidence=0.5)


def test_detector_raises_without_inference_sdk(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "inference_sdk":
            raise ImportError("mocked missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(DetectorError, match="inference-sdk"):
        RoboflowWorkflowDetector(api_key="fake-key")


# ── RoboflowWorkflowDetector.predict (mocked) ─────────────────────────────────

class _FakeClient:
    """Minimal mock of InferenceHTTPClient for unit tests."""
    def configure(self, cfg): pass
    def run_workflow(self, workspace_name, workflow_id, images):
        assert workspace_name == _WORKSPACE
        assert workflow_id == _WORKFLOW_ID
        return [{
            "predictions": [
                {"class": "bruise", "confidence": 0.88,
                 "x": 200, "y": 150, "width": 100, "height": 80,
                 "detection_id": "abc123", "parent_id": "image.[0]"},
                {"class": "cut_open_wound", "confidence": 0.30,   # below threshold
                 "x": 50, "y": 50, "width": 30, "height": 30,
                 "detection_id": "def456", "parent_id": "image.[0]"},
            ]
        }]


def _make_detector_with_mock(monkeypatch):
    import importlib
    import sys

    # Stub out inference_sdk so the import succeeds without the real package
    fake_sdk = types.ModuleType("inference_sdk")
    fake_sdk.InferenceHTTPClient = lambda api_url, api_key: _FakeClient()
    fake_sdk.InferenceConfiguration = lambda **kw: None
    sys.modules["inference_sdk"] = fake_sdk

    from app.ai import workflow_detector as wd
    importlib.reload(wd)

    det = wd.RoboflowWorkflowDetector(api_key="fake-key", confidence=0.45)
    # Replace the real client with our mock (already the same class, but be explicit)
    det._client = _FakeClient()
    return det


def test_predict_returns_correct_detections(monkeypatch):
    det = _make_detector_with_mock(monkeypatch)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.predict(img)
    assert len(dets) == 1                  # low-conf cut filtered out
    assert dets[0].label == "bruise"
    assert abs(dets[0].confidence - 0.88) < 1e-6
    x1, y1, x2, y2 = dets[0].bbox
    assert x1 == 150 and y1 == 110        # centre-xywh → xyxy
    assert x2 == 250 and y2 == 190


def test_predict_empty_response(monkeypatch):
    det = _make_detector_with_mock(monkeypatch)
    det._client = type("C", (), {
        "configure": lambda s, c: None,
        "run_workflow": lambda s, **kw: [],
    })()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    assert det.predict(img) == []


# ── Factory wiring ─────────────────────────────────────────────────────────────

def test_factory_workflow_mode_missing_key_falls_back_to_null(tmp_path):
    """workflow mode with no key must fall back to NullDetector, never crash."""
    import os
    from app.config import Settings
    from app.core.factory import build_service

    env_backup = os.environ.pop("ROBOFLOW_API_KEY", None)
    try:
        s = Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "data" / "db.sqlite3",
            yolo_weights=tmp_path / "missing.pt",
            model_mode="workflow",
            image_max_side=800, image_quality=80, video_max_side=640,
            video_fps=8, video_crf=30, max_upload_mb=5,
            retention_hours=1, confidence_threshold=0.45, ood_threshold=0.40,
        )
        svc = build_service(s)
        assert svc.pipeline.detector.__class__.__name__ == "NullDetector"
    finally:
        if env_backup is not None:
            os.environ["ROBOFLOW_API_KEY"] = env_backup


# ── Optional live smoke test ───────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("ROBOFLOW_API_KEY"),
    reason="ROBOFLOW_API_KEY not set — skipping live smoke test"
)
def test_smoke_live_workflow():
    """
    Runs the real workflow against a small synthetic image.
    Asserts the expected output structure exists.
    Set ROBOFLOW_API_KEY in your environment to run this test.
    """
    from app.ai.workflow_detector import RoboflowWorkflowDetector

    det = RoboflowWorkflowDetector(
        api_key=os.environ["ROBOFLOW_API_KEY"],
        confidence=0.10,   # low threshold so any detection comes through
    )
    # 640×480 blank image — expect [] (no injury) but no crash
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = det.predict(img)
    assert isinstance(dets, list)
    assert all(isinstance(d, Detection) for d in dets)
    assert all(d.label in VALID_CLASSES for d in dets)
