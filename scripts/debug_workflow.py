#!/usr/bin/env python3
"""
Debug script — dumps the EXACT raw type and structure returned by run_workflow().
Run this first to see what we're actually getting.

Usage:
    cd project_out
    set ROBOFLOW_API_KEY=your_key_here
    python scripts/debug_workflow.py path\to\image.jpg
"""
from __future__ import annotations
import sys, os, json, base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import _load_dotenv
_load_dotenv()

import cv2, numpy as np

API_KEY     = os.getenv("ROBOFLOW_API_KEY", "")
WORKSPACE   = "jaisurya-i4vpr"
WORKFLOW_ID = "firstaid4class-2-vfirstaid4class-2-v45sb-1-yolo26m-t1-logic"
API_URL     = "https://serverless.roboflow.com"

if not API_KEY:
    print("ERROR: ROBOFLOW_API_KEY not set in .env"); sys.exit(1)

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if image_path:
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot read {image_path}"); sys.exit(1)
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
else:
    print("No image given — using synthetic test image")
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:380, 100:540] = (60, 80, 200)

ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

from inference_sdk import InferenceHTTPClient, InferenceConfiguration
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
client.configure(InferenceConfiguration(api_key_transport="header"))

print(f"\nCalling {WORKSPACE}/{WORKFLOW_ID} ...")
results = client.run_workflow(
    workspace_name=WORKSPACE,
    workflow_id=WORKFLOW_ID,
    images={"image": b64},
)

print(f"\n=== RESULT TYPE: {type(results)} ===")
print(f"Length: {len(results) if isinstance(results, list) else 'n/a'}")

if isinstance(results, list) and results:
    first = results[0]
    print(f"\n=== results[0] TYPE: {type(first)} ===")

    if isinstance(first, dict):
        for k, v in first.items():
            print(f"\n  KEY '{k}' -> type={type(v).__name__}")
            if hasattr(v, '__dict__'):
                print(f"    __dict__ keys: {list(v.__dict__.keys())}")
                for attr, val in v.__dict__.items():
                    print(f"      .{attr} = {type(val).__name__} = {repr(val)[:120]}")
            elif isinstance(v, list):
                print(f"    list of {len(v)} items")
                if v:
                    print(f"    [0] type={type(v[0]).__name__} = {repr(v[0])[:200]}")
                    if hasattr(v[0], '__dict__'):
                        print(f"    [0].__dict__ = {v[0].__dict__}")
            else:
                print(f"    value = {repr(v)[:200]}")
    else:
        print(f"results[0] is not a dict: {repr(first)[:500]}")

print("\n=== RAW JSON ATTEMPT ===")
try:
    print(json.dumps(results, indent=2, default=str)[:3000])
except Exception as e:
    print(f"Cannot JSON-serialize: {e}")
    print(repr(results)[:1000])
