from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.config import Settings
from app.core.factory import build_service

app = FastAPI(title="SAFE-AI First-Aid Assistance API", version="1.1.0")
settings = Settings.from_env()
service = build_service(settings)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SAFE-AI First Aid</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1050px;margin:0 auto;padding:28px;background:#f6f8fb;color:#18212f}
.card{background:#fff;border:1px solid #d9e0ea;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 4px 18px rgba(20,40,70,.05)}
h1{margin-bottom:4px}.muted{color:#667085}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}label{display:block;font-weight:600;margin:10px 0 6px}input,select{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:9px;box-sizing:border-box}button{margin-top:16px;padding:11px 18px;border:0;border-radius:9px;background:#1f5eff;color:white;font-weight:700;cursor:pointer}.warn{background:#fff6d6;border:1px solid #edd477;padding:12px;border-radius:10px}.result{white-space:pre-wrap;background:#f8fafc;border-radius:10px;padding:14px;font-family:ui-monospace,monospace;overflow:auto}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head>
<body><h1>🩹 SAFE-AI First-Aid Assistance</h1>
<div class='muted'>Preliminary visual screening only — not a medical diagnosis.</div>
<div class='card'><h2>1. Create session</h2><button onclick='newSession()'>New session</button><div id='sid' class='result'>No session</div></div>
<div class='card'><h2>2. Analyze image/video</h2>
<label>Session ID</label><input id='session' required>
<label>Image / video</label><input id='file' type='file' accept='image/*,video/*' required>
<div class='grid'>
<div><label>Mechanism</label><select id='mechanism'><option>unknown</option><option>fall</option><option>cut/contact</option><option>heat</option><option>impact</option><option>other</option></select></div>
<div><label>Bleeding status</label><select id='bleeding'><option>unknown</option><option>none</option><option>minor</option><option>persistent/heavy</option></select></div>
<div><label>Pain</label><select id='pain'><option>unknown</option><option>mild</option><option>moderate</option><option>severe</option></select></div>
<div><label>Movement limitation</label><select id='movement'><option>unknown</option><option>none</option><option>some</option><option>major</option></select></div></div>
<label><input type='checkbox' id='heavy_bleeding'> Reported heavy/continuous bleeding</label>
<button onclick='analyze()'>Analyze</button><div id='out' class='result'></div></div>
<div class='card'><h2>3. Storage</h2><button onclick='stats()'>Refresh storage</button><div id='stats' class='result'></div></div>
<script>
async function newSession(){const r=await fetch('/sessions',{method:'POST'});const x=await r.json();document.getElementById('sid').textContent=x.session_id;document.getElementById('session').value=x.session_id}
async function stats(){const r=await fetch('/storage');document.getElementById('stats').textContent=JSON.stringify(await r.json(),null,2)}
async function analyze(){const s=document.getElementById('session').value;const f=document.getElementById('file').files[0];if(!s||!f){alert('Session ID and file are required');return}const ctx={mechanism:document.getElementById('mechanism').value,bleeding_status:document.getElementById('bleeding').value,pain_level:document.getElementById('pain').value,movement_limitation:document.getElementById('movement').value,red_flags:document.getElementById('heavy_bleeding').checked?['heavy_bleeding']:[]};const fd=new FormData();fd.append('file',f);fd.append('context_json',JSON.stringify(ctx));const r=await fetch('/analyze/'+encodeURIComponent(s),{method:'POST',body:fd});const x=await r.json();document.getElementById('out').textContent=JSON.stringify(x,null,2)}
</script></body></html>"""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_mode": settings.model_mode,
        "detector": getattr(service.pipeline.detector, "name", service.pipeline.detector.__class__.__name__),
        "storage": "sqlite3+compressed-media",
        "ffmpeg_required_for_video": True,
        "ffmpeg_available": __import__("shutil").which("ffmpeg") is not None,
    }


@app.post("/sessions")
def create_session():
    return {"session_id": service.new_session()}


@app.get("/history/{session_id}")
def history(session_id: str, limit: int = 20):
    try:
        return {"items": service.history(session_id, limit)}
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/storage")
def storage():
    return service.storage_stats()


@app.post("/cleanup")
def cleanup():
    return {"removed": service.cleanup()}


@app.post("/analyze/{session_id}")
async def analyze(session_id: str, file: UploadFile = File(...), context_json: str = Form("{}")):
    if not file.content_type:
        raise HTTPException(400, "Missing content type")
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as e:
        raise HTTPException(400, "Invalid context_json") from e
    if not isinstance(context, dict):
        raise HTTPException(400, "context_json must be a JSON object")
    try:
        # Pass the UploadFile stream directly so large videos are not duplicated in RAM.
        return service.process_upload(file.file, file.content_type, session_id, context)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, "Processing failed") from e
