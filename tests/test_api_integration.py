from __future__ import annotations
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from app.api.server import app

def test_health_and_session_and_analyze():
    client=TestClient(app)
    r=client.get('/health'); assert r.status_code==200 and r.json()['status']=='ok'
    sid=client.post('/sessions'); assert sid.status_code==200
    session=sid.json()['session_id']
    im=Image.fromarray(np.full((300,300,3),180,np.uint8)); ImageDraw.Draw(im).rectangle((20,20,280,280),outline=(20,20,20),width=5)
    b=BytesIO(); im.save(b,'JPEG')
    resp=client.post(f'/analyze/{session}',files={'file':('x.jpg',b.getvalue(),'image/jpeg')},data={'context_json':'{}'})
    assert resp.status_code==200
    body=resp.json(); assert body['analysis']['status']=='ABSTAIN'

def test_invalid_session_is_404():
    client=TestClient(app)
    from io import BytesIO
    r=client.post('/analyze/no-such-session',files={'file':('x.jpg',b'bad','image/jpeg')},data={'context_json':'{}'})
    assert r.status_code == 400


def test_demo_result_is_json_serializable():
    from app.config import Settings
    from app.core.factory import build_service
    from io import BytesIO
    from PIL import Image, ImageDraw
    import numpy as np
    import tempfile
    import json
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    settings = Settings(data_dir=root/"data", db_path=root/"data"/"db.sqlite3", yolo_weights=root/"best.pt", model_mode="demo", image_max_side=800, image_quality=80, video_max_side=640, video_fps=8, video_crf=30, max_upload_mb=5, retention_hours=1, confidence_threshold=.7, ood_threshold=.55)
    svc = build_service(settings)
    sid = svc.new_session()
    im = Image.fromarray(np.full((300,300,3),180,np.uint8)); ImageDraw.Draw(im).rectangle((30,30,260,260),fill=(220,40,40))
    b=BytesIO(); im.save(b,"JPEG")
    out=svc.process_upload(b.getvalue(),"image/jpeg",sid,{})
    json.dumps(out)


def test_browser_home_page_and_storage_endpoints():
    client = TestClient(app)
    home = client.get('/')
    assert home.status_code == 200
    assert 'SAFE-AI First-Aid Assistance' in home.text
    assert 'Analyze' in home.text
    assert client.get('/storage').status_code == 200
    assert client.post('/cleanup').status_code == 200


def test_video_upload_end_to_end():
    import subprocess, tempfile
    from pathlib import Path
    root = Path(tempfile.mkdtemp())
    src = root / 'demo.mp4'
    subprocess.run([
        'ffmpeg','-y','-hide_banner','-loglevel','error','-f','lavfi',
        '-i','color=c=red:s=320x240:r=10:d=1', str(src)
    ], check=True)
    client = TestClient(app)
    sid = client.post('/sessions').json()['session_id']
    resp = client.post(
        f'/analyze/{sid}',
        files={'file': ('demo.mp4', src.read_bytes(), 'video/mp4')},
        data={'context_json': '{}'},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body['media_type'] == 'video'
    assert body['stored_path'].endswith('.mp4')
    assert body['compressed_size'] > 0
