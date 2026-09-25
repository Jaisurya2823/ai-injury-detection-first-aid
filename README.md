# SAFE-AI First-Aid Assistance System

A local-first engineering prototype for **preliminary visual screening of common visible injuries** with explicit abstention, safety routing, evidence retrieval, compressed media storage, and SQLite3 metadata.

> **Safety:** This project is not a medical diagnosis system and is not a substitute for professional medical evaluation.

## 1. Architecture

`Input -> Quality Gate -> Human/Body Localization -> YOLO -> Segmentation -> Feature Extraction -> Multi-label Classification -> Confidence/OOD/Abstention -> Context -> Risk/Severity -> Red-Flag Safety Engine -> Evidence-Based Guidance/RAG -> Explanation -> Web/API`

Data plane:

`SQLite3 metadata + compressed media files (WebP/JPEG images, MP4/H.264 videos) + SHA-256 dedup + retention cleanup`

Docker and Kubernetes are intentionally not used.

## 2. Run immediately

The **FastAPI browser UI does not require Streamlit** and is the easiest smoke-test path:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open `http://127.0.0.1:8000/`.

Health: `http://127.0.0.1:8000/health`

API docs: `http://127.0.0.1:8000/docs`

## 3. Roboflow Workflow mode (recommended — no GPU, no training)

The default `.env.example` uses `FIRST_AID_MODEL_MODE=workflow`, which calls the
hosted Roboflow Workflow running **YOLO26m** on Roboflow's cloud.

**Workflow:** `firstaid_4class 2 vfirstaid_4class-2-v45sb-1-yolo26m-t1 Logic`
**Workspace:** `jaisurya-i4vpr`

No GPU, no `best.pt`, no Ultralytics required — only your Roboflow API key.

```bash
pip install inference-sdk
```

Then create your `.env` and fill in your Roboflow API key:

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` in any text editor and set:

```text
FIRST_AID_MODEL_MODE=workflow
ROBOFLOW_API_KEY=your_key_here   # app.roboflow.com → Settings → API Keys
```

The app reads `.env` automatically on startup — you do **not** need to export
variables in your shell or PowerShell session.

The detector sends each image to the Roboflow endpoint via `InferenceHTTPClient.run_workflow()`,
parses the prediction list defensively (key-agnostic), maps dataset class names such as
`cut_open_wound` and `burn_mild` to the system's `VALID_CLASSES`, and returns `Detection`
objects — the same type every other detector returns.

Auth is sent in the `Authorization: Bearer` header (inference-sdk ≥ 1.5), never in
the query string. Transient failures are retried up to 3 times with exponential back-off.
If the API key is missing or `inference-sdk` is not installed, the system falls back
to `NullDetector` and logs a clear error — it never crashes.

## 4. Demo mode vs YOLO weights

`FIRST_AID_MODEL_MODE=demo` detects a **synthetic red marker** only. It is clearly labeled
as demo behavior and must never be presented as a trained medical model.

For fully local inference with your own trained weights:

```bash
pip install ultralytics
```

```text
FIRST_AID_MODEL_MODE=yolo
FIRST_AID_YOLO_WEIGHTS=models/best.pt
```

Without valid weights the system falls back to `null` mode and abstains rather than
pretending a generic detector is an injury model.

## 5. Streamlit UI

Once Streamlit is installed:

```bash
# Linux / macOS
streamlit run app/streamlit_app.py

# Windows  (avoids ModuleNotFoundError: No module named 'app')
python run_streamlit.py
```

The Streamlit UI supports file upload, camera capture, context questions, results, evidence, technical details, history, and storage statistics.

## 5. Storage lifecycle

1. Upload arrives in a temporary file.
2. Size and type are validated.
3. SHA-256 is calculated.
4. Images are resized and compressed to WebP.
5. Videos are transcoded to MP4/H.264 using FFmpeg.
6. Only the compressed media path and metadata are persisted.
7. Temporary uploads are deleted in a `finally` block.
8. Expired files can be removed with the cleanup endpoint or service.

Large raw videos are never inserted into SQLite.

## 6. API

`POST /sessions` creates a session.

`POST /analyze/{session_id}` accepts multipart media and an optional `context_json` form field.

`GET /history/{session_id}` returns recent analysis history.

`GET /storage` returns storage statistics.

`POST /cleanup` removes expired and temporary media.

## 7. Tests

```bash
pytest -q
```

A separate stress runner can repeatedly execute functional and fault-resilience suites:

```bash
python scripts/run_stress_tests.py
```

## 8. Project layout

```text
app/
  ai/         # quality, detection, segmentation, features, reliability, risk
  api/        # FastAPI server and schemas
  core/       # service orchestration and factory
  db/         # SQLite3 schema and repository operations
  media/      # upload, compression, deduplication, cleanup
  rag/        # local retrieval and guidance
  safety/     # deterministic safety routing
  streamlit_app.py
knowledge/    # local source metadata/content for RAG
models/       # place trained weights here (not bundled)
data/        # runtime database and compressed media
scripts/      # health and stress tools
tests/        # automated unit/integration/fault tests
docs/         # architecture, API and test documentation
```
