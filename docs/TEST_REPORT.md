# Validation Report

Date: 2026-09-24

## Final automated status

- Python compile check: **PASS**
- Full test suite: **27/27 PASS** after the final code changes
- Functional regression stress: **8/8 rounds PASS**, 25 tests per round
- Fault/crash-resilience stress: **8/8 rounds PASS**, 15 targeted tests per round on the final code
- Process startup + `/health`: **8/8 PASS** after the final launcher and endpoint changes, using separate ports and fresh data directories
- Real image end-to-end: **PASS**
- Real video compression + processing end-to-end: **PASS**
- Browser home page/API smoke test: **PASS**

## Fault cases exercised

- corrupt image
- corrupt video
- oversized upload
- unsupported MIME type
- missing FFmpeg for video conversion
- expired media cleanup
- temporary upload cleanup
- duplicate hash within a session
- duplicate hash across sessions isolation
- low quality/blurred image
- empty/no detector output
- missing YOLO weights
- detector exceptions
- unknown/other classification safety route
- low confidence safety route
- reported red-flag escalation
- RAG retrieval
- API JSON serialization
- legacy SQLite migration
- image analysis API
- video analysis API

## Real-process validation

The launcher was started repeatedly as a real subprocess. Each round verified:

1. process starts;
2. SQLite database is created;
3. `/health` becomes reachable;
4. the process can be terminated cleanly;
5. the next process can start on a fresh port.

## Environment notes

The current build environment does not have Streamlit installed, so the primary no-extra-UI dependency smoke path is the FastAPI browser application. Streamlit remains supported via `requirements-ui.txt`.

No injury-trained `best.pt` is bundled. Real medical-model inference therefore requires your project-trained YOLO weights and the optional `requirements-yolo.txt` dependency set.

The demo detector is synthetic and is for UI/test demonstration only; it is not a medical model.
