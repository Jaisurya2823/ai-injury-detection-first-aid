# SAFE-AI First-Aid Assistance — System Architecture

## 1. Scope

This is a local-first engineering prototype for **preliminary visual screening of common visible injuries**. It is not a diagnostic system and does not replace professional medical evaluation.

## 2. Control flow

```text
User
  |
  +--> Image / Camera / Video
              |
              v
      Input validation
              |
              v
      Image/video quality gate
              |
         +----+----+
         |         |
       FAIL       PASS
         |         |
  request input    v
             Human/body localization
                     |
                     v
              YOLO detection
                     |
                     v
              Segmentation
                     |
                     v
          Visual feature extraction
                     |
                     v
            Multi-label classifier
                     |
                     v
          Confidence/OOD/abstention
                |           |
           UNCERTAIN     RELIABLE
                |           |
        request better      v
        input / escalate  Context collection
                              |
                              v
                      Risk/severity engine
                              |
                              v
                       Red-flag engine
                              |
                              v
                       Safety rule engine
                              |
                              v
                         RAG retrieval
                              |
                              v
                     Evidence-based guidance
                              |
                              v
                       Explanation layer
                              |
                         +----+----+
                         |    |   |
                        Web Mobile API
```

## 3. Model boundary

- `RoboflowWorkflowDetector` (`mode=workflow`) — **default, recommended**. Calls the
  hosted Roboflow Workflow `firstaid4class-2-vfirstaid4class-2-v45sb-1-yolo26m-t1-logic`
  in workspace `jaisurya-i4vpr` via `inference-sdk`. Runs YOLO26m on Roboflow's cloud.
  No GPU, no weights file, no Ultralytics required. Requires `ROBOFLOW_API_KEY`.
  Auth uses `Authorization: Bearer` header. Retries with exponential back-off.
  Falls back to `NullDetector` on init failure.
- `RoboflowDetector` (`mode=roboflow`) — calls a single hosted model via the inference
  API (older approach, requires `ROBOFLOW_MODEL_ID` in addition to `ROBOFLOW_API_KEY`).
- `YOLODetector` (`mode=yolo`) — local `best.pt` weights; requires Ultralytics install.
- `NullDetector` — safe fallback; returns no detections.
- `DemoDetector` — strictly synthetic UI/test behaviour; must not be presented as medical inference.
- The segmentation component is a robust ROI-based baseline. A trained segmentation model
  can replace it without changing the pipeline contract.

## 4. Reliability boundary

The reliability layer can abstain when:

- no supported injury is detected;
- confidence is below threshold;
- the OOD proxy is high;
- detector outputs disagree.

The system never needs to force a classification when evidence is insufficient.

## 5. Safety boundary

The deterministic safety engine owns:

- red-flag escalation;
- low-confidence/insufficient-evidence handling;
- `unknown`/`other` routing;
- preliminary severity routing;
- explicit non-diagnostic warnings.

RAG supplies supporting evidence and source metadata. It is **not** the authority for deciding whether a user needs emergency care.

## 6. Storage architecture

```text
SQLite3
  |
  +-- users
  +-- sessions
  +-- media metadata
  +-- analysis results
  +-- context answers
  +-- guidance records
  +-- knowledge document metadata

Compressed media directory
  |
  +-- images/*.webp
  +-- videos/*.mp4
  +-- tmp/*
```

Raw uploads are temporary only. Images are normalized to WebP. Videos are transcoded to MP4/H.264 with FFmpeg. SHA-256 prevents duplicate storage within the same session. Expiration and cleanup prevent indefinite media growth.

## 7. Deployment

Docker and Kubernetes are intentionally excluded.

### Local/server deployment

```text
Python + FastAPI + optional Streamlit
       |
       +--> SQLite3
       |
       +--> compressed local media
```

The launcher supports `FIRST_AID_HOST` and `FIRST_AID_PORT` so multiple isolated test instances can run safely.

## 8. Upgrade path

The application can later replace individual pieces without redesigning the whole workflow:

- SQLite3 -> PostgreSQL
- local media -> object storage
- local inference -> GPU service
- ROI segmentation -> trained segmentation network
- TF-IDF retriever -> vector database/embedding retriever
- Streamlit/FastAPI -> mobile client/API gateway
