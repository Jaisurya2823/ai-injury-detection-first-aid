# API Contract

Base URL: `http://127.0.0.1:8000`

## GET `/`

Browser UI for local demo/testing.

## GET `/health`

Returns application, detector and FFmpeg state.

## POST `/sessions`

Returns:

```json
{"session_id":"<uuid>"}
```

## POST `/analyze/{session_id}`

Multipart fields:

- `file`: image/video upload
- `context_json`: JSON object containing optional context fields

Accepted context keys:

`mechanism`, `time_since_injury`, `bleeding_status`, `pain_level`, `movement_limitation`, `red_flags`

## GET `/history/{session_id}`

Returns recent analysis records for that session.

## GET `/storage`

Returns stored file count, original bytes, compressed bytes, saved bytes and compression ratio.

## POST `/cleanup`

Removes expired media and leftover temporary files.

## Errors

- `400`: invalid upload, invalid context, unknown session, unsupported type, decode/compression error
- `404`: unknown history session
- `500`: unexpected server-side processing error
