# PJM-67 — Endpoint POST /matches/{id}/process

**Estat:** Completat ✓ (2026-06-08)
**Ticket:** PJM-67
**Data:** 2026-06-08

---

## Flux implementat

```
POST /api/v1/matches/{match_id}/process
  ↓
match_service.process_match()
  1. get_match_by_id() → 404 si no existeix
  2. Comprova status != "processing"|"done" → 409 si ja en curs
  3. Comprova video_raw != None → 409 si no hi ha vídeo
  4. update_match_status("processing")
  5. rpush("video_to_process", payload)
  ↓
202 Accepted {"match_id": "...", "status": "processing"}
```

## Payload Redis publicat

```json
{
  "job_type":      "process_match",
  "match_id":      "6a0f28430b76aee263b55b6e",
  "minio_bucket":  "raw-videos",
  "minio_key":     "6a0f28430b76aee263b55b6e/original.mp4",
  "roi_polygon":   [{"x": 100, "y": 50}],
  "start_seconds": 0.0,
  "end_seconds":   120.0
}
```

## Fitxers modificats

| Fitxer | Canvi |
|---|---|
| `app/repositories/match_repository.py` | +`get_match_by_id`, +`update_match_status` |
| `app/services/match_service.py` | +`process_match`, +`QUEUE_VIDEO`, +`import json` |
| `app/schemas/matches.py` | +`ProcessMatchResponse` |
| `app/routers/matches.py` | +`POST /{match_id}/process`, +imports |

## Codis HTTP

| Cas | Codi |
|---|---|
| Èxit | 202 Accepted |
| Partit no trobat | 404 Not Found |
| Ja processing/done | 409 Conflict |
| Sense vídeo pujat | 409 Conflict |
| Error intern | 500 Internal Server Error |
