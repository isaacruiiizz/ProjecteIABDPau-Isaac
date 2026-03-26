# PJM-25 — sc-video-manager: trossejament de vídeo d'etiquetatge

**Ticket:** PJM-25
**Estat:** Completat ✓ (2026-03-25)
**Data:** 2026-03-24
**Etiqueta Jira:** `ai`

---

## Descripció

Implementar a `sc-video-manager` el suport per a `job_type: "process_labeling"`: descarregar un vídeo del bucket `labeling-videos`, extreure 1 frame cada `frame_interval` segons amb FFmpeg, i pujar cada frame al bucket `labeling-frames`. Cap escriptura a MongoDB, cap publicació a Redis.

---

## Fitxers afectats / nous

| Fitxer | Acció |
|---|---|
| `services/sc-video-manager/requirements.txt` | Actualitzar amb dependències reals |
| `services/sc-video-manager/.env.example` | Afegir `MINIO_BUCKET_LABELING_VIDEOS` i `MINIO_BUCKET_LABELING_FRAMES` |
| `services/sc-video-manager/app/config.py` | Nou — pydantic Settings |
| `services/sc-video-manager/app/main.py` | Nou — entrypoint: `setup_logging`, bucle `BLPOP`, dispatch per `job_type` |
| `services/sc-video-manager/app/services/__init__.py` | Nou — buit |
| `services/sc-video-manager/app/services/labeling_service.py` | Nou — lògica `process_labeling` |

> **Nota:** En aquest ticket només s'implementa `process_labeling`. El suport per a `process_match` (extracció de frames per a inferència) és un ticket independent.

---

## Fases d'implementació

### Fase 1 — Dependències i configuració

**1.1 `requirements.txt`**

```
redis==5.2.1
minio==7.2.15
sentry-sdk==2.24.1
pydantic-settings==2.8.2
```

**1.2 `.env.example`** — Afegir al final:

```env
# Buckets d'etiquetatge
MINIO_BUCKET_LABELING_VIDEOS=labeling-videos
MINIO_BUCKET_LABELING_FRAMES=labeling-frames
```

**1.3 `app/config.py`**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis
    REDIS_HOST: str = "sc-redis"
    REDIS_PORT: int = 6379
    REDIS_QUEUE_VIDEO: str = "video_to_process"
    REDIS_QUEUE_FRAMES: str = "task_frames"

    # MinIO
    MINIO_ENDPOINT: str = "sc-object-storage:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET_RAW: str = "raw-videos"
    MINIO_BUCKET_PENDING: str = "pending-frames"
    MINIO_BUCKET_PROCESSED_FRAMES: str = "processed-frames"
    MINIO_BUCKET_OUTPUT: str = "processed-videos"
    MINIO_BUCKET_LABELING_VIDEOS: str = "labeling-videos"
    MINIO_BUCKET_LABELING_FRAMES: str = "labeling-frames"

    # FFmpeg
    VIDEO_FPS_DEFAULT: int = 25

    # Sentry
    SENTRY_DSN: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
```

---

### Fase 2 — Lògica `process_labeling`

**`app/services/labeling_service.py`**

Passos que executa:

1. Rep el payload `process_labeling` (ja deserialitzat com a `dict`).
2. Descarrega el vídeo de MinIO (`labeling-videos / {minio_key}`) a un fitxer temporal (`/tmp/{session_id}.mp4`).
3. Usa `subprocess` per cridar FFmpeg:
   ```
   ffmpeg -i /tmp/{session_id}.mp4 \
          -vf fps=1/{frame_interval} \
          /tmp/{session_id}/frame_%06d.jpg
   ```
4. Per cada frame generat:
   - Puja a MinIO: bucket `labeling-frames`, clau `{session_id}/frame_{N:06d}.jpg`
5. Esborra els fitxers temporals.
6. Retorna el nombre de frames pujats (per al log).

Gestió d'errors:
- Qualsevol excepció → captura, `sentry_sdk.capture_exception` amb `session_id` al context, relança per ser gestionada al bucle principal que descarta i torna a escoltar.

---

### Fase 3 — Entrypoint i bucle principal

**`app/main.py`**

```
setup_logging("sc-video-manager", settings.SENTRY_DSN)
Connecta Redis i MinIO
BLPOP bucle infinit:
  Rep missatge de video_to_process
  Parseja JSON
  Llegeix job_type
  "process_labeling" → labeling_service.process(payload)
  Qualsevol altra cosa / camp absent → log WARNING + descarta
  Error no controlat → log ERROR + Sentry + descarta (mai reencua)
```

---

## Payload esperat (referència)

```json
{
  "job_type": "process_labeling",
  "session_id": "b7e2f1a0-...",
  "minio_bucket": "labeling-videos",
  "minio_key": "b7e2f1a0-.../original.mp4",
  "frame_interval": 2
}
```

Claus de sortida: `{session_id}/frame_000001.jpg` (zero-padded 6 dígits)

---

## Regles crítiques (recordatori)

- `sc-video-manager` és **worker pur** — sense HTTP, sense ports exposats.
- Per a `process_labeling`: **no escriure res a MongoDB**, **no publicar res a Redis**.
- Error → Sentry amb `session_id`, descartar missatge, tornar a `BLPOP`.
- **Mai reencuar** un missatge erroni.

---

### Fase 4 — Verificació

1. `docker compose build sc-video-manager` → sense errors de build.
2. `docker compose up sc-video-manager` → el worker arrenca i queda bloquejat a `BLPOP` (log INFO visible).
3. Publicar missatge de prova via `redis-cli`:
   ```bash
   redis-cli RPUSH video_to_process '{"job_type":"process_labeling","session_id":"test-001","minio_bucket":"labeling-videos","minio_key":"test-001/original.mp4","frame_interval":2}'
   ```
4. Comprovar que els frames apareixen a MinIO sota `labeling-frames/test-001/frame_000001.jpg`, etc.
5. Comprovar que **no** hi ha cap missatge publicat a `task_frames` (`redis-cli LLEN task_frames` → 0).
6. Comprovar logs JSON a stdout (format `{"time":..., "service":"sc-video-manager", ...}`).
