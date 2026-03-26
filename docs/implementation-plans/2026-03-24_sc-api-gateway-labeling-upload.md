# Pla d'implementació: Endpoint upload de vídeo per a etiquetatge

**Ticket:** PJM-24
**Data:** 2026-03-24
**Estat:** Completat ✓ (2026-03-24)

---

## Objectiu

Implementar `POST /api/v1/labeling/upload`:
1. Rep el fitxer `.mp4` per multipart
2. Genera un `session_id` (UUID)
3. Puja el vídeo a MinIO → bucket `labeling-videos`, clau `{session_id}/original.mp4`
4. Publica missatge a Redis `video_to_process` via `RPUSH`
5. Retorna `session_id`, `minio_key` i `status`

Accessible **únicament per rol `admin`**.

---

## Context (de les specs)

**Bucket i clau** (spec 03-infraestructura.md §5):
- Bucket: `labeling-videos`
- Clau: `{session_id}/original.mp4`
- Escriu: `sc-api-gateway` (admin)
- Llegeix: `sc-video-manager`

**Payload Redis exacte** (spec 05-config.md §3):
```json
{
  "job_type": "process_labeling",
  "session_id": "b7e2f1a0-...",
  "minio_bucket": "labeling-videos",
  "minio_key": "b7e2f1a0-.../original.mp4",
  "frame_interval": 2
}
```
- Cua: `video_to_process`
- Mètode: `RPUSH` (mai Pub/Sub)
- `frame_interval`: paràmetre de la petició (defecte: `2`)

**Dependències ja a `requirements.txt`:**
- `redis>=5.2.0` — client Redis (no inicialitzat al gateway encara)
- `boto3>=1.37.0` — client S3/MinIO
- `python-multipart>=0.0.20` — suport `UploadFile` FastAPI

**Codi existent rellevant:**
- `main.py`: lifespan gestiona Motor clients. Caldrà afegir Redis + boto3.
- `dependencies.py`: patró `client | None = None` + getter. Seguirem el mateix.
- `config.py`: `REDIS_HOST`, `REDIS_PORT`, `MINIO_*` ja existeixen.

---

## Fitxers afectats

| Fitxer | Acció | Contingut |
|--------|-------|-----------|
| `app/schemas/labeling.py` | **Crear** | `LabelingUploadResponse` |
| `app/services/labeling_service.py` | **Crear** | `upload_labeling_video()` |
| `app/routers/labeling.py` | **Crear** | `POST /api/v1/labeling/upload` |
| `app/dependencies.py` | **Modificar** | + `redis_client`, `get_redis()`, `s3_client`, `get_s3()` |
| `app/main.py` | **Modificar** | + init/close Redis+S3 al lifespan, + `include_router(labeling)` |

No cal cap repository nou (spec: "per a feines `process_labeling` no s'escriu res a MongoDB").

---

## Fase 1 — `schemas/labeling.py` (nou)

```python
from pydantic import BaseModel


class LabelingUploadResponse(BaseModel):
    session_id: str
    minio_key: str
    status: str = "queued"
```

---

## Fase 2 — `dependencies.py` (modificar)

Afegir al final, mantenint tot el codi existent:

```python
import boto3
import redis.asyncio as aioredis

# Redis async client
redis_client: aioredis.Redis | None = None

def get_redis() -> aioredis.Redis:
    """Retorna el client Redis async. Usat via Depends()."""
    return redis_client

# boto3 S3 client (boto3 és sync; s'usa amb asyncio.to_thread() al service)
s3_client = None  # boto3.client("s3", ...)

def get_s3():
    """Retorna el client S3 (boto3). Usat via Depends()."""
    return s3_client
```

---

## Fase 3 — `main.py` (modificar)

Dues modificacions:

**3a. Lifespan** — inicialitzar/tancar Redis i S3:
```python
import boto3
import redis.asyncio as aioredis

@asynccontextmanager
async def lifespan(app: FastAPI):
    # MongoDB (ja existent)
    deps.auth_client = AsyncIOMotorClient(settings.MONGO_AUTH_URI)
    deps.app_client  = AsyncIOMotorClient(settings.MONGO_APP_URI)
    # Redis
    deps.redis_client = aioredis.Redis(
        host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=False
    )
    # MinIO via boto3 S3
    deps.s3_client = boto3.client(
        "s3",
        endpoint_url=f"http{'s' if settings.MINIO_USE_SSL else ''}://{settings.MINIO_ENDPOINT}",
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
    )
    yield
    deps.auth_client.close()
    deps.app_client.close()
    await deps.redis_client.aclose()
```

**3b. Router** — registrar el router de labeling:
```python
from app.routers import auth, health, labeling

app.include_router(labeling.router)
```

---

## Fase 4 — `services/labeling_service.py` (nou)

Lògica de negoci: upload a MinIO + RPUSH a Redis.
boto3 és síncron → `asyncio.to_thread()` per no bloquejar l'event loop.

```python
import asyncio
import json
import logging
from uuid import uuid4

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

BUCKET = "labeling-videos"
QUEUE  = "video_to_process"


async def upload_labeling_video(
    file_bytes: bytes,
    frame_interval: int,
    s3,
    redis: aioredis.Redis,
) -> dict:
    session_id = str(uuid4())
    minio_key  = f"{session_id}/original.mp4"

    # Upload a MinIO (boto3 sync → thread)
    await asyncio.to_thread(
        s3.put_object,
        Bucket=BUCKET,
        Key=minio_key,
        Body=file_bytes,
        ContentType="video/mp4",
    )
    logger.info("Vídeo pujat a MinIO: bucket=%s key=%s", BUCKET, minio_key)

    # Publicar a Redis (RPUSH)
    payload = json.dumps({
        "job_type":      "process_labeling",
        "session_id":    session_id,
        "minio_bucket":  BUCKET,
        "minio_key":     minio_key,
        "frame_interval": frame_interval,
    })
    await redis.rpush(QUEUE, payload)
    logger.info("Missatge publicat a Redis: queue=%s session_id=%s", QUEUE, session_id)

    return {"session_id": session_id, "minio_key": minio_key}
```

---

## Fase 5 — `routers/labeling.py` (nou)

```python
import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.dependencies import get_current_user, get_redis, get_s3, require_roles
from app.schemas.labeling import LabelingUploadResponse
from app.services import labeling_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/labeling", tags=["labeling"])


@router.post("/upload", response_model=LabelingUploadResponse, status_code=202)
async def upload_labeling_video(
    video: UploadFile = File(..., description="Fitxer .mp4 per a etiquetatge"),
    frame_interval: int = Query(default=2, ge=1, le=60),
    _user=Depends(require_roles("admin")),
    redis=Depends(get_redis),
    s3=Depends(get_s3),
):
    file_bytes = await video.read()
    result = await labeling_service.upload_labeling_video(
        file_bytes=file_bytes,
        frame_interval=frame_interval,
        s3=s3,
        redis=redis,
    )
    return LabelingUploadResponse(**result)
```

**Notes de disseny:**
- `status_code=202 Accepted` — el vídeo s'ha rebut i encuat, però el trossejament és asíncron.
- `frame_interval` com a `Query` param (1–60s). Valor per defecte: 2 (de l'exemple de l'spec).
- `_user` s'ignora (nom amb `_`): només serveix per fer complir el rol `admin`.
- `file_bytes = await video.read()` llegeix tot el fitxer en memòria. Acceptable per a ús admin puntual; no és un endpoint d'ús massiu.

---

## Endpoint documentat

| Mètode | Path | Auth | Cos | Resposta |
|--------|------|------|-----|----------|
| `POST` | `/api/v1/labeling/upload` | `admin` | `multipart/form-data`: `video` (file) + `frame_interval` (query, default 2) | `202 {"session_id": "...", "minio_key": "...", "status": "queued"}` |

---

## Verificació post-implementació

- [ ] `POST /api/v1/labeling/upload` amb rol `admin` → `202` + `session_id`
- [ ] Mateix endpoint amb rol `coach` → `403 Forbidden`
- [ ] Sense token → `401 Unauthorized`
- [ ] Fitxer pujat a MinIO bucket `labeling-videos` amb clau `{session_id}/original.mp4`
- [ ] Missatge a Redis `video_to_process` amb tots els camps obligatoris
- [ ] `frame_interval=0` → `422 Unprocessable Entity` (validació `ge=1`)
