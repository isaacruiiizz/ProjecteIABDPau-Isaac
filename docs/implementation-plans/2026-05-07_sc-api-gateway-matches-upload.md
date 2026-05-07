# PJM-28 — POST /api/v1/matches: upload vídeo i metadata

**Data:** 2026-05-07  
**Estat:** En Procés  
**Sprint:** 3 — Pipeline IA MVP  
**Ticket:** PJM-28  
**Etiqueta:** `backend`  

---

## Objectiu

Endpoint mínim: rep un vídeo MP4 + títol, el puja a MinIO `raw-videos` i crea el document de partit a MongoDB assignat a l'usuari que l'ha pujat. Retorna el `match_id`.

**Simplificació deliberada:** sense equips, sense `team_id`, sense rols específics — qualsevol usuari autenticat pot pujar un vídeo.

---

## Fitxers afectats

| Fitxer | Acció |
|---|---|
| `app/schemas/matches.py` | Crear |
| `app/repositories/match_repository.py` | Crear |
| `app/services/match_service.py` | Crear |
| `app/routers/matches.py` | Crear |
| `app/main.py` | Modificar — afegir `matches` a imports i `include_router` |

---

## Disseny per capa

### Capa 1 — `schemas/matches.py`

```python
from pydantic import BaseModel

class MatchCreateResponse(BaseModel):
    match_id: str
    status: str
```

---

### Capa 2 — `repositories/match_repository.py`

```python
from motor.motor_asyncio import AsyncIOMotorDatabase

async def create_match(db: AsyncIOMotorDatabase, doc: dict) -> str:
    result = await db["matches"].insert_one(doc)
    return str(result.inserted_id)
```

---

### Capa 3 — `services/match_service.py`

```python
import asyncio
from datetime import datetime, timezone
from bson import ObjectId

from app.repositories import match_repository

BUCKET_RAW = "raw-videos"

async def upload_match(file_bytes: bytes, title: str, user_id: str, s3, db) -> dict:
    match_id = str(ObjectId())
    minio_key = f"{match_id}/original.mp4"

    await asyncio.to_thread(
        s3.put_object,
        Bucket=BUCKET_RAW,
        Key=minio_key,
        Body=file_bytes,
        ContentType="video/mp4",
    )

    now = datetime.now(timezone.utc)
    await match_repository.create_match(db, {
        "_id":          ObjectId(match_id),
        "user_id":      user_id,          # sub del JWT, string
        "title":        title,
        "date":         now,
        "status":       "pending",
        "video_raw":    minio_key,
        "video_output": None,
        "fps":          None,
        "start_frame":  None,
        "end_frame":    None,
        "roi_polygon":  [],
        "created_at":   now,
        "updated_at":   now,
    })
    return {"match_id": match_id, "status": "pending"}
```

---

### Capa 4 — `routers/matches.py`

```python
import logging
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_app_db, get_current_user, get_s3
from app.schemas.auth import TokenPayload
from app.schemas.matches import MatchCreateResponse
from app.services import match_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/matches", tags=["matches"])


@router.post("", response_model=MatchCreateResponse, status_code=201)
async def create_match(
    video: UploadFile = File(..., description="Fitxer .mp4 del partit"),
    title: str = Form(..., min_length=1, max_length=200),
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
    s3=Depends(get_s3),
):
    if not video.content_type or "video" not in video.content_type:
        raise HTTPException(status_code=422, detail="El fitxer ha de ser un vídeo MP4")

    file_bytes = await video.read()
    try:
        result = await match_service.upload_match(
            file_bytes=file_bytes,
            title=title,
            user_id=current_user.sub,
            s3=s3,
            db=db,
        )
    except Exception:
        logger.exception("Error creant el partit")
        raise HTTPException(status_code=500, detail="Error pujant el vídeo")

    return MatchCreateResponse(**result)
```

---

### `main.py` — canvi mínim

```python
from app.routers import auth, health, labeling, matches
# ...
app.include_router(matches.router)
```

---

## Flux complet

```
Frontend  multipart/form-data (video + title + Bearer token)
        │
        ▼
POST /api/v1/matches
  └─ get_current_user()  ← valida JWT, obté current_user.sub
        │
        ▼
match_service.upload_match(file_bytes, title, user_id)
  1. ObjectId() → match_id
  2. s3.put_object("raw-videos", "{match_id}/original.mp4", bytes)
  3. db["matches"].insert_one({user_id, title, status:"pending", ...})
        │
        ▼
201 { "match_id": "...", "status": "pending" }
```

---

## Document MongoDB `matches`

```json
{
  "_id":          "ObjectId",
  "user_id":      "string (sub del JWT)",
  "title":        "Lliga J12 vs Joventut",
  "date":         "ISODate",
  "status":       "pending",
  "video_raw":    "{match_id}/original.mp4",
  "video_output": null,
  "fps":          null,
  "start_frame":  null,
  "end_frame":    null,
  "roi_polygon":  [],
  "created_at":   "ISODate",
  "updated_at":   "ISODate"
}
```

---

## Errors

| Cas | HTTP |
|---|---|
| Token absent / invàlid | 401 |
| Content-type no és vídeo | 422 |
| MinIO / MongoDB falla | 500 |

---

## Dependències Sprint 3

```
PJM-28 ──prerequisit──► PJM-29  PATCH /matches/{id}/config
PJM-28 ──prerequisit──► PJM-30  ProcessPage frontend
```
