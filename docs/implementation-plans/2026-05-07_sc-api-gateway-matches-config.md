# PJM-29 — PATCH /api/v1/matches/{id}/config: ROI i temps

**Data:** 2026-05-07  
**Estat:** En Procés  
**Sprint:** 3 — Pipeline IA MVP  
**Ticket:** PJM-29  
**Etiqueta:** `backend`  

---

## Objectiu

Endpoint per desar la configuració que l'usuari defineix al frontend just abans de processar el vídeo: el polígon ROI (zona de joc), el segon d'inici i el segon de fi del partit.

---

## Context

Prerequisit: PJM-28 ja implementat — existeix el `match_id`.

**Cap fitxer nou.** Tot va als mateixos fitxers creats al PJM-28:

| Fitxer | Acció |
|---|---|
| `app/schemas/matches.py` | Afegir `RoiPoint`, `MatchConfigRequest`, `MatchConfigResponse` |
| `app/repositories/match_repository.py` | Afegir `update_match_config()` |
| `app/services/match_service.py` | Afegir `update_config()` |
| `app/routers/matches.py` | Afegir `PATCH /{match_id}/config` |
| `app/main.py` | **Cap canvi** — router ja registrat |

---

## Disseny per capa

### Capa 1 — `schemas/matches.py` (afegir)

```python
from pydantic import field_validator, model_validator

class RoiPoint(BaseModel):
    x: float
    y: float

class MatchConfigRequest(BaseModel):
    roi_polygon: list[RoiPoint]
    start_seconds: float
    end_seconds: float

    @field_validator('roi_polygon')
    @classmethod
    def min_3_points(cls, v):
        if len(v) < 3:
            raise ValueError('roi_polygon mínim 3 punts')
        return v

    @model_validator(mode='after')
    def end_after_start(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError('end_seconds ha de ser major que start_seconds')
        return self

class MatchConfigResponse(BaseModel):
    match_id: str
    roi_polygon: list[RoiPoint]
    start_seconds: float
    end_seconds: float
```

---

### Capa 2 — `repositories/match_repository.py` (afegir)

```python
from datetime import datetime, timezone
from bson import ObjectId

async def update_match_config(db: AsyncIOMotorDatabase, match_id: str, config: dict) -> dict | None:
    return await db["matches"].find_one_and_update(
        {"_id": ObjectId(match_id)},
        {"$set": {**config, "updated_at": datetime.now(timezone.utc)}},
        return_document=True,
    )
```

---

### Capa 3 — `services/match_service.py` (afegir)

```python
async def update_config(match_id: str, roi_polygon: list, start_seconds: float, end_seconds: float, db) -> dict:
    doc = await match_repository.update_match_config(db, match_id, {
        "roi_polygon":   [{"x": p.x, "y": p.y} for p in roi_polygon],
        "start_seconds": start_seconds,
        "end_seconds":   end_seconds,
    })
    if doc is None:
        raise ValueError("Partit no trobat")
    return {
        "match_id":      str(doc["_id"]),
        "roi_polygon":   doc["roi_polygon"],
        "start_seconds": doc["start_seconds"],
        "end_seconds":   doc["end_seconds"],
    }
```

**Nota MVP:** `start_seconds` / `end_seconds` es guarden directament. La conversió a frames la farà `sc-video-manager` quan conegui el FPS real del vídeo.

---

### Capa 4 — `routers/matches.py` (afegir)

```python
@router.patch("/{match_id}/config", response_model=MatchConfigResponse)
async def update_match_config(
    match_id: str,
    body: MatchConfigRequest,
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
):
    try:
        result = await match_service.update_config(
            match_id=match_id,
            roi_polygon=body.roi_polygon,
            start_seconds=body.start_seconds,
            end_seconds=body.end_seconds,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Error actualitzant config del partit")
        raise HTTPException(status_code=500, detail="Error actualitzant la configuració")
    return MatchConfigResponse(**result)
```

---

## Flux complet

```
PATCH /api/v1/matches/{match_id}/config
  Body JSON: { roi_polygon:[{x,y},...], start_seconds:60.0, end_seconds:3600.0 }
        │
        ▼
router → valida JWT → match_service.update_config()
        │
        ▼
repository: find_one_and_update({_id}, {$set: roi_polygon, start/end_seconds, updated_at})
        │
        ▼
200 { match_id, roi_polygon, start_seconds, end_seconds }
```

---

## Validacions

| Regla | On | HTTP |
|---|---|---|
| `roi_polygon` mínim 3 punts | Pydantic `@field_validator` | 422 |
| `end_seconds > start_seconds` | Pydantic `@model_validator` | 422 |
| `match_id` no trobat a MongoDB | `ValueError` al service | 404 |

---

## Document MongoDB — camps afectats

```json
{
  "roi_polygon":   [{"x": 120.0, "y": 80.0}, {"x": 1800.0, "y": 80.0}, {"x": 960.0, "y": 900.0}],
  "start_seconds": 60.0,
  "end_seconds":   3600.0,
  "updated_at":    "ISODate UTC"
}
```

---

## Endpoint a documentar (`docs/endpoints.md`)

```
PATCH /api/v1/matches/{match_id}/config
  Auth: Bearer (qualsevol usuari autenticat)
  Body JSON:
    - roi_polygon: [{x: float, y: float}]  (mínim 3 punts)
    - start_seconds: float
    - end_seconds: float  (> start_seconds)
  Response 200: { match_id, roi_polygon, start_seconds, end_seconds }
  Response 404: match_id no trobat
  Response 422: validació fallida
```

---

## Dependències Sprint 3

```
PJM-28 ──prerequisit──► PJM-29 (aquest) ──prerequisit──► PJM-30 ProcessPage frontend
```
