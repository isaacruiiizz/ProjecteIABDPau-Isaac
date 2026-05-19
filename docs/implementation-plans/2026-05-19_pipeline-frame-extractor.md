# PJM-64 — Extracció de frames amb ROI i rang de temps

**Estat:** Pendent de confirmació
**Ticket:** PJM-64
**Data:** 2026-05-19

---

## Objectiu

Donada la configuració d'un partit (roi_polygon, start_seconds, end_seconds), extreure
frames del vídeo original de MinIO, retallar-los a la zona ROI i pujar-los a MinIO
al bucket `pending-frames`. Aquests frames seran consumits pel ticket PJM-65 (detecció).

---

## Context — el que ja existeix

### El document de partit a MongoDB té:
```
{
  "_id":           "64abc123...",
  "user_id":       "...",
  "status":        "pending",
  "video_raw":     "64abc123.../original.mp4",   ← clau a MinIO bucket raw-videos
  "roi_polygon":   [{"x": 100, "y": 50}, {"x": 800, "y": 50},
                    {"x": 800, "y": 500}, {"x": 100, "y": 500}],
  "start_seconds": 10.0,
  "end_seconds":   120.0
}
```

### Coordenades del ROI
Els punts `roi_polygon` estan en píxels del vídeo original.
El frontend guarda les coordenades en l'espai del canvas, que s'inicialitza
a `canvas.width = video.videoWidth` i `canvas.height = video.videoHeight`.
Per tant, els valors ja coincideixen amb els píxels reals del vídeo.

Calcularem el **bounding box rectangular**: min_x, min_y, max_x, max_y dels 4 punts.

---

## Fitxers a crear o modificar

```
services/sc-api-gateway/
├── Dockerfile                                   ← MODIFICAR (afegir ffmpeg)
└── app/
    └── services/
        └── pipeline/
            ├── __init__.py                      ← CREAR (buit)
            └── frame_extractor.py               ← CREAR (tota la lògica)
```

---

## Detall de cada fitxer

### 1. `Dockerfile` — afegir ffmpeg

Ara mateix:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*
```

Canvi (afegir `ffmpeg`):
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

---

### 2. `pipeline/__init__.py` — fitxer buit

```python
```

Només existeix perquè Python tracti `pipeline/` com un paquet importable.

---

### 3. `pipeline/frame_extractor.py` — tota la lògica

Una sola funció pública: `extract_frames(match_doc, s3) -> int`

```python
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

BUCKET_RAW     = "raw-videos"
BUCKET_PENDING = "pending-frames"
FRAMES_PER_SEC = 2   # 2 frames per segon


def _roi_bounding_box(roi_polygon: list[dict]) -> tuple[int, int, int, int]:
    """Retorna (x, y, w, h) del bounding box rectangular del polígon ROI."""
    xs = [int(p["x"]) for p in roi_polygon]
    ys = [int(p["y"]) for p in roi_polygon]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    # FFmpeg necessita amplada i alçada parells
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    return x, y, w, h


def extract_frames(match_doc: dict, s3) -> int:
    """
    Descarrega el vídeo de MinIO, extreu frames aplicant ROI i rang de temps
    amb FFmpeg, i puja els frames a pending-frames/{match_id}/.

    Retorna el nombre de frames pujats.
    """
    match_id      = str(match_doc["_id"])
    minio_key     = match_doc["video_raw"]        # "64abc.../original.mp4"
    roi_polygon   = match_doc["roi_polygon"]
    start_seconds = float(match_doc["start_seconds"])
    end_seconds   = float(match_doc["end_seconds"])

    x, y, w, h = _roi_bounding_box(roi_polygon)
    logger.info("match=%s ROI bounding box: x=%d y=%d w=%d h=%d", match_id, x, y, w, h)

    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"sc-frames-{match_id}-"))
    video_path = tmp_dir / "original.mp4"
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    try:
        # 1. Descarrega el vídeo de MinIO
        logger.info("Descarregant vídeo: bucket=%s key=%s", BUCKET_RAW, minio_key)
        s3.download_file(BUCKET_RAW, minio_key, str(video_path))
        logger.info("Vídeo descarregat: %s", video_path)

        # 2. Extreu frames amb FFmpeg
        #    -ss i -to defineixen el rang temporal
        #    fps=2 extreu 2 frames per segon
        #    crop=w:h:x:y retalla al bounding box del ROI
        frame_pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-ss", str(start_seconds),
            "-to", str(end_seconds),
            "-i", str(video_path),
            "-vf", f"fps={FRAMES_PER_SEC},crop={w}:{h}:{x}:{y}",
            "-q:v", "2",
            frame_pattern,
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg ha fallat: {result.stderr[-500:]}")

        # 3. Puja cada frame a pending-frames
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        uploaded = 0
        for frame_file in frame_files:
            object_key = f"{match_id}/{frame_file.name}"
            s3.upload_file(
                str(frame_file),
                BUCKET_PENDING,
                object_key,
            )
            uploaded += 1

        logger.info("match=%s frames pujats: %d", match_id, uploaded)
        return uploaded

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

---

## Cap paquet nou a requirements.txt

| Eina | Com s'usa | Ja disponible? |
|------|-----------|----------------|
| `ffmpeg` (binari) | `subprocess.run(["ffmpeg", ...])` | No → afegir al Dockerfile |
| `boto3` | `s3.download_file / upload_file` | Sí, ja a requirements.txt |
| `subprocess`, `tempfile`, `shutil`, `pathlib` | stdlib Python | Sí, sempre disponibles |

---

## Resultat esperat a MinIO

Després d'executar `extract_frames` per un vídeo de 2 minuts (start=10, end=130):
```
pending-frames/
└── 64abc123.../
    ├── frame_000001.jpg   ← segon 10.0 del vídeo, retallat al ROI
    ├── frame_000002.jpg   ← segon 10.5
    ├── frame_000003.jpg   ← segon 11.0
    ...
    └── frame_000240.jpg   ← segon 130.0 (2fps × 120s = 240 frames)
```

---

## Verificació manual (abans de fer l'endpoint)

Amb el contenidor corrent:
```bash
# 1. Entrar al contenidor
docker exec -it projecteiabdpau-isaac-sc-api-gateway-1 python

# 2. Executar la funció directament
import boto3
from app.config import settings
from app.services.pipeline.frame_extractor import extract_frames

s3 = boto3.client("s3",
    endpoint_url=settings.MINIO_ENDPOINT_URL,
    aws_access_key_id=settings.MINIO_ACCESS_KEY,
    aws_secret_access_key=settings.MINIO_SECRET_KEY,
)
match_doc = {
    "_id": "ID_DEL_PARTIT",            # copiar d'un partit real de MongoDB
    "video_raw": "ID_DEL_PARTIT/original.mp4",
    "roi_polygon": [{"x":100,"y":50},{"x":800,"y":50},{"x":800,"y":500},{"x":100,"y":500}],
    "start_seconds": 10.0,
    "end_seconds": 40.0,
}
n = extract_frames(match_doc, s3)
print(f"Frames extrets: {n}")

# 3. Comprovar a MinIO
# mc ls local/pending-frames/ID_DEL_PARTIT/
```

---

## Ordre d'implementació

1. Modificar `Dockerfile` (afegir ffmpeg)
2. Crear `pipeline/__init__.py`
3. Crear `pipeline/frame_extractor.py`
4. Rebuild: `docker compose build sc-api-gateway`
5. Restart: `docker compose up -d sc-api-gateway`
6. Verificació manual amb un partit real
