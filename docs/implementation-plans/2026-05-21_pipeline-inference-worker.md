# PJM-65 — Detecció de jugadors per frame amb RT-DETR

**Estat:** En Procés  
**Ticket:** PJM-65  
**Data:** 2026-05-21  

---

## 1. Anàlisi del que ja existeix

El servei `sc-inference-worker` té codi parcial del sprint anterior que cal aprofitar i completar:

| Fitxer | Estat | Acció |
|---|---|---|
| `app/main.py` | Complet | Petita modificació: passar `stop_event` a `inference_worker.run()` |
| `app/config.py` | Complet | Afegir `RTDETR_MODEL_KEY` i `RTDETR_CONFIDENCE` |
| `app/services/rfdetr_service.py` | Complet | No tocar — és per al pipeline d'etiquetatge |
| `app/workers/labeling_worker.py` | Complet | No tocar |
| `app/workers/inference_worker.py` | Placeholder | **Implementar** |
| `app/services/rtdetr_service.py` | No existeix | **Crear** |
| `requirements.txt` | Parcial | Afegir `ultralytics` |

---

## 2. Context tècnic

### Model
- **Fitxer**: `rtdetr/best.pt` al bucket `models` de MinIO (63MB)
- **Entrenament**: Ultralytics RT-DETR — confirmat per `runs/detect/SmartChrono/rtdetr_local/args.yaml` (`task: detect`, `imgsz: 480`)
- **Biblioteca**: `ultralytics` — la mateixa amb la que es va entrenar
- **Device**: CPU (el servidor no té GPU drivers; configurable via env var `INFERENCE_DEVICE`)

### Flux complet d'un frame
```
PJM-67 (futur) → encola missatge a task_frames
      │
      ▼
sc-inference-worker BLPOP ["task_frames", "model_promoted"]
      │
      ├─ model_promoted → recarregar model en calent
      │
      └─ task_frames:
            1. Descarregar frame de MinIO (pending-frames)
            2. RTDETRService.predict(image_bytes) → deteccions
            3. Publicar resultat a detected_frames_results
```

### Payloads Redis (de l'especificació)
**Entrada** (`task_frames`):
```json
{
  "match_id": "6a0f28430b76aee263b55b6e",
  "frame_id": "6a0f28430b76aee263b55b6e/frame_000001.jpg",
  "minio_bucket": "pending-frames",
  "minio_key": "6a0f28430b76aee263b55b6e/frame_000001.jpg",
  "frame_number": 1,
  "timestamp_s": 0.5
}
```

**Sortida** (`detected_frames_results`):
```json
{
  "match_id": "6a0f28430b76aee263b55b6e",
  "frame_id": "6a0f28430b76aee263b55b6e/frame_000001.jpg",
  "frame_number": 1,
  "timestamp_s": 0.5,
  "detections": [
    {"x1": 0.12, "y1": 0.30, "x2": 0.18, "y2": 0.55, "confidence": 0.91, "class_id": 0, "class_name": "player_own"}
  ]
}
```
Coordenades normalitzades [0..1] per coherència amb `rfdetr_service.py`.

---

## 3. Canvis a implementar (per ordre)

### Fase 1 — `requirements.txt`
Afegir `ultralytics>=8.3.0` a la llista existent.

### Fase 2 — `app/config.py`
Afegir camps nous a `Settings`:
```python
RTDETR_MODEL_KEY: str = "rtdetr/best.pt"      # clau dins el bucket models
RTDETR_CONFIDENCE: float = 0.5                 # llindar de confiança inferència
INFERENCE_DEVICE: str = "cpu"                  # 'cpu' o '0' per GPU
INFERENCE_CLAHE: bool = True                   # normalització de contrast per backlight
INFERENCE_SHARPEN: bool = True                 # unsharp mask per motion blur
```

### Fase 3 — `app/services/rtdetr_service.py` (fitxer nou)
Responsabilitat única: càrrega, preprocessament i inferència del model Ultralytics RT-DETR.

#### 3a. Preprocessament de la imatge (`_preprocess`)

Tres transformacions aplicades **per ordre** abans de passar el frame al model:

**1. CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- **Per qué**: L'spec documenta explícitament el problema de contrallum (finestres grans al fons). Sense CLAHE, els jugadors a la meitat del camp apareixen en semisombra i el model els perd. CLAHE normalitza el contrast localment sense "cremar" les zones lluminoses.
- **Com**: Convertir a espai LAB, aplicar CLAHE únicament al canal L (lluminositat), reconvertir a BGR. Paràmetres: `clipLimit=2.0`, `tileGridSize=(8, 8)`.
- **Activable via env var**: `INFERENCE_CLAHE=true` (default: true)

**2. Sharpening lleuger (unsharp mask)**
- **Per qué**: Els frames de vídeo tenen motion blur quan els jugadors es mouen ràpid. Un sharpening suau recupera contorns del bounding box, especialment als jugadors de fons (petits, ~20px d'alçada).
- **Com**: kernel unsharp mask 3×3 amb `alpha=1.3`, `beta=-0.3`. No augmentar soroll de fons.
- **Activable via env var**: `INFERENCE_SHARPEN=true` (default: true)

**3. Resize explícit a `imgsz=480`**
- **Per qué**: El model va ser entrenat amb `imgsz: 480`. Passar-ho explícitament a `predict()` garanteix que la inferència usa la mateixa distribució que l'entrenament. Ultralytics ho passaria per defecte a 640 si no s'especifica.
- **Com**: `self._model.predict(array, imgsz=480, ...)` — Ultralytics fa el resize intern amb letterbox (no distorsiona).

**No s'aplica:**
- Blanc i negre → el model es va entrenar en color; eliminaria informació de samarreta que usa el jersey_classifier
- Undistortion → requereix matriu de calibratge de la càmera concreta; no disponible
- Augmentació → és per a entrenament, no per a inferència

#### 3b. Estructura del servei

```
RTDETRService
  __init__(model_path, confidence, device, clahe, sharpen)
    → carrega RTDETR(model_path) amb Ultralytics
    → inicialitza cv2.createCLAHE si clahe=True
    → guarda configuració de sharpening
    → model en memòria durant tota la vida del procés

  _preprocess(bgr_array) → bgr_array
    → aplica CLAHE al canal L (si activat)
    → aplica unsharp mask (si activat)
    → retorna array preparat

  predict(image_bytes) → list[dict]
    → decode: cv2.imdecode(image_bytes)
    → _preprocess(array)
    → self._model.predict(array, conf=..., imgsz=480, device=..., verbose=False)
    → extreu xyxyn (coordenades normalitzades [0..1]) de r.boxes
    → retorna [{x1, y1, x2, y2, confidence, class_id, class_name}]
    → llista buida si cap detecció

  get_image_array(image_bytes) → np.ndarray
    → retorna array RGB preprocessat (per Jersey Classifier en tickets futurs)
```

#### 3c. Nota sobre dorsals (tickets futurs)

El `classifier/best.pt` (2.8MB, a MinIO) s'usarà per llegir el número de dorsal a partir d'un crop del jugador. Per a aquests crops cal un preprocessament addicional **no implementat en PJM-65**:
- **Upscaling LANCZOS** a alçada fixa (64px): els jugadors al fons només ocupen ~20px d'alçada; cal ampliar el crop abans de passar-lo al classificador.
- **Sharpening addicional**: els crops petits perden detall en el resize; un pas extra de sharpening ajuda.

Això anirà en un `ClassifierService` separat en el ticket corresponent.

### Fase 4 — `app/workers/inference_worker.py` (reemplaçar el placeholder)
Segueix el mateix patró que `labeling_worker.py`:

```
run(stop_event)
  1. Crear clients Redis i MinIO (idèntics als de labeling_worker)
  2. Descarregar model de MinIO → /tmp/rtdetr_best.pt
  3. Instanciar RTDETRService
  4. Bucle mentre not stop_event.is_set():
       raw = redis_client.blpop(["task_frames", "model_promoted"], timeout=5)
       if raw is None: continue
       queue, data = raw
       if queue == b"model_promoted":
           → recarregar RTDETRService (hotswap del model)
       else (task_frames):
           → _process_frame(payload, minio_client, rtdetr_service, redis_client)
```

Funció `_process_frame`:
```
1. Descarregar image_bytes de pending-frames
2. detections = rtdetr_service.predict(image_bytes)
3. Construir result_payload amb match_id, frame_id, frame_number, timestamp_s, detections
4. redis_client.rpush(REDIS_QUEUE_RESULTS, json.dumps(result_payload))
5. Log: frame processat, N deteccions
```

Errors: descartar i logar. Mai reencuar (per especificació del CLAUDE.md).

### Fase 5 — `app/main.py`
Una línia canviada: passar `stop_event` a `inference_worker.run`:
```python
# ABANS:
inference_thread = threading.Thread(target=inference_worker.run, ...)
# DESPRÉS:
inference_thread = threading.Thread(target=inference_worker.run, args=(stop_event,), ...)
```

---

## 4. Verificació

### Test manual (sense PJM-67)
Injectar manualment un missatge a `task_frames` via Redis CLI:

```bash
docker exec <redis_container> redis-cli RPUSH task_frames '{
  "match_id": "6a0f28430b76aee263b55b6e",
  "frame_id": "6a0f28430b76aee263b55b6e/frame_000001.jpg",
  "minio_bucket": "pending-frames",
  "minio_key": "6a0f28430b76aee263b55b6e/frame_000001.jpg",
  "frame_number": 1,
  "timestamp_s": 0.5
}'
```

Verificar:
1. Logs del worker: `inference_frame_done` amb N deteccions
2. `redis_client.blpop("detected_frames_results")` retorna el payload esperat
3. El worker continua processant el següent frame sense caure

### Criteri d'acceptació
- Worker arrenca, descarrega el model de MinIO i comença a escoltar
- Un missatge a `task_frames` produeix un missatge a `detected_frames_results` amb les bounding boxes
- Si un frame falla (error MinIO, error inferència), el worker continua amb el següent

---

## 5. Fitxers que NO canvien
- `app/services/rfdetr_service.py` — pipeline d'etiquetatge, no tocar
- `app/workers/labeling_worker.py` — ja funciona
- `app/utils/jersey_classifier.py` — no és necessari per PJM-65
- `Dockerfile` — ja té Python 3.11-slim + torch CPU instal·lat
