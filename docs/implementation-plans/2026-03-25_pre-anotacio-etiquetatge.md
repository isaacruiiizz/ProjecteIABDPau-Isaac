# Pla d'implementació — Pre-anotació automàtica del pipeline d'etiquetatge

**Data:** 2026-03-25
**Estat:** Completat ✓ (2026-03-26)

---

## Problema

L'etiquetador ha de marcar manualment els bounding boxes des de zero per a cada un dels ~1.800 frames per vídeo. Això representa 2–3 dies de feina per vídeo.

## Solució

Afegir un pas de pre-anotació automàtica: just després que `sc-video-manager` generi els frames, `sc-inference-worker` executa YOLOv8n (model base, sense fine-tuning) sobre cada frame i envia les deteccions a Label Studio com a **prediccions**. L'etiquetador només ha de validar, corregir i completar — no etiquetar des de zero.

---

## Flux actual vs. nou

### Flux actual
```
Upload vídeo → sc-video-manager (extreu frames) → labeling-frames MinIO
                                                        ↓
                                              Label Studio sync (manual)
                                                        ↓
                                         Etiquetador marca TOTS els BBs
```

### Flux nou
```
Upload vídeo → sc-video-manager (extreu frames) → labeling-frames MinIO
                     ↓ (publica a Redis: labeling_frames_to_infer)
              sc-inference-worker (YOLOv8n base)
                     ↓ (per cada frame)
              Label Studio API → POST /api/predictions/
                                         ↓
                             Etiquetador VALIDA i CORREGEIX
```

---

## Components afectats

| Component | Canvi |
|---|---|
| `sc-video-manager` | Publicar cada frame a `labeling_frames_to_infer` + trigger LS sync |
| `sc-inference-worker` | Implementar nou worker de pre-anotació (BLPOP + YOLOv8n + LS API) |
| `sc-inference-worker/requirements.txt` | Afegir `ultralytics`, `minio`, `requests` |
| `sc-inference-worker/.env.example` | Afegir vars LS |
| `docker-compose.yml` | Afegir vars d'entorn LS a sc-inference-worker |

---

## Detall tècnic per fase

### Fase 1 — sc-video-manager: publicar frames a Redis i trigger sync LS

**Fitxer:** `services/sc-video-manager/app/services/labeling_service.py`

Canvis:
1. Després de pujar cada frame a MinIO, publicar a Redis:
   ```json
   {
     "session_id": "abc123",
     "minio_key": "abc123/frame_000001.jpg",
     "frame_name": "frame_000001.jpg"
   }
   ```
   Cua: `labeling_frames_to_infer` (LPUSH)

2. Un cop tots els frames pujats, cridar Label Studio API per triggerar sync del Source Storage:
   - `POST /api/storages/s3/{storage_id}/sync`
   - El `storage_id` ve d'una variable d'entorn `LABEL_STUDIO_SOURCE_STORAGE_ID`
   - Sense aquesta crida, Label Studio pot trigar minuts en auto-sincronitzar

**Fitxer:** `services/sc-video-manager/app/config.py`

Nous camps:
- `LABEL_STUDIO_URL`
- `LABEL_STUDIO_API_TOKEN`
- `LABEL_STUDIO_SOURCE_STORAGE_ID`

---

### Fase 2 — sc-inference-worker: worker de pre-anotació

**Estructura nova:**
```
services/sc-inference-worker/app/
├── main.py                    ← entrypoint: BLPOP dues cues
├── config.py                  ← pydantic Settings
├── workers/
│   ├── __init__.py
│   ├── labeling_worker.py     ← NOU: pre-anotació etiquetatge
│   └── inference_worker.py    ← futur: inferència en producció
├── services/
│   ├── __init__.py
│   ├── yolo_service.py        ← NOU: càrrega i inferència YOLOv8n
│   └── label_studio_service.py ← NOU: client API Label Studio
└── utils/
    ├── __init__.py
    ├── bbox_converter.py      ← NOU: conversió YOLO → format LS
    └── jersey_classifier.py   ← NOU: classifica player_own / others per color de samarreta
```

#### 2a. `yolo_service.py`

- Carrega `yolov8n.pt` des de MinIO `models/yolo/base/yolov8n.pt`
  - **Model base:** YOLOv8n pre-entrenat amb COCO (80 classes, 3.2M paràmetres, mAP@50-95: 37.3)
  - **COCO inclou la classe `person` (id 0)**, que és la que necessitem per detectar jugadors
  - **Ús com a base per a fine-tuning:** el model s'usarà en pre-anotació _sense_ fine-tuning; un cop acumulades anotacions validades per l'etiquetador, es farà fine-tuning específic per a futbol sala (jugadors, àrbits, porters)
  - Font oficial: [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) — `YOLO("yolov8n.pt")` el descarrega automàticament
- Executa inferència sobre un frame (imatge PIL/numpy)
- Retorna llista de deteccions: `[{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.87, "class_id": 0}]`
- Filtra per classe 0 (`person`) i confiança > `INFERENCE_CONFIDENCE_THRESHOLD` (default: 0.3 per pre-anotació)
- Retorna deteccions sense etiqueta de classe — la classificació `player_own` / `others` la fa `jersey_classifier.py`

#### 2b. `jersey_classifier.py`

Classifica cada detecció com `player_own` o `others` analitzant el color dominant de la regió del bounding box:

```python
def classify(image: np.ndarray, detections: list[dict], own_color_hsv: tuple) -> list[dict]:
    """
    Per cada detecció, retalla la franja superior del BB (samarreta, ignorant shorts/cames),
    calcula el color dominant (k-means k=2 o histograma HSV) i compara amb own_color_hsv.
    Assigna 'player_own' si la distància de color és < JERSEY_COLOR_THRESHOLD, 'others' si no.
    """
```

- `own_color_hsv`: color de la samarreta del nostre equip, configurable via `JERSEY_OWN_COLOR_HSV`, posteriorment via frontend (ex: `"120,80,150"`)
- `JERSEY_COLOR_THRESHOLD`: distància màxima en espai HSV (default: 30)
- Si `JERSEY_OWN_COLOR_HSV` no està configurat → totes les deteccions reben `player_own` (comportament per defecte fins que es configuri)
- Retalla el 40% superior del BB per evitar confusió amb shorts/gespa

#### 2c. `bbox_converter.py`

Conversió de format YOLO normalitzat a format Label Studio percentatge:
```
YOLO:          (x1, y1, x2, y2) normalitzat [0..1]
Label Studio:  (x%, y%, width%, height%) relatiu a la imatge
```
```python
x_pct = x1 * 100
y_pct = y1 * 100
w_pct = (x2 - x1) * 100
h_pct = (y2 - y1) * 100
```

#### 2d. `label_studio_service.py`

Mètodes:
- `get_task_by_frame(session_id, frame_name, project_id) → task_id | None`
  - `GET /api/tasks/?project={id}&data__image__contains={frame_name}`
  - Filtra per session_id dins la URL de la imatge
- `post_prediction(task_id, detections, model_version) → bool`
  - `POST /api/predictions/`
  - Format del payload:
    ```json
    {
      "task": 123,
      "model_version": "yolov8n-base-v0",
      "score": 0.85,
      "result": [
        {
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "x": 10.5,
            "y": 20.3,
            "width": 15.2,
            "height": 25.1,
            "rotation": 0,
            "rectanglelabels": ["player_own"]   // o "others" segons jersey_classifier
          }
        }
      ]
    }
    ```

#### 2e. `labeling_worker.py`

Loop principal:
```python
while True:
    msg = redis.blpop("labeling_frames_to_infer", timeout=5)
    if not msg:
        continue
    payload = json.loads(msg[1])
    # 1. Descarrega frame de MinIO (labeling-frames bucket)
    frame_img = minio_service.download_image(payload["minio_key"])
    # 2. Inferència YOLOv8n
    detections = yolo_service.predict(frame_img)
    if not detections:
        continue  # frame sense jugadors, no cal predicció
    # 3. Classifica player_own / others per color de samarreta
    detections = jersey_classifier.classify(frame_img, detections)
    # 4. Busca task_id a Label Studio (retry amb backoff si sync no ha acabat)
    task_id = label_studio_service.get_task_by_frame(
        payload["session_id"], payload["frame_name"], project_id
    )
    if task_id is None:
        # reencua amb delay (sync LS pot estar en curs)
        time.sleep(2)
        redis.lpush("labeling_frames_to_infer", msg[1])
        continue
    # 5. Publica predicció a Label Studio
    label_studio_service.post_prediction(task_id, detections)
```

**Nota:** màxim 3 reencuaments per frame. Si el task no apareix, descartar i logar.

#### 2f. `main.py`

- Inicia el `labeling_worker` en un thread separat
- En el futur, el `inference_worker` (producció) en un altre thread
- Gestió de senyals per shutdown net

---

### Fase 3 — Variables d'entorn i Docker

**`services/sc-inference-worker/.env.example`** (nous camps):
```
LABEL_STUDIO_URL=http://sc-label-studio:8081
LABEL_STUDIO_API_TOKEN=          # SECRET - obtenir de LS settings
LABEL_STUDIO_PROJECT_ID=1
MINIO_ENDPOINT=http://sc-object-storage:9000
MINIO_ACCESS_KEY=                # SECRET
MINIO_SECRET_KEY=                # SECRET
MINIO_BUCKET_LABELING_FRAMES=labeling-frames
REDIS_URL=redis://sc-redis:6379
INFERENCE_LABELING_CONFIDENCE=0.3
JERSEY_OWN_COLOR_HSV=            # ex: "120,80,150" — color HSV dominant samarreta pròpia. Si buit, fallback a player_own per tot
JERSEY_COLOR_THRESHOLD=30        # distància màxima HSV per considerar "player_own"
```

**`services/sc-video-manager/.env.example`** (nous camps):
```
LABEL_STUDIO_URL=http://sc-label-studio:8081
LABEL_STUDIO_API_TOKEN=          # SECRET
LABEL_STUDIO_SOURCE_STORAGE_ID=1
```

**`docker-compose.yml`**: el sc-inference-worker afegirà `profiles: [labeling]`

---

### Fase 4 — requirements.txt

**`services/sc-inference-worker/requirements.txt`:**
```
ultralytics>=8.3.0
minio>=7.2.0
requests>=2.31.0
redis>=5.0.0
sentry-sdk>=2.0.0
python-dotenv>=1.0.0
Pillow>=10.0.0
numpy>=1.26.0
```

---

## Decisions de disseny

| Decisió | Raó |
|---|---|
| Classificació per color de samarreta (`player_own` / `others`) | Més precís que marcar-ho tot igual. `JERSEY_OWN_COLOR_HSV` configurable. Si no es configura, fallback a `player_own` per tot |
| Confiança baixa (0.3) | Millor tenir deteccions de més (fàcil d'esborrar) que perdre jugadors (cal afegir-los) |
| Reencua si task no existeix | La sync de LS pot trigar 1–5 segons. Retry amb backoff suau |
| No publicar `model_promoted` | Aquesta fase no afecta el pipeline de producció |

---

## Ordre d'implementació

1. Fase 3 — `.env.example` i `requirements.txt` 
2. Fase 2 — `sc-inference-worker` (estructura + serveis + worker) 
3. Fase 1 — `sc-video-manager` (afegir publicació Redis + LS sync) 
4. Fase 4 — Test end-to-end: pujar vídeo → verificar prediccions a LS

---

## Resultat esperat

Quan l'etiquetador obri Label Studio, cada frame tindrà els bounding boxes ja dibuixats per YOLOv8n amb etiqueta `player_own` o `others` assignada per color de samarreta. L'etiquetador:
1. Revisa cada frame (~5s per frame en lloc de ~30s)
2. Elimina les deteccions errònies (falsos positius)
3. Afegeix les deteccions que falten (jugadors no detectats)
4. Corregeix les etiquetes mal assignades (ex: rival classificat com `player_own`)

Estimació d'estalvi: **reducció del 70–80% del temps d'etiquetatge**.
