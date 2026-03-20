# Config

## 1. Variables d'Entorn

Cada servei té el seu propi fitxer `.env` ubicat a `services/{nom-servei}/.env`. Tots els fitxers `.env` estan exclosos del repositori via `.gitignore`. El repositori inclou un fitxer `.env.example` per a cada servei amb els noms de les variables i valors d'exemple segurs per a desenvolupament local.

**Convenció de noms:**
- Majúscules amb separador `_`.
- Prefix del servei per a variables compartides que apareixen a més d'un contenidor (ex: `REDIS_HOST`, `MONGO_URI`).
- Les variables que contenen secrets reals (claus, contrasenyes) s'indiquen amb el comentari `# SECRET`.

### `services/sc-api-gateway/.env`
 
```env
# Servidor
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development
 
# JWT # SECRET
JWT_SECRET=dev_jwt_secret_canvia_en_produccio_32bytes
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
 
# Internal API Key # SECRET
INTERNAL_API_KEY=dev_internal_key_canvia_en_produccio
 
# MongoDB
MONGO_AUTH_URI=mongodb://sc-mongodb:27017/sc-auth-db
MONGO_APP_URI=mongodb://sc-mongodb:27017/sc-app-db
 
# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379
 
# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=minioadmin          # SECRET
MINIO_SECRET_KEY=minioadmin          # SECRET
MINIO_USE_SSL=false
 
# Sentry
SENTRY_DSN=                          # buit en dev, URL en prod
```

### `services/sc-video-manager/.env`
 
```env
# Internal API Key # SECRET
INTERNAL_API_KEY=dev_internal_key_canvia_en_produccio
 
# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379
REDIS_QUEUE_VIDEO=video_to_process
REDIS_QUEUE_FRAMES=task_frames
 
# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=minioadmin          # SECRET
MINIO_SECRET_KEY=minioadmin          # SECRET
MINIO_USE_SSL=false
MINIO_BUCKET_RAW=raw-videos
MINIO_BUCKET_PENDING=pending-frames
MINIO_BUCKET_PROCESSED_FRAMES=processed-frames
MINIO_BUCKET_OUTPUT=processed-videos
 
# FFmpeg
VIDEO_FPS_DEFAULT=25
VIDEO_MAX_RESOLUTION=1920x1080
 
# Sentry
SENTRY_DSN=
```

### `services/sc-inference-worker/.env`
 
```env
# Internal API Key # SECRET
INTERNAL_API_KEY=dev_internal_key_canvia_en_produccio
 
# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379
REDIS_QUEUE_FRAMES=task_frames
REDIS_QUEUE_RESULTS=detected_frames_results
 
# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=minioadmin          # SECRET
MINIO_SECRET_KEY=minioadmin          # SECRET
MINIO_USE_SSL=false
MINIO_BUCKET_PENDING=pending-frames
MINIO_BUCKET_FEEDBACK=feedback-data
MINIO_BUCKET_MODELS=models
 
# IA
YOLO_MODEL_PATH=yolo/weights/v1.pt
CNN_MODEL_PATH=cnn/weights/v1.keras
INFERENCE_CONFIDENCE_THRESHOLD=0.6
PREFETCH_BUFFER_SIZE=8
 
# GPU
CUDA_VISIBLE_DEVICES=0
 
# Sentry
SENTRY_DSN=
```

### `services/sc-logic-aggregator/.env`
 
```env
# Internal API Key # SECRET
INTERNAL_API_KEY=dev_internal_key_canvia_en_produccio
 
# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379
REDIS_QUEUE_RESULTS=detected_frames_results
 
# MongoDB
MONGO_APP_URI=mongodb://sc-mongodb:27017/sc-app-db
 
# Lògica de cronometratge
TRACKING_DISAPPEARANCE_BUFFER_SECONDS=3
TRACKING_MIN_CONFIDENCE=0.6
 
# Sentry
SENTRY_DSN=
```

### `services/sc-active-learner/.env`
 
```env
# Internal API Key # SECRET
INTERNAL_API_KEY=dev_internal_key_canvia_en_produccio
 
# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=minioadmin          # SECRET
MINIO_SECRET_KEY=minioadmin          # SECRET
MINIO_USE_SSL=false
MINIO_BUCKET_FEEDBACK=feedback-data
MINIO_BUCKET_MODELS=models
 
# Entrenament
TRAINING_MIN_SAMPLES=50
TRAINING_EPOCHS=10
TRAINING_BATCH_SIZE=16
YOLO_BASE_WEIGHTS=yolo/weights/v1.pt
CNN_BASE_WEIGHTS=cnn/weights/v1.keras
 
# Sentry
SENTRY_DSN=
```

### `services/sc-frontend/.env`
 
```env
# API
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
 
# Entorn
VITE_ENV=development
```
 
---
 
### `services/sc-mongodb/.env`
 
```env
MONGO_INITDB_ROOT_USERNAME=admin         # SECRET
MONGO_INITDB_ROOT_PASSWORD=admin         # SECRET
 
# Seed — primer usuari admin de l'aplicació
ADMIN_EMAIL=admin@smartchrono.local      # SECRET
ADMIN_PASSWORD=admin1234                 # SECRET
ADMIN_DISPLAY_NAME=Administrador
```

### `services/sc-redis/.env`
 
```env
REDIS_PASSWORD=                      # buit en dev, obligatori en prod # SECRET
```
 
---
 
### `services/sc-object-storage/.env`
 
```env
MINIO_ROOT_USER=minioadmin           # SECRET
MINIO_ROOT_PASSWORD=minioadmin       # SECRET
```
 
---
 
### `services/sc-grafana/.env`
 
```env
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin     # SECRET
GF_SERVER_HTTP_PORT=3001
```

### `services/sc-label-studio/.env`
 
```env
# Label Studio
LABEL_STUDIO_PORT=8081
LABEL_STUDIO_USERNAME=admin@smartchrono.local  # SECRET
LABEL_STUDIO_PASSWORD=admin                    # SECRET
 
# Integració MinIO (S3)
MINIO_ENDPOINT=http://sc-object-storage:9000
MINIO_ACCESS_KEY=minioadmin                    # SECRET
MINIO_SECRET_KEY=minioadmin                    # SECRET
MINIO_BUCKET_FRAMES=labeling-frames
MINIO_BUCKET_DATASETS=datasets
 
# Persistència
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data
```

### Variables compartides entre serveis
 
Les variables següents apareixen a més d'un `.env` i han de tenir el mateix valor a tots els serveis on apareguin. En entorns locals es copien manualment; en producció s'injecten via Docker Secrets o el sistema de secrets de l'orquestrador.
 
| Variable | Serveis | Descripció |
| :--- | :--- | :--- |
| `INTERNAL_API_KEY` | api-gateway, video-manager, inference-worker, logic-aggregator, active-learner | Clau de comunicació entre microserveis |
| `REDIS_HOST` / `REDIS_PORT` | api-gateway, video-manager, inference-worker, logic-aggregator | Adreça del broker Redis |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | api-gateway, video-manager, inference-worker, active-learner | Credencials d'accés a MinIO |
| `SENTRY_DSN` | Tots els serveis Python | DSN de Sentry per a reporting d'errors |

## 2 Decisions Tècniques per a la Implementació

### Healthchecks i ordre d'arrencada (docker-compose)

Tots els serveis que depenen d'altres han de declarar `depends_on` amb `condition: service_healthy`. Els serveis base han de definir un `healthcheck` explícit.
 
| Servei | Healthcheck |
| :--- | :--- |
| `sc-mongodb` | `mongosh --eval "db.adminCommand('ping')"` |
| `sc-redis` | `redis-cli ping` |
| `sc-object-storage` | `curl -f http://localhost:9000/minio/health/live` |
| `sc-api-gateway` | `curl -f http://localhost:8000/health` |
 
Ordre d'arrencada per dependències:
 
1. `sc-mongodb`, `sc-redis`, `sc-object-storage` — sense dependències, arrenquen en paral·lel.
2. `sc-api-gateway` — espera `sc-mongodb` + `sc-redis` + `sc-object-storage` sans.
3. `sc-video-manager`, `sc-logic-aggregator`, `sc-active-learner` — esperen `sc-redis` + `sc-object-storage` sans.
4. `sc-inference-worker` — espera `sc-redis` + `sc-object-storage` sans.
5. `sc-frontend` — espera `sc-api-gateway` sa.
6. `sc-prometheus`, `sc-grafana`, `sc-dozzle` — sense dependències crítiques, arrenquen en paral·lel.

### Convenció d'endpoints REST
 
Tots els endpoints segueixen el prefix `/api/v1/` seguit del nom del recurs en plural i minúscules.
 
```
/api/v1/{recurs}
/api/v1/{recurs}/{id}
/api/v1/{recurs}/{id}/{sub-recurs}
```

Els endpoints d'autenticació **no** porten prefix `/api/v1/` perquè no són recursos de negoci.
 
L'endpoint `/health` tampoc porta prefix — és un endpoint de sistema usat pels healthchecks de Docker:
```
GET /health → {"status": "ok"}
```

### Format de respostes de l'API
 
**Resposta correcta:** format lliure segons el recurs, definit per cada endpoint. FastAPI serialitza automàticament els models Pydantic.
 
**Resposta d'error:** format estàndard FastAPI sense modificacions.

Els errors 500 mai exposen stack traces al client. El stack trace complet s'envia únicament a Sentry (vegeu punt 2.5).

### Política CORS
 
**Desenvolupament local:** CORS permissiu per facilitar el treball entre `:3000` (frontend) i `:8000` (API).
 
```python
# sc-api-gateway/main.py — només quan API_ENV=development
from fastapi.middleware.cors import CORSMiddleware
 
if settings.API_ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
```
 
**Producció:** frontend i API s'allotgen a la mateixa màquina i es serveixen des del mateix origen. CORS no és necessari — el middleware no s'afegeix quan `API_ENV=production`.

## 3. Protocol de Missatges de les Cues Redis

Tots els missatges publicats a les cues Redis són objectes JSON serialitzats. Cap servei pot publicar un missatge sense els camps obligatoris definits aquí.

### Cua `video_to_process` — Publicador: `sc-api-gateway` · Consumidor: `sc-video-manager`

Aquesta cua gestiona **dos tipus de feina** per a `sc-video-manager`, diferenciats pel camp `job_type`:

**`job_type: "process_match"` — Processament de partit real**
```json
{
  "job_type": "process_match",
  "match_id": "a3f1c2d4-7e81-...",
  "minio_bucket": "raw-videos",
  "minio_key": "a3f1c2d4-7e81-.../original.mp4",
  "fps": 25,
  "start_frame": 1500,
  "end_frame": 138000
}
```
Resultat: `sc-video-manager` extreu frames → puja a `pending-frames` → publica tasques a `task_frames`.

**`job_type: "process_labeling"` — Trossejament per etiquetatge**
```json
{
  "job_type": "process_labeling",
  "session_id": "b7e2f1a0-...",
  "minio_bucket": "labeling-videos",
  "minio_key": "b7e2f1a0-.../original.mp4",
  "frame_interval": 2
}
```
Resultat: `sc-video-manager` extreu 1 frame cada `frame_interval` segons → puja a `labeling-frames`. No publica res a cap altra cua.

### Cua `task_frames` — Publicador: `sc-video-manager` · Consumidor: `sc-inference-worker`
 
Un missatge per frame. Només s'usa per a partits reals (`process_match`), mai per a etiquetatge.
 
```json
{
  "match_id": "a3f1c2d4-7e81-...",
  "frame_id": "frame_000001",
  "minio_bucket": "pending-frames",
  "minio_key": "a3f1c2d4-7e81-.../frame_000001.jpg",
  "frame_number": 1,
  "timestamp_s": 0.04
}
```

### Cua `detected_frames_results` — Publicador: `sc-inference-worker` · Consumidor: `sc-logic-aggregator`
 
Un missatge per frame processat, amb totes les deteccions del frame.
 
```json
{
  "match_id": "a3f1c2d4-7e81-...",
  "frame_id": "frame_000001",
  "frame_number": 1,
  "timestamp_s": 0.04,
  "detections": [
    {
      "track_id": 3,
      "bbox": { "x1": 120, "y1": 340, "x2": 180, "y2": 430 },
      "class": "player_own",
      "dorsal": 9,
      "dorsal_confidence": 0.91
    },
    {
      "track_id": 7,
      "bbox": { "x1": 560, "y1": 280, "x2": 620, "y2": 370 },
      "class": "player_own",
      "dorsal": null,
      "dorsal_confidence": 0.0
    }
  ]
}
```
 
- `dorsal` és `null` si la CNN no ha pogut identificar el número o la confiança és < `INFERENCE_CONFIDENCE_THRESHOLD`.
- `class` accepta: `player_own`, `other`. Les deteccions `other` s'inclouen al missatge però `sc-logic-aggregator` les ignora.

### Regles generals
 
- Tots els missatges s'afegeixen a la cua amb `RPUSH` i es consumeixen amb `BLPOP` (blocking pop). Cap servei usa Pub/Sub per a les cues de treball.
- Si un missatge no es pot processar (error de parsing, camps obligatoris absents), el servei consumidor l'ha de registrar com a `ERROR` a Sentry i descartar-lo. **Mai reencuar un missatge erroni** — generaria un bucle infinit.
- El camp `job_type` és obligatori a `video_to_process`. Si `sc-video-manager` rep un missatge sense `job_type`, el descarta i registra l'error.
- `sc-inference-worker` escolta dues cues simultàniament: `task_frames` (frames a processar)
  i `model_promoted` (nous models a carregar). Per fer-ho amb una sola crida bloquejant,
  s'usa `BLPOP` amb múltiples claus — Redis retorna el primer missatge disponible de
  qualsevol de les dues:
 
  ```python
  # sc-inference-worker — bucle principal
  queue, message = redis_client.blpop(['task_frames', 'model_promoted'], timeout=0)
  if queue == b'task_frames':
      process_frame(message)
  elif queue == b'model_promoted':
      load_new_model(message)
  ```
 
  L'ordre de les claus defineix la prioritat en cas d'empat: `task_frames` té prioritat
  sobre `model_promoted`. Si arriben missatges a les dues cues simultàniament, el worker
  processarà primer el frame i després carregarà el model.

## 4. `sc-video-manager` — Arquitectura Worker Pur
 
`sc-video-manager` és un **worker pur basat en Redis**. No exposa cap endpoint HTTP ni port al host. Tota la comunicació amb la resta del sistema es fa exclusivament a través de la cua `video_to_process` de Redis (entrada) i de MongoDB (escriptura d'estat).

### Per què worker pur i no servidor HTTP
 
- L'`sc-api-gateway` ja és l'únic punt d'entrada HTTP del sistema. Afegir un servidor HTTP a `sc-video-manager` crearia un segon punt d'entrada que hauria d'autenticar-se, mantenir-se i monitoritzar-se innecessàriament.
- Redis com a broker ja proporciona el canal de comunicació. Publicar un missatge a `video_to_process` és suficient per desencadenar qualsevol tipus de feina.
- Un worker que cau i es reinicia simplement reprèn la cua des d'on estava. Un servidor HTTP perd les peticions en vol.

### Cicle de vida del worker
 
```
Arrencada
    │
    ▼
BLPOP video_to_process (bloquejant, espera missatge)
    │
    ▼
Llegeix job_type del missatge
    │
    ├── job_type: "process_match"
    │       │
    │       ▼
    │   Actualitza matches.status → "processing" a MongoDB
    │       │
    │       ▼
    │   Descarrega vídeo de MinIO (raw-videos)
    │       │
    │       ▼
    │   Extreu frames amb FFmpeg (rang start_frame..end_frame)
    │       │
    │       ▼
    │   Per cada frame:
    │     - Puja a MinIO (pending-frames)
    │     - RPUSH task_frames amb el payload del frame
    │       │
    │       ▼
    │   Actualitza matches.status → "frames_ready" a MongoDB
    │       │
    │       ▼
    │   Torna a BLPOP (espera el proper missatge)
    │
    └── job_type: "process_labeling"
            │
            ▼
        Descarrega vídeo de MinIO (labeling-videos)
            │
            ▼
        Extreu 1 frame cada {frame_interval} segons amb FFmpeg
            │
            ▼
        Per cada frame:
          - Puja a MinIO (labeling-frames)
            │
            ▼
        No publica res a cap altra cua
            │
            ▼
        Torna a BLPOP (espera el proper missatge)
```

### Estats que escriu a MongoDB
 
| Moment | Camp actualitzat | Valor |
| :--- | :--- | :--- |
| Inici de processament de partit | `matches.status` | `"processing"` |
| Tots els frames extrets i pujats | `matches.status` | `"frames_ready"` |
| Error durant l'extracció | `matches.status` | `"error"` |

Per a feines de tipus `process_labeling` no s'escriu res a MongoDB — no hi ha cap document de partit associat.
 
### Gestió d'errors
 
Si qualsevol pas falla (descàrrega de MinIO, FFmpeg, upload de frame), el worker:
1. Actualitza `matches.status → "error"` a MongoDB (si és un `process_match`).
2. Registra l'error complet a Sentry amb el `match_id` o `session_id` com a context.
3. **No reencua el missatge** — el descarta i torna a escoltar la cua.
4. Continua processant el proper missatge normalment.

## 5. Protocol de Promoció de Models
 
Quan `sc-active-learner` entrena un nou model i supera les mètriques d'acceptació (vegeu punt 4.5), publica un missatge a la cua `model_promoted` de Redis. `sc-inference-worker` escolta aquesta cua i carrega el nou model en calent sense reiniciar el contenidor.
 
---
 
### Cua `model_promoted` — Publicador: `sc-active-learner` · Consumidor: `sc-inference-worker`
 
```json
{
  "model_type": "yolo",
  "version": "v2",
  "minio_bucket": "models",
  "minio_key": "yolo/weights/v2.pt",
  "metrics": {
    "map50": 0.91
  }
}
```
 
- `model_type` accepta: `yolo`, `cnn`.
- `version` segueix el versionat incremental definit al punt 2.9.
- `metrics` conté les mètriques de validació que han superat el llindar — útil per a traçabilitat a Sentry.
 
---
 
### Flux complet de promoció
 
```
sc-active-learner entrena nou model
        │
        ▼
Valida sobre test_set fix de MinIO
        │
        ├── No supera el llindar
        │       │
        │       ▼
        │   Guarda el model com a candidat a MinIO
        │   Registra mètriques a Sentry (INFO)
        │   No publica res a Redis
        │
        └── Supera el llindar
                │
                ▼
            Puja pesos a MinIO (models/yolo/weights/v2.pt)
                │
                ▼
            RPUSH model_promoted amb el payload
                │
                ▼
            sc-inference-worker rep el missatge (BLPOP)
                │
                ▼
            Descarrega nous pesos de MinIO
                │
                ▼
            Carrega el model en memòria (substitueix l'anterior)
                │
                ▼
            Registra a Sentry: "Model yolo v2 carregat"
                │
                ▼
            Continua processant frames amb el nou model
```
 
### Comportament de `sc-inference-worker` durant la càrrega
 
- El worker **no atura** el processament de frames mentre descarrega el nou model. Continua usant el model anterior fins que la càrrega és completa.
- La substitució és atòmica: el nou model es carrega en memòria completament abans de substituir el punter a l'anterior.
- Si la descàrrega o càrrega falla, el worker manté el model anterior i registra l'error a Sentry. **No reintenta** — el model antic segueix operatiu.
 
### Models que no superen el llindar
 
Si el nou model no supera les mètriques d'acceptació, `sc-active-learner` el guarda igualment a MinIO com a candidat però amb un prefix `candidate/`:
 
```
models/yolo/weights/v2.pt          ← actiu (ha superat el llindar)
models/yolo/candidate/v3.pt        ← candidat (no ha superat, pendent revisió manual)
```
 
Els candidats mai s'envien a `model_promoted` i `sc-inference-worker` mai els carrega automàticament.