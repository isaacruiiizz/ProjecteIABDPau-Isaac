# Pla d'Implementació — Fitxers .env.example per a tots els serveis

**Ticket:** PJM-15
**Data:** 2026-03-20
**Sprint:** Sprint 1 — Fonaments
**Estat:** Completat ✓ (2026-03-20)

---

## 1. Objectiu

Crear els fitxers `.env.example` per a tots els serveis que en necessiten, basant-se íntegrament en el punt 2.13 de `docs/specs.md`. Cap variable secreta portarà el valor real — tots els `# SECRET` tindran valors d'exemple segurs per a ús en local.

---

## 2. Verificació prèvia de .gitignore

`.gitignore` arrel conté:
```
**/.env          ← tots els .env reals exclosos del repositori ✓
!**/.env.example ← els .env.example estan inclosos ✓
```

---

## 3. Serveis i fitxers afectats

| Servei | Fitxer creat |
|--------|-------------|
| `sc-api-gateway` | `services/sc-api-gateway/.env.example` |
| `sc-video-manager` | `services/sc-video-manager/.env.example` |
| `sc-inference-worker` | `services/sc-inference-worker/.env.example` |
| `sc-logic-aggregator` | `services/sc-logic-aggregator/.env.example` |
| `sc-active-learner` | `services/sc-active-learner/.env.example` |
| `sc-frontend` | `services/sc-frontend/.env.example` |
| `sc-mongodb` | `services/sc-mongodb/.env.example` |
| `sc-redis` | `services/sc-redis/.env.example` |
| `sc-object-storage` | `services/sc-object-storage/.env.example` |
| `sc-grafana` | `services/sc-grafana/.env.example` |
| `sc-label-studio` | `services/sc-label-studio/.env.example` |
| `CLAUDE.md` | **Actualitzar** — afegir nous directoris a l'estructura |

> `sc-prometheus` i `sc-dozzle` no apareixen al punt 2.13 — no necessiten `.env`.

---

## 4. Contingut de cada .env.example

### `services/sc-api-gateway/.env.example`
```env
# Servidor
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development

# JWT                                        # SECRET
JWT_SECRET=change_me_min_32_chars_xxxxxxxxxxx
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Internal API Key                           # SECRET
INTERNAL_API_KEY=change_me_internal_key_xxxxx

# MongoDB
MONGO_AUTH_URI=mongodb://sc-mongodb:27017/sc-auth-db
MONGO_APP_URI=mongodb://sc-mongodb:27017/sc-app-db

# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379

# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=change_me_minio_user         # SECRET
MINIO_SECRET_KEY=change_me_minio_password     # SECRET
MINIO_USE_SSL=false

# Sentry (buit en dev, URL en prod)
SENTRY_DSN=
```

---

### `services/sc-video-manager/.env.example`
```env
# Internal API Key                           # SECRET
INTERNAL_API_KEY=change_me_internal_key_xxxxx

# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379
REDIS_QUEUE_VIDEO=video_to_process
REDIS_QUEUE_FRAMES=task_frames

# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=change_me_minio_user         # SECRET
MINIO_SECRET_KEY=change_me_minio_password     # SECRET
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

---

### `services/sc-inference-worker/.env.example`
```env
# Internal API Key                           # SECRET
INTERNAL_API_KEY=change_me_internal_key_xxxxx

# Redis
REDIS_HOST=sc-redis
REDIS_PORT=6379
REDIS_QUEUE_FRAMES=task_frames
REDIS_QUEUE_RESULTS=detected_frames_results

# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=change_me_minio_user         # SECRET
MINIO_SECRET_KEY=change_me_minio_password     # SECRET
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

---

### `services/sc-logic-aggregator/.env.example`
```env
# Internal API Key                           # SECRET
INTERNAL_API_KEY=change_me_internal_key_xxxxx

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

---

### `services/sc-active-learner/.env.example`
```env
# Internal API Key                           # SECRET
INTERNAL_API_KEY=change_me_internal_key_xxxxx

# MinIO
MINIO_ENDPOINT=sc-object-storage:9000
MINIO_ACCESS_KEY=change_me_minio_user         # SECRET
MINIO_SECRET_KEY=change_me_minio_password     # SECRET
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

---

### `services/sc-frontend/.env.example`
```env
# API
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws

# Entorn
VITE_ENV=development
```

---

### `services/sc-mongodb/.env.example`
```env
# MongoDB root                               # SECRET
MONGO_INITDB_ROOT_USERNAME=change_me_mongo_user
MONGO_INITDB_ROOT_PASSWORD=change_me_mongo_password

# Seed — primer usuari admin de l'aplicació
ADMIN_EMAIL=admin@smartchrono.local          # SECRET
ADMIN_PASSWORD=change_me_admin_password      # SECRET
ADMIN_DISPLAY_NAME=Administrador
```

---

### `services/sc-redis/.env.example`
```env
# Buit en dev, obligatori en prod            # SECRET
REDIS_PASSWORD=
```

---

### `services/sc-object-storage/.env.example`
```env
MINIO_ROOT_USER=change_me_minio_user         # SECRET
MINIO_ROOT_PASSWORD=change_me_minio_password # SECRET
```

---

### `services/sc-grafana/.env.example`
```env
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=change_me_grafana_password  # SECRET
GF_SERVER_HTTP_PORT=3001
```

---

### `services/sc-label-studio/.env.example`
```env
# Label Studio
LABEL_STUDIO_PORT=8081
LABEL_STUDIO_USERNAME=admin@smartchrono.local          # SECRET
LABEL_STUDIO_PASSWORD=change_me_ls_password            # SECRET

# Integració MinIO (S3)
MINIO_ENDPOINT=http://sc-object-storage:9000
MINIO_ACCESS_KEY=change_me_minio_user                  # SECRET
MINIO_SECRET_KEY=change_me_minio_password              # SECRET
MINIO_BUCKET_FRAMES=labeling-frames
MINIO_BUCKET_DATASETS=datasets

# Persistència
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data
```

---

## 5. Decisions tècniques

### D1 — Valors d'exemple per a secrets
**Decisió:** Tots els `# SECRET` usen el prefix `change_me_` en lloc dels valors per defecte de la spec (`minioadmin`, `admin1234`, etc.).
**Motiu:** Els valors per defecte de la spec (`minioadmin`) podrien ser copiats sense canviar a producció. El prefix `change_me_` és inequívocament un placeholder.
**Excepció:** Variables no secretes (`REDIS_HOST`, `MINIO_USE_SSL`, etc.) mantenen els valors reals de la spec perquè són configuració, no credencials.

### D2 — Serveis sense .env.example
**Decisió:** `sc-prometheus` i `sc-dozzle` no reben `.env.example`.
**Motiu:** El punt 2.13 de la spec no defineix variables d'entorn per a aquests serveis. Prometheus es configura via `prometheus.yml` (PJM-47) i Dozzle no necessita configuració externa.

### D3 — Directoris nous creats
Els serveis d'infraestructura (`sc-mongodb`, `sc-redis`, `sc-object-storage`, `sc-grafana`, `sc-label-studio`) no tenien directori a `services/`. Es crearan com a efecte secundari d'aquest ticket.

---

## 6. Riscos

| Risc | Probabilitat | Impacte | Mitigació |
|------|-------------|---------|-----------|
| Usuari copia .env.example → .env sense canviar secrets | Mitja | Credencials febles en prod | Prefix `change_me_` força atenció |
| Variable nova afegida a la spec sense actualitzar .env.example | Baixa | Servei no arrenca | Documentat: actualitzar .env.example quan canvia 2.13 |

---

## 7. Ordre d'implementació

1. Serveis Python (ja tenen directori): `sc-api-gateway`, `sc-video-manager`, `sc-inference-worker`, `sc-logic-aggregator`, `sc-active-learner`, `sc-frontend`
2. Serveis d'infraestructura (cal crear el directori): `sc-mongodb`, `sc-redis`, `sc-object-storage`, `sc-grafana`, `sc-label-studio`
3. Actualitzar `CLAUDE.md` — estructura de fitxers

---

## 8. Fora d'abast d'aquest ticket

- Creació dels `.env` reals (mai al repositori) → responsabilitat de cada desenvolupador
- Scripts d'inicialització de MongoDB (`01-init.js`) → PJM-18
- Script de creació de buckets MinIO (`create-buckets.sh`) → PJM-16
- Configuració de Grafana (`provisioning/`) → Sprint 7
