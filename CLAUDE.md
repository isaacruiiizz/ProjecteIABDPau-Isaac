# SmartChrono IP — Guia per a Claude Code

## Llegeix primer

Abans de qualsevol acció, llegeix docs/specs/ — cada fitxer cobreix un àmbit concret.
Llegeix només els fitxers rellevants per al ticket actual.

Si tens qualsevol dubte el fitxer sencer amb més de 1900 linees es troba a `docs/specs.md`, allà està tot junt.

No inventes cap decisió que no hi estigui documentada.

Documenta tots els endpoints a **`docs/endpoints.md`** i totes les decisions tècniques a **`docs/decisions.md`** seguint les plantilles del punt 5 i 6 de l'spec.

---

## Directiva de treball — OBLIGATÒRIA SEMPRE

Abans de qualsevol canvi significatiu (nou servei, nou mòdul, canvi d'esquema, refactorització):

1. Consulta Jira via MCP → identifica el ticket actiu del sprint actual
2. Mou el ticket a `In Progress` abans d'escriure cap línia de codi
3. Crea un pla a `/docs/implementation-plans/YYYY-MM-DD_nom.md` amb `**Estat:** Pendent de confirmació`
4. Actualitza el pla a `**Estat:** En Procés` en el moment que l'usuari confirmi i abans d'implementar
5. Espera confirmació explícita de l'usuari abans d'implementar
6. Implementa per fases verificables, no tot d'un cop
7. Reporta al final de cada fase què s'ha fet i si cal reiniciar algun servei
8. Actualitza el pla a `**Estat:** Completat ✓ (YYYY-MM-DD)` un cop implementat
9. Mou el ticket a `Done` amb una nota breu del que s'ha implementat
10. Fes commit a `develop` de tots els canvis del ticket amb el missatge: `feat(<etiqueta>): <descripció breu> [<TICKET-ID>]`

Aquesta directiva s'aplica sempre, independentment de com estigui formulada la petició.

---

## Estructura de carpetes i fitxers

> **OBLIGATORI:** Actualitzar aquest apartat cada cop que s'afegeix o elimina una carpeta o fitxer significatiu. Sense excepció.

```
ProjecteIABDPau-Isaac/
├── CLAUDE.md                              ← instruccions per a Claude Code (aquest fitxer)
├── docker-compose.yml                     ← tots els serveis Docker  [PJM-14 ✓]
├── .mcp.json                              ← configuració MCP (Jira, etc.)
├── .gitignore
│
├── docs/
│   ├── specs.md                           ← especificació completa unificada (>1900 línies)
│   ├── specs/                             ← especificacions per àmbit — LLEGIR PRIMER
│   │   ├── 01-arquitectura.md             ← visió general, serveis, comunicació
│   │   ├── 02-logica-ia.md                ← pipeline IA, models, active learning
│   │   ├── 03-infraestructura.md          ← Docker, MinIO, Redis, buckets
│   │   ├── 04-seguretat-bd.md             ← JWT, autenticació, MongoDB, esquemes
│   │   ├── 05-config.md                   ← variables d'entorn per servei (punt 2.13)
│   │   ├── 06-projecte.md                 ← Jira, sprints, convencions
│   │   └── 07-estructura.md               ← estructura de carpetes i fitxers
│   ├── endpoints.md                       ← [per crear — PJM-18] registre d'endpoints
│   ├── decisions.md                       ← [per crear — PJM-18] decisions tècniques
│   └── implementation-plans/              ← un fitxer Markdown per ticket implementat
│       ├── 2026-03-19_docker-compose.md   ← PJM-14 ✓
│       └── 2026-03-21_sc-label-studio-minio.md  ← PJM-17 ✓
│
├── services/
│   ├── sc-api-gateway/                    ← FastAPI 3 capes, port 8000  [PJM-55 ✓]
│   │   ├── Dockerfile                     ← python:3.11-slim + curl + uvicorn
│   │   ├── requirements.txt               ← [placeholder — PJM-18]
│   │   ├── .env.example                   ← [PJM-15 ✓]
│   │   └── app/
│   │       └── __init__.py
│   ├── sc-video-manager/                  ← worker pur Redis, sense HTTP  [PJM-55 ✓]
│   │   ├── Dockerfile                     ← python:3.11-slim + ffmpeg + libgl1
│   │   ├── requirements.txt               ← [placeholder]
│   │   ├── .env.example                   ← [PJM-15 ✓]
│   │   └── app/
│   │       └── __init__.py
│   ├── sc-inference-worker/               ← worker GPU, BLPOP 2 cues  [PJM-55 ✓]
│   │   ├── Dockerfile                     ← nvidia/cuda:12.6.3-runtime-ubuntu22.04
│   │   ├── requirements.txt               ← [placeholder]
│   │   ├── .env.example                   ← [PJM-15 ✓]
│   │   └── app/
│   │       └── __init__.py
│   ├── sc-logic-aggregator/               ← [PJM-55 ✓]
│   │   ├── Dockerfile                     ← python:3.11-slim
│   │   ├── requirements.txt               ← [placeholder]
│   │   ├── .env.example                   ← [PJM-15 ✓]
│   │   └── app/
│   │       └── __init__.py
│   ├── sc-active-learner/                 ← [PJM-55 ✓]
│   │   ├── Dockerfile                     ← python:3.11-slim
│   │   ├── requirements.txt               ← [placeholder]
│   │   ├── .env.example                   ← [PJM-15 ✓]
│   │   └── app/
│   │       └── __init__.py
│   ├── sc-frontend/                       ← React + Vite + Tailwind, port 3000  [PJM-55 ✓]
│   │   ├── Dockerfile                     ← multistage: node:22-alpine build + serve
│   │   ├── package.json                   ← [placeholder — PJM-20]
│   │   ├── .env.example                   ← [PJM-15 ✓]
│   │   └── index.html                     ← [placeholder — PJM-20]
│   ├── sc-label-studio/                   ← servei OPCIONAL, port 8081  [PJM-15 ✓] [PJM-17 ✓]
│   │   ├── .env.example
│   │   └── init/
│   │       └── setup-project.py           ← crea projecte LS i connecta MinIO S3
│   ├── sc-mongodb/                        ← [init scripts per crear — PJM-18]  [PJM-15 ✓]
│   │   └── .env.example
│   ├── sc-redis/                          ← [PJM-15 ✓]
│   │   └── .env.example
│   ├── sc-object-storage/                 ← MinIO  [PJM-15 ✓] [PJM-16 ✓]
│   │   ├── .env.example
│   │   └── init/
│   │       ├── create-buckets.sh          ← script init: 9 buckets + lifecycle + 5 usuaris IAM
│   │       └── policies/                  ← polítiques IAM per servei (JSON)
│   │           ├── sc-api-gateway.json
│   │           ├── sc-video-manager.json
│   │           ├── sc-inference-worker.json
│   │           ├── sc-active-learner.json
│   │           └── sc-label-studio.json
│   ├── sc-prometheus/                     ← [per crear — Sprint 7] prometheus.yml
│   ├── sc-grafana/                        ← [PJM-15 ✓]
│   │   └── .env.example
│   └── sc-dozzle/                         ← [per crear — Sprint 4]
│
└── training_pipeline/                     ← scripts manuals fora de Docker (no sistema live)
    ├── README.md
    ├── docker-compose.yaml                ← compose local per a Label Studio d'entrenament
    ├── entrenar_yolo_v2.py                ← script d'entrenament YOLOv8
    ├── organitzar_dataset_yolo.py         ← preparació del dataset YOLO
    ├── preparar_dades.py                  ← extracció de frames
    ├── verificar_sistema.py               ← verificació de l'entorn
    ├── yolov8n.pt                         ← model base YOLOv8n
    ├── config/
    │   └── label_studio_config.xml        ← configuració Label Studio
    ├── export_ls/
    │   └── project_export.json            ← export del projecte Label Studio
    ├── frames/                            ← frames extrets (frame_000000.jpg … frame_000037.jpg)
    └── label_studio_data/
        ├── .env
        ├── export/                        ← exports JSON de les anotacions
        └── media/upload/1/                ← imatges pujades a Label Studio
```

### On anar a buscar cada cosa

| Necessito... | Anar a... |
|---|---|
| Arquitectura i serveis | `docs/specs/01-arquitectura.md` |
| Pipeline IA i active learning | `docs/specs/02-logica-ia.md` |
| Docker, MinIO, Redis, buckets | `docs/specs/03-infraestructura.md` |
| JWT, autenticació, MongoDB, esquemes | `docs/specs/04-seguretat-bd.md` |
| Variables d'entorn per servei | `docs/specs/05-config.md` |
| Jira, sprints, convencions | `docs/specs/06-projecte.md` |
| Estructura de carpetes | `docs/specs/07-estructura.md` |
| Especificació completa (referència) | `docs/specs.md` |
| Historial de decisions tècniques | `docs/decisions.md` |
| Plans d'implementació per ticket | `docs/implementation-plans/YYYY-MM-DD_nom.md` |
| Endpoints documentats | `docs/endpoints.md` |
| Configuració Docker de tots els serveis | `docker-compose.yml` |
| Codi d'un servei concret | `services/{nom-servei}/` |
| Variables d'entorn d'un servei | `services/{nom-servei}/.env` (gitignored) |
| Variables d'entorn exemple | `services/{nom-servei}/.env.example` |
| Scripts d'entrenament YOLO | `training_pipeline/` |
| Model base YOLOv8 | `training_pipeline/yolov8n.pt` |
| Anotacions Label Studio | `training_pipeline/label_studio_data/` |

---

## Regles crítiques — no trencar mai

### Arquitectura
- `sc-video-manager` és un **worker pur**. No té servidor HTTP. Escolta Redis amb `BLPOP video_to_process` i distingeix feines pel camp `job_type` (`process_match` o `process_labeling`)
- `sc-inference-worker` escolta **dues cues simultàniament** amb `BLPOP ['task_frames', 'model_promoted']`. `task_frames` té prioritat
- `sc-label-studio` és **opcional**. No és dependència de cap altre servei. S'aixeca amb `docker compose up sc-label-studio`
- Cap servei es comunica per HTTP directe excepte via `sc-api-gateway`. La resta usa Redis (cues) o MinIO (fitxers)

### FastAPI — estructura de 3 capes obligatòria
Tots els serveis Python segueixen aquesta estructura:
```
app/
├── main.py          ← entrypoint
├── config.py        ← pydantic Settings
├── dependencies.py
├── schemas/         ← CAPA 1: Pydantic, cap lògica
├── repositories/    ← CAPA 2: accés a dades (MongoDB, Redis, MinIO)
├── services/        ← CAPA 3: lògica de negoci
└── routers/         ← endpoints (criden services/, mai repositories/)
```
Els `routers/` **mai** criden directament els `repositories/`. Tot passa per `services/`.

### Base de dades
- Dues bases de dades MongoDB separades: `sc-auth-db` (identitat) i `sc-app-db` (negoci)
- `sc-api-gateway` usa dos clients Motor independents: `auth_db` i `app_db`
- `players` i `matches` tenen `team_id` obligatori. L'índex únic de `players` és `team_id + dorsal` (no `dorsal` sol)
- El camp `status` de `matches` accepta: `pending`, `processing`, `frames_ready`, `done`, `error`
- El primer usuari admin es crea via `services/sc-mongodb/init/01-init.js` llegint variables d'entorn `ADMIN_EMAIL`, `ADMIN_PASSWORD`. Usa `force_reset: true` perquè el hash el fa l'API en el primer login

### Redis — payloads obligatoris
Tots els missatges Redis són JSON. Formats definits al punt 2.15 de l'spec:
- `video_to_process` → camp obligatori `job_type`: `"process_match"` o `"process_labeling"`
- `task_frames` → `match_id`, `frame_id`, `minio_bucket`, `minio_key`, `frame_number`, `timestamp_s`
- `detected_frames_results` → `match_id`, `frame_id`, `frame_number`, `timestamp_s`, `detections[]`
- `model_promoted` → `model_type`, `version`, `minio_bucket`, `minio_key`, `metrics`
- Mai reencuar un missatge erroni → descartar i logar a Sentry

### MinIO
- Tot l'storage passa per MinIO. Cap carpeta compartida entre contenidors
- 9 buckets: `raw-videos`, `pending-frames`, `processed-frames`, `processed-videos`, `feedback-data`, `models`, `labeling-videos`, `labeling-frames`, `datasets`
- Claus: `{match_id}/frame_000001.jpg` (zero-padded 6 dígits)
- Models candidats que no superen el llindar → `models/yolo/candidate/v{N}.pt` (mai es promouen automàticament)

### Seguretat
- `JWT_SECRET` mínim 32 bytes. Mai hardcoded
- Refresh Token Rotation: cada `/auth/refresh` invalida el token anterior
- `X-Internal-API-Key` a totes les peticions entre microserveis
- CORS només actiu quan `API_ENV=development`
- Filtre de queries per `team_id` per a rols `coach`, `assistant`, `player`. Admin veu tot sense filtre

### Endpoints
- Prefix: `/api/v1/{recurs}` en plural i minúscules
- Autenticació: `/auth/login`, `/auth/refresh`, `/auth/logout` (sense prefix `/api/v1/`)
- Sistema: `GET /health` → `{"status": "ok"}` (sense prefix, usat pels healthchecks)
- Format errors: estàndard FastAPI `{"detail": "..."}`. Mai exposar stack traces al client

### Docker
- Healthchecks amb `condition: service_healthy` a tots els `depends_on`
- Ordre d'arrencada: MongoDB+Redis+MinIO → API Gateway → Workers → Frontend → Observabilitat
- Política de logs: `max-size: 10m`, `max-file: 5` a tots els contenidors via àncora YAML
- Dozzle munta `/var/run/docker.sock` en mode `read-only`

### Codi
- Python: `ruff` per a linting
- TypeScript/React: `eslint` + `prettier`
- Imports Python: absoluts (`from app.services.match_service import ...`), mai relatius amb `..`
- Fitxers: `snake_case` Python, `PascalCase` React components, `kebab-case` configs
- Logs: format JSON a stdout. `ERROR`/`CRITICAL` → Sentry. Funció `setup_logging()` a cada `main.py`

---

## Jira — convencions

- Etiquetes: `backend`, `frontend`, `ai`, `infra`, `docs`
- Prioritats: `Alta`, `Mitja`, `Baixa`
- Estats: `To Do → In Progress → In Review → Done`
- Títols: `[ETIQUETA] Descripció en infinitiu`
- Èpiques: EP-01 Infraestructura · EP-02 Autenticació · EP-03 Partits · EP-04 Pipeline IA · EP-05 Frontend · EP-06 Active Learning · EP-07 Observabilitat

---

## Variables d'entorn

Cada servei té `services/{nom}/.env` (exclòs de git) i `services/{nom}/.env.example` (al repo).
Secrets marcats amb `# SECRET`. Mai hardcoded al codi.
Detall complet de cada `.env` al punt 2.13 de l'spec.

## Protocol de versionat ("Version X.Y")

Quan l'usuari digui **"Version X.Y"**, **"v1.2.3"**, **"Release 2.0"** o similar, seguir el protocol SemVer:

### MINOR (Y canvia: v1.0.x → v1.1.0) — nova funcionalitat ← cas més habitual
```bash
git checkout main && git merge develop --ff-only
git tag -a vX.Y.0 -m "Release vX.Y.0"
git push origin main && git push origin vX.Y.0
git checkout develop && git merge main && git push origin develop
```

### MAJOR (X canvia: v1.x.x → v2.0.0) — canvi trencador
```bash
git checkout -b release/vX.0.0 develop
git checkout main && git merge release/vX.0.0 --no-ff
git tag -a vX.0.0 -m "Release vX.0.0"
git push origin main && git push origin vX.0.0
git branch -d release/vX.0.0 && git push origin --delete release/vX.0.0
git checkout develop && git merge main && git push origin develop
```

### PATCH (Z canvia: v1.0.0 → v1.0.1) — bugfix
```bash
git checkout -b hotfix/vX.Y.Z main
# (aplicar el fix)
git checkout main && git merge hotfix/vX.Y.Z --no-ff
git tag -a vX.Y.Z -m "Hotfix vX.Y.Z"
git push origin main && git push origin vX.Y.Z
git checkout develop && git merge main && git push origin develop
git branch -d hotfix/vX.Y.Z && git push origin --delete hotfix/vX.Y.Z
```

**Formats acceptats:** "Version 1.0" → `v1.0.0` | "v2.1.3" → `v2.1.3` | "Release 1.5" → `v1.5.0`