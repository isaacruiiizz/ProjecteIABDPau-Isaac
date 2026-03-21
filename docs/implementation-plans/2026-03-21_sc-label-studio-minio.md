# Pla d'Implementació — PJM-17
# [infra] Configurar sc-label-studio amb integració S3/MinIO

**Estat:** Completat ✓ (2026-03-22)
**Ticket:** PJM-17
**Data:** 2026-03-21
**Etiqueta commit:** `feat(infra): configurar sc-label-studio amb integració S3/MinIO [PJM-17]`

---

## Context

`sc-label-studio` és un servei **opcional** per etiquetar frames de vídeo (bounding boxes de jugadors) per al model YOLO. S'integra amb MinIO per:
- **Llegir** frames del bucket `labeling-frames`
- **Exportar** anotacions al bucket `datasets`

S'aixeca manualment amb `docker compose up sc-label-studio`. Mai s'inclou en `docker compose up` general.

---

## Estat actual

| Fitxer | Estat |
|---|---|
| `services/sc-label-studio/.env.example` | Ja existeix (PJM-15) |
| `services/sc-label-studio/init/setup-project.sh` | Falta crear |
| `docker-compose.yml` (bloc `sc-label-studio`) | Parcial — falta healthcheck, depends_on MinIO, init container |

**Problemes detectats al docker-compose actual:**
- El servei no té `healthcheck` — no es pot fer dependre `sc-label-studio-init` d'ell
- No té `depends_on: sc-object-storage` — Label Studio pot arrencar abans que MinIO estigui sa
- Falta el servei `sc-label-studio-init` que executi `setup-project.sh`

---

## Fitxers a crear / modificar

### 1. `services/sc-label-studio/init/setup-project.sh` — NOU

Script bash executat pel container `sc-label-studio-init` un cop Label Studio és sa. Fa:

1. Espera activa fins que l'API de Label Studio respon a `/health` (retry amb backoff)
2. Obté el token d'API autenticant-se amb `LABEL_STUDIO_USERNAME` + `LABEL_STUDIO_PASSWORD`
3. Comprova si ja existeix el projecte (idempotent — si existeix, no el recrea)
4. Crea el projecte `SmartChrono — Etiquetatge de Jugadors` amb la plantilla XML:
   - `RectangleLabels` amb dues classes: `player_own` (el nostre equip) i `other` (rivals, àrbitres)
   - Sense cap TextArea ni classe de dorsal — els dorsals els llegeix la CNN, no Label Studio
5. Configura el **Source Storage** (S3 → bucket `labeling-frames`): frames a etiquetar
6. Configura l'**Export Storage** (S3 → bucket `datasets`): exportació d'anotacions

Variables d'entorn necessàries (injectades pel docker-compose des de `sc-label-studio/.env`):
- `LABEL_STUDIO_URL`, `LABEL_STUDIO_USERNAME`, `LABEL_STUDIO_PASSWORD`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- `MINIO_BUCKET_FRAMES`, `MINIO_BUCKET_DATASETS`

### 2. `docker-compose.yml` — bloc `sc-label-studio` — MODIFICAR

**Canvis al servei `sc-label-studio` existent:**
- Afegir `depends_on: sc-object-storage (service_healthy)` — garanteix que MinIO estigui sa
- Afegir `healthcheck` sobre `GET /health` de Label Studio al port 8081
- Afegir variable d'entorn `LABEL_STUDIO_URL=http://localhost:8081` (per l'init)
- Mantenir `profiles: [labeling]` i resta de configuració

**Nou servei `sc-label-studio-init`:**
- Imatge: `python:3.11-slim`
- `profiles: [labeling]`
- `restart: "no"`
- `depends_on: sc-label-studio (service_healthy)` + `sc-object-storage (service_healthy)`
- `entrypoint: ["sh", "-c", "pip install requests -q && python3 /init/setup-project.py"]`
- `env_file: ./services/sc-label-studio/.env`
- Variable addicional: `LABEL_STUDIO_URL=http://sc-label-studio:8081`
- Volume: `./services/sc-label-studio/init:/init:ro`
- Xarxes: `sc-frontend-net` (accés a LS) + `sc-backend-net` (accés a MinIO)
- `logging: *default-logging`

**Nota sobre l'script:** Per evitar dependències externes (`jq`, `curl`+parsing JSON), l'script
s'implementa en Python 3 (setup-project.py) usant la biblioteca `requests` estàndard.
La imatge `python:3.11-slim` el fa lleuger i no requereix Dockerfile propi.

---

## Plantilla XML del projecte

Label Studio s'usa **exclusivament per etiquetar el model YOLO** (detecció de jugadors).
Les classes són les dues que defineix l'spec (punt 2.2): `player_own` i `other`.

Els dorsals NO s'etiqueten aquí — els llegeix la CNN automàticament sobre crops extrets
pel YOLO durant la inferència. No cal cap `TextArea` ni cap classe de dorsal.

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="player_own" background="#ffb700"/>
    <Label value="other" background="#0074D9"/>
  </RectangleLabels>
</View>
```

---

## Fases d'implementació

### Fase 1 — Script d'inicialització
- Crear `services/sc-label-studio/init/setup-project.py`

### Fase 2 — docker-compose.yml
- Modificar el bloc `sc-label-studio` (healthcheck + depends_on + env)
- Afegir `sc-label-studio-init`

### Fase 3 — Actualitzar CLAUDE.md
- Reflectir `init/setup-project.py` a l'arbre de directoris

---

## Consideracions tècniques

- **Idempotència:** l'script comprova si el projecte ja existeix per nom abans de crear-lo.
  Segur de re-executar si el container es reinicia.
- **Port intern:** `LABEL_STUDIO_PORT=8081` — Label Studio llegeix aquesta variable per canviar
  el port d'escolta. El healthcheck i l'init apunten al port 8081.
- **Format endpoint MinIO per a Label Studio:** `http://sc-object-storage:9000` (sense `/` final).
  Label Studio usa l'endpoint S3 natiu de MinIO.
- **Xarxes del `sc-label-studio-init`:** necessita `sc-frontend-net` per accedir a Label Studio
  (port 8081 exposat a `sc-frontend-net`) i `sc-backend-net` per accedir a MinIO.
- **El servei `sc-label-studio` propi** necessita `sc-backend-net` per accedir a MinIO S3
  quan processa i desa anotacions. Ja té `sc-frontend-net` i `sc-backend-net` al compose.
