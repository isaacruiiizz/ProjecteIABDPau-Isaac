# Pla d'Implementació — Dockerfiles base per a tots els serveis

**Ticket:** PJM-55
**Data:** 2026-03-20
**Sprint:** Sprint 1 — Fonaments
**Estat:** Completat ✓ (2026-03-20)

---

## 1. Objectiu

Crear els `Dockerfile` base per als 6 serveis que no usen imatge oficial directament:
- **5 serveis Python** (`sc-api-gateway`, `sc-video-manager`, `sc-inference-worker`, `sc-logic-aggregator`, `sc-active-learner`)
- **1 servei frontend** (`sc-frontend`)

Els 7 serveis restants (`sc-mongodb`, `sc-redis`, `sc-object-storage`, `sc-prometheus`, `sc-grafana`, `sc-dozzle`, `sc-label-studio`) usen imatge oficial al `docker-compose.yml` i **no necessiten `Dockerfile`**.

---

## 2. Fitxers afectats

| Fitxer | Acció |
|--------|-------|
| `services/sc-api-gateway/Dockerfile` | **Crear** |
| `services/sc-api-gateway/requirements.txt` | **Crear** (buit, placeholder) |
| `services/sc-api-gateway/app/__init__.py` | **Crear** (buit) |
| `services/sc-video-manager/Dockerfile` | **Crear** |
| `services/sc-video-manager/requirements.txt` | **Crear** (buit, placeholder) |
| `services/sc-video-manager/app/__init__.py` | **Crear** (buit) |
| `services/sc-inference-worker/Dockerfile` | **Crear** |
| `services/sc-inference-worker/requirements.txt` | **Crear** (buit, placeholder) |
| `services/sc-inference-worker/app/__init__.py` | **Crear** (buit) |
| `services/sc-logic-aggregator/Dockerfile` | **Crear** |
| `services/sc-logic-aggregator/requirements.txt` | **Crear** (buit, placeholder) |
| `services/sc-logic-aggregator/app/__init__.py` | **Crear** (buit) |
| `services/sc-active-learner/Dockerfile` | **Crear** |
| `services/sc-active-learner/requirements.txt` | **Crear** (buit, placeholder) |
| `services/sc-active-learner/app/__init__.py` | **Crear** (buit) |
| `services/sc-frontend/Dockerfile` | **Crear** |
| `CLAUDE.md` | **Actualitzar** — afegir els nous directoris a l'apartat d'estructura |

Els `requirements.txt` i `app/__init__.py` són placeholders mínims per permetre que `docker compose build` funcioni. El contingut real arriba amb els tickets d'implementació de cada servei (PJM-18, etc.).

---

## 3. Contingut dels Dockerfiles

### 3.1 Serveis Python genèrics — `python:3.11-slim`

S'aplica a: `sc-api-gateway`, `sc-logic-aggregator`, `sc-active-learner`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> `curl` necessari per als healthchecks de Docker (únicament `sc-api-gateway` en té).
> `sc-logic-aggregator` i `sc-active-learner` usen `python -m app.main` (worker, sense Uvicorn).

**Variació per a workers** (`sc-logic-aggregator`, `sc-active-learner`):
```dockerfile
CMD ["python", "-m", "app.main"]
```

---

### 3.2 `sc-video-manager` — Python + FFmpeg + OpenCV

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["python", "-m", "app.main"]
```

> `ffmpeg`: extracció i muntatge de vídeo.
> `libgl1` + `libglib2.0-0`: dependències de sistema necessàries per a OpenCV.

---

### 3.3 `sc-inference-worker` — CUDA 12.6 + Python 3.11

```dockerfile
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["python3", "-m", "app.main"]
```

> Ubuntu 22.04 inclou `python3.11` als repositoris oficials.
> `ln -sf` assegura que `python3` apunta a 3.11 (Ubuntu 22.04 té 3.10 per defecte).
> `libgl1` + `libglib2.0-0`: necessàries per a OpenCV + Ultralytics.

---

### 3.4 `sc-frontend` — Build multistage Node 22

```dockerfile
# ── Etapa 1: Build ─────────────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

COPY package*.json .
RUN npm ci

COPY . .
RUN npm run build

# ── Etapa 2: Runtime ───────────────────────────────────────────────────────
FROM node:22-alpine

WORKDIR /app

RUN npm install -g serve

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
```

> `npm ci` (en lloc de `npm install`) per instal·lació reproductible i determinista.
> `serve` és un servidor HTTP lleuger de Node per servir fitxers estàtics Vite.
> Port 3000 coincideix amb el `docker-compose.yml` existent (`"3000:3000"`).

---

## 4. Decisions tècniques

### D1 — Imatge base Python: `python:3.11-slim`
**Decisió:** Usar `python:3.11-slim` per a tots els serveis Python excepte l'inference-worker.
**Motiu:** Equilibri entre mida d'imatge (petita) i compatibilitat. La variant `slim` inclou les eines mínimes necessàries.
**Alternativa descartada:** `python:3.11-alpine` — incompatible amb moltes dependències científiques (NumPy, OpenCV, PyTorch) que requereixen glibc.

### D2 — Imatge base CUDA: `nvidia/cuda:12.6.3-runtime-ubuntu22.04`
**Decisió:** Usar CUDA 12.6.3 (última versió estable 12.x) sobre Ubuntu 22.04.
**Motiu:** L'especificació (CLAUDE.md) indica `nvidia/cuda:12.x`. La 12.6.3 és la versió 12.x més recent sense problemes de seguretat coneguts. Ubuntu 22.04 té python3.11 als repositoris oficials sense PPA addicionals.
**Nota vs spec:** `docs/specs.md` punt 2.2 menciona "CUDA 11.8" però la instrucció explícita de l'usuari (CLAUDE.md) especifica 12.x, que té precedència.

### D3 — Frontend: `serve` vs `nginx`
**Decisió:** Usar `node:22-alpine` + `serve` com a runtime del frontend.
**Motiu:** Evita afegir una dependència externa (`nginx`) i un `nginx.conf` extra. El `docker-compose.yml` ja defineix el port `3000:3000` i `serve` l'usa directament sense configuració addicional.
**Alternativa descartada:** `nginx:alpine` — requereix fitxer `nginx.conf` per mapear al port 3000 (nginx escolta al 80 per defecte), i comportaria modificar el `docker-compose.yml` ja tancat (PJM-14 Done).

### D4 — Dependències de sistema per OpenCV/Ultralytics
**Decisió:** Instal·lar `libgl1` + `libglib2.0-0` als serveis que usen OpenCV (`sc-video-manager`, `sc-inference-worker`).
**Motiu:** OpenCV en imatge `slim` o Ubuntu pura falla sense aquestes biblioteques. És el mínim necessari per a `import cv2`.

### D5 — Placeholders per a `requirements.txt` i `app/__init__.py`
**Decisió:** Crear fitxers buits per permetre `docker compose build` sense errors.
**Motiu:** El `docker-compose.yml` (PJM-14) fa `build: context: ./services/sc-{nom}` i espera que existeixin els fitxers referencials pel `COPY`. Els continguts reals arriben amb els tickets d'implementació corresponents.

---

## 5. Riscos

| Risc | Probabilitat | Impacte | Mitigació |
|------|-------------|---------|-----------|
| CUDA 12.6 incompatible amb PyTorch wheel disponible | Baixa | Inference-worker no build | Ajustar versió de PyTorch al `requirements.txt` quan s'implementi |
| `python3.11` no disponible a Ubuntu 22.04 repos sense PPA | Molt baixa | Inference-worker no build | python3.11 és a `universe` repo, habitualment disponible |
| `npm ci` falla sense `package-lock.json` | Alta (no existeix encara) | Frontend no build | Documentat: el frontend necessita `package-lock.json` per buildar |
| `libgl1` no suficient per a tots els casos OpenCV | Baixa | ImportError en runtime | Ampliar les deps de sistema si apareix l'error |

---

## 6. Ordre d'implementació

Els Dockerfiles no tenen dependències entre ells. Es creen en aquest ordre per grup:

1. `sc-api-gateway` (Python + curl, Uvicorn)
2. `sc-video-manager` (Python + FFmpeg + OpenCV)
3. `sc-logic-aggregator` (Python worker)
4. `sc-active-learner` (Python worker)
5. `sc-inference-worker` (CUDA)
6. `sc-frontend` (multistage Node)
7. Actualitzar `CLAUDE.md` — estructura de fitxers

**Verificació:** `docker compose build --no-cache` per confirmar que tots els Dockerfiles compilen correctament (fallaran en runtime si els `requirements.txt` estan buits, però el build ha de passar).

---

## 7. Fora d'abast d'aquest ticket

- Contingut real dels `requirements.txt` → tickets d'implementació de cada servei
- Estructura interna `app/` (schemas/, repositories/, services/, routers/) → PJM-18 i successors
- Fitxers `.env.example` → PJM-15
- `package.json` del frontend → PJM-20
