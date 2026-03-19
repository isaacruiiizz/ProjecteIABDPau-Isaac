# Pla d'Implementació — docker-compose.yml

**Ticket:** PJM-14
**Data:** 2026-03-19
**Sprint:** Sprint 1 — Fonaments
**Estat:** Completat ✓ (2026-03-20)

---

## 1. Objectiu

Crear el fitxer `docker-compose.yml` arrel del monorepo amb tots els 13 serveis de SmartChrono IP correctament configurats: xarxes, healthchecks, ordre d'arrencada, política de logs i variables d'entorn via `env_file`.

---

## 2. Fitxers afectats

| Fitxer | Acció |
|--------|-------|
| `docker-compose.yml` | **Crear** — fitxer principal |
| `docs/decisions.md` | **Actualitzar** — registrar decisions tècniques |

Cap altre fitxer de codi es toca en aquest ticket. Els `.env` i `.env.example` són responsabilitat del ticket PJM-15.

---

## 3. Estructura del docker-compose.yml

### 3.1 Àncora de logs (top-level extension)

```yaml
x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "5"
```

Tots els serveis heretaran: `logging: *default-logging`

### 3.2 Xarxes (4 xarxes isolades — punt 2.8 de l'spec)

```yaml
networks:
  sc-frontend-net:
    driver: bridge
  sc-backend-net:
    driver: bridge
    internal: true
  sc-ai-net:
    driver: bridge
    internal: true
  sc-observability-net:
    driver: bridge
    internal: true
```

### 3.3 Serveis — resum per grup

#### Grup 1: Infraestructura base (arrenquen en paral·lel, sense dependències)

| Servei | Imatge | Ports host | Xarxes | Healthcheck |
|--------|--------|------------|--------|-------------|
| `sc-mongodb` | `mongo:8` | cap | `sc-backend-net` | `mongosh --eval "db.adminCommand('ping')"` |
| `sc-redis` | `redis:7-alpine` | cap | `sc-backend-net`, `sc-ai-net` | `redis-cli ping` |
| `sc-object-storage` | `minio/minio:latest` | `9000` (opcional) | `sc-backend-net`, `sc-ai-net` | `curl -f http://localhost:9000/minio/health/live` |

#### Grup 2: API Gateway (espera grup 1)

| Servei | Imatge | Ports host | Xarxes | depends_on |
|--------|--------|------------|--------|------------|
| `sc-api-gateway` | `./services/sc-api-gateway` (build) | `8000` | `sc-frontend-net`, `sc-backend-net` | mongodb (healthy), redis (healthy), object-storage (healthy) |

Healthcheck: `curl -f http://localhost:8000/health`

#### Grup 3: Workers (esperen redis + object-storage)

| Servei | Imatge | Ports | Xarxes | depends_on |
|--------|--------|-------|--------|------------|
| `sc-video-manager` | `./services/sc-video-manager` (build) | cap | `sc-backend-net` | redis (healthy), object-storage (healthy) |
| `sc-logic-aggregator` | `./services/sc-logic-aggregator` (build) | cap | `sc-backend-net` | redis (healthy), object-storage (healthy) |
| `sc-active-learner` | `./services/sc-active-learner` (build) | cap | `sc-backend-net` | redis (healthy), object-storage (healthy) |
| `sc-inference-worker` | `./services/sc-inference-worker` (build) | cap | `sc-ai-net` | redis (healthy), object-storage (healthy) |

> `sc-inference-worker` s'afegeix al grup gpu i necessita `deploy.resources.reservations.devices` per a NVIDIA CUDA.

#### Grup 4: Frontend (espera api-gateway)

| Servei | Imatge | Ports host | Xarxes | depends_on |
|--------|--------|------------|--------|------------|
| `sc-frontend` | `./services/sc-frontend` (build) | `3000` | `sc-frontend-net` | api-gateway (healthy) |

#### Grup 5: Observabilitat (arrenquen en paral·lel)

| Servei | Imatge | Ports host | Xarxes | Nota |
|--------|--------|------------|--------|------|
| `sc-prometheus` | `prom/prometheus:latest` | cap | `sc-observability-net` | — |
| `sc-grafana` | `grafana/grafana:latest` | `3001` | `sc-observability-net` | — |
| `sc-dozzle` | `amir20/dozzle:latest` | `8080` | `sc-frontend-net`, `sc-observability-net` | munta `/var/run/docker.sock:read-only` |

#### Servei opcional (no s'inicia amb `docker compose up`)

| Servei | Imatge | Ports host | Xarxes | Nota |
|--------|--------|------------|--------|------|
| `sc-label-studio` | `heartexlabs/label-studio:latest` | `8081` | `sc-frontend-net`, `sc-backend-net` | `profiles: ["labeling"]` |

> Usant `profiles: ["labeling"]` el servei NO s'inicia amb `docker compose up`. S'inicia explícitament amb `docker compose --profile labeling up sc-label-studio`.

### 3.4 Volums

```yaml
volumes:
  sc-mongodb-data:
  sc-redis-data:
  sc-minio-data:
  sc-grafana-data:
  sc-prometheus-data:
  sc-label-studio-data:
```

### 3.5 Variables d'entorn

Cada servei usarà `env_file: ./services/{nom}/.env` (creat per PJM-15). El `docker-compose.yml` **no** inclou cap variable d'entorn hardcoded — tot via `env_file`.

Excepció: `sc-mongodb` usa `environment:` per als seeds d'inicialització (MONGO_INITDB_ROOT_USERNAME/PASSWORD), llegits de l'`env_file`.

---

## 4. Decisions tècniques

### D1 — `profiles` per a sc-label-studio
**Decisió:** Usar `profiles: ["labeling"]` en comptes d'un `docker-compose.label-studio.yml` separat.
**Motiu:** Més net, evita duplicació de xarxes. L'usuari executa `docker compose --profile labeling up sc-label-studio`.
**Alternativa descartada:** Fitxer `docker-compose.override.yml` — afegeix complexitat innecessària.

### D2 — `internal: true` per a xarxes de backend
**Decisió:** Marcar `sc-backend-net`, `sc-ai-net` i `sc-observability-net` com a `internal: true`.
**Motiu:** Impedeix que els contenidors interns facin peticions sortints a Internet. Principi de mínim privilegi.
**Excepció:** `sc-frontend-net` no és `internal` perquè el frontend ha de ser accessible des del navegador.

### D3 — Healthcheck de sc-object-storage
**Decisió:** Usar `curl -f http://localhost:9000/minio/health/live` (endpoint natiu MinIO).
**Motiu:** Definit explícitament a la spec (punt 2.14). MinIO exposa aquest endpoint per a healthchecks.

### D4 — GPU per a sc-inference-worker
**Decisió:** Usar `deploy.resources.reservations.devices` amb `driver: nvidia` i `count: 1`.
**Motiu:** Forma estàndard Docker Compose per a GPU NVIDIA. Requereix `nvidia-container-toolkit` a l'host.
**Risc:** Si l'host no té GPU NVIDIA, el servei fallarà en arrencar. Documentar al README.

### D5 — Imatges base
**Decisió:** Usar últimes versions estables amb tag explícit (no `:latest` per a serveis crítics).
- MongoDB: `mongo:8`
- Redis: `redis:7-alpine`
- MinIO: `minio/minio` (tag de release actual)
- Prometheus: `prom/prometheus:v3`
- Grafana: `grafana/grafana:11`
- Dozzle: `amir20/dozzle:latest` (Dozzle és safe amb latest)

---

## 5. Riscos

| Risc | Probabilitat | Impacte | Mitigació |
|------|-------------|---------|-----------|
| Host sense NVIDIA GPU | Alta (dev local) | Worker no arrenca | Afegir `required: false` al device o comentar el bloc GPU en dev |
| Versions de MongoDB incompatibles amb Motor | Baixa | API falla en init | Usar `mongo:8` confirmat compatible amb Motor 3.x |
| `internal: true` bloqueja pulls d'imatges | — | No aplica | Les imatges es descarreguen en build time, no en runtime |
| sc-label-studio amb perfil oblidat | Baixa | Confusió | Documentar clarament a README i al CLAUDE.md |

---

## 6. Ordre d'implementació

1. Esquelet yaml amb x-logging i networks
2. Serveis grup 1 (mongodb, redis, object-storage) + healthchecks
3. Serveis grup 2 (api-gateway) + depends_on
4. Serveis grup 3 (workers) + depends_on + GPU bloc
5. Servei grup 4 (frontend) + depends_on
6. Serveis grup 5 (observabilitat) + Dozzle socket
7. Servei opcional (label-studio) + profile
8. Volums top-level

**Verificació:** `docker compose config` per validar la sintaxi YAML abans de fer cap `up`.

---

## 7. Fora d'abast d'aquest ticket

- Creació dels `Dockerfile` de cada servei → **PJM-55** (Sprint 1, depèn de PJM-14)
- Fitxers `.env` i `.env.example` (PJM-15)
- Configuració de Prometheus (`prometheus.yml`) (PJM-47)
- Script d'inicialització de MongoDB (PJM-18)
- Configuració de MinIO buckets (PJM-16)
