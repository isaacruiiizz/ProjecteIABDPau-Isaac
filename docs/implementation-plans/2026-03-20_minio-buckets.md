# Pla d'Implementació — MinIO: buckets, polítiques i lifecycle

**Ticket:** PJM-16
**Data:** 2026-03-20
**Sprint:** Sprint 1 — Fonaments
**Estat:** Completat ✓ (2026-03-21)

---

## 1. Objectiu

Configurar `sc-object-storage` (MinIO) amb:
- Els 9 buckets exactes definits a la spec (punt 2.9 / `docs/specs/03-infraestructura.md`)
- Polítiques d'accés IAM per servei — cada servei només accedeix als buckets que li corresponen
- Regles de lifecycle (retenció) per bucket
- Un usuari MinIO per servei (evitar credencial compartida global)
- Un servei init al `docker-compose.yml` que executa el script en arrancar

---

## 2. Fitxers afectats

| Acció | Fitxer |
|-------|--------|
| Crear | `services/sc-object-storage/init/create-buckets.sh` |
| Crear | `services/sc-object-storage/init/policies/sc-api-gateway.json` |
| Crear | `services/sc-object-storage/init/policies/sc-video-manager.json` |
| Crear | `services/sc-object-storage/init/policies/sc-inference-worker.json` |
| Crear | `services/sc-object-storage/init/policies/sc-active-learner.json` |
| Crear | `services/sc-object-storage/init/policies/sc-label-studio.json` |
| Modificar | `docker-compose.yml` — afegir servei `sc-minio-init` |
| Modificar | `.env.example` dels 5 serveis amb accés MinIO — credencials per servei |
| Actualitzar | `CLAUDE.md` — estructura de fitxers |

---

## 3. Els 9 buckets i les seves polítiques

### Taula de buckets (d'`docs/specs/03-infraestructura.md`, secció 5)

| Bucket | Escriu | Llegeix | Retenció |
|--------|--------|---------|----------|
| `raw-videos` | `sc-api-gateway` | `sc-video-manager` | 30 dies |
| `pending-frames` | `sc-video-manager` | `sc-inference-worker` | 7 dies |
| `processed-frames` | `sc-video-manager` | `sc-video-manager` | 7 dies |
| `processed-videos` | `sc-video-manager` | `sc-api-gateway` | Indefinida |
| `feedback-data` | `sc-inference-worker` | `sc-active-learner` | Indefinida |
| `models` | `sc-active-learner` | `sc-inference-worker` | Indefinida |
| `labeling-videos` | `sc-api-gateway` | `sc-video-manager` | 30 dies |
| `labeling-frames` | `sc-video-manager` | `sc-label-studio` | 30 dies |
| `datasets` | `sc-label-studio` | `sc-active-learner` | Indefinida |

### Accés consolidat per servei

| Servei | Buckets — escriptura | Buckets — lectura |
|--------|---------------------|-------------------|
| `sc-api-gateway` | `raw-videos`, `labeling-videos` | `processed-videos` |
| `sc-video-manager` | `pending-frames`, `processed-frames`, `processed-videos`, `labeling-frames` | `raw-videos`, `labeling-videos`, `pending-frames`* |
| `sc-inference-worker` | `feedback-data` | `pending-frames`, `models` |
| `sc-active-learner` | `models`, `datasets`** | `feedback-data`, `models`, `datasets` |
| `sc-label-studio` | `datasets` | `labeling-frames` |

> *`sc-video-manager` llegeix els seus propis frames de `pending-frames` per muntar l'overlay i el vídeo final.
> **`sc-active-learner` escriu el dataset exportat quan actua com a pont (no és Label Studio).

---

## 4. Contingut de cada fitxer

### `services/sc-object-storage/init/create-buckets.sh`

```bash
#!/usr/bin/env sh
set -e

ALIAS="minio"
ENDPOINT="http://sc-object-storage:9000"

# 1. Connectar amb credencials root
mc alias set "$ALIAS" "$ENDPOINT" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# 2. Crear els 9 buckets
for bucket in raw-videos pending-frames processed-frames processed-videos \
              feedback-data models labeling-videos labeling-frames datasets; do
  mc mb --ignore-existing "$ALIAS/$bucket"
done

# 3. Lifecycle — buckets amb retenció de 30 dies
for bucket in raw-videos labeling-videos labeling-frames; do
  mc ilm add --expiry-days 30 "$ALIAS/$bucket"
done

# 4. Lifecycle — buckets amb retenció de 7 dies
for bucket in pending-frames processed-frames; do
  mc ilm add --expiry-days 7 "$ALIAS/$bucket"
done

# 5. Crear usuaris IAM per servei
mc admin user add "$ALIAS" sc-api-gateway      "$SC_API_GATEWAY_MINIO_PASSWORD"
mc admin user add "$ALIAS" sc-video-manager    "$SC_VIDEO_MANAGER_MINIO_PASSWORD"
mc admin user add "$ALIAS" sc-inference-worker "$SC_INFERENCE_WORKER_MINIO_PASSWORD"
mc admin user add "$ALIAS" sc-active-learner   "$SC_ACTIVE_LEARNER_MINIO_PASSWORD"
mc admin user add "$ALIAS" sc-label-studio     "$SC_LABEL_STUDIO_MINIO_PASSWORD"

# 6. Carregar i assignar polítiques
for service in sc-api-gateway sc-video-manager sc-inference-worker sc-active-learner sc-label-studio; do
  mc admin policy create "$ALIAS" "policy-$service" "/init/policies/$service.json"
  mc admin policy attach "$ALIAS" "policy-$service" --user "$service"
done

echo "MinIO init completat: 9 buckets + lifecycle + 5 usuaris IAM"
```

---

### Polítiques IAM (JSON per servei)

#### `sc-api-gateway.json`
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::raw-videos/*",
        "arn:aws:s3:::labeling-videos/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::processed-videos/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::raw-videos",
        "arn:aws:s3:::labeling-videos",
        "arn:aws:s3:::processed-videos"
      ]
    }
  ]
}
```

#### `sc-video-manager.json`
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::raw-videos/*",
        "arn:aws:s3:::labeling-videos/*",
        "arn:aws:s3:::pending-frames/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::pending-frames/*",
        "arn:aws:s3:::processed-frames/*",
        "arn:aws:s3:::processed-videos/*",
        "arn:aws:s3:::labeling-frames/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::raw-videos",
        "arn:aws:s3:::labeling-videos",
        "arn:aws:s3:::pending-frames",
        "arn:aws:s3:::processed-frames",
        "arn:aws:s3:::processed-videos",
        "arn:aws:s3:::labeling-frames"
      ]
    }
  ]
}
```

#### `sc-inference-worker.json`
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::pending-frames/*",
        "arn:aws:s3:::models/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": ["arn:aws:s3:::feedback-data/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::pending-frames",
        "arn:aws:s3:::models",
        "arn:aws:s3:::feedback-data"
      ]
    }
  ]
}
```

#### `sc-active-learner.json`
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::feedback-data/*",
        "arn:aws:s3:::models/*",
        "arn:aws:s3:::datasets/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::models/*",
        "arn:aws:s3:::datasets/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::feedback-data",
        "arn:aws:s3:::models",
        "arn:aws:s3:::datasets"
      ]
    }
  ]
}
```

#### `sc-label-studio.json`
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::labeling-frames/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:DeleteObject"],
      "Resource": ["arn:aws:s3:::datasets/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::labeling-frames",
        "arn:aws:s3:::datasets"
      ]
    }
  ]
}
```

---

### Canvis a `docker-compose.yml`

Afegir un servei init que executa el script un cop MinIO és healthy:

```yaml
sc-minio-init:
  image: minio/mc:latest
  container_name: sc-minio-init
  depends_on:
    sc-object-storage:
      condition: service_healthy
  volumes:
    - ./services/sc-object-storage/init:/init:ro
  env_file:
    - ./services/sc-object-storage/.env
  entrypoint: ["/bin/sh", "/init/create-buckets.sh"]
  networks:
    - sc-backend-net
  restart: "no"
  logging: *default-logging
```

---

### Canvis a `.env.example` dels 5 serveis

Cada servei passa de tenir `MINIO_ACCESS_KEY=change_me_minio_user` a tenir el seu usuari IAM fix i una contrasenya específica:

**Exemple per a `sc-api-gateway/.env.example`:**
```env
MINIO_ACCESS_KEY=sc-api-gateway
MINIO_SECRET_KEY=change_me_sc_api_gateway_minio_password  # SECRET
```

A `services/sc-object-storage/.env.example`, afegir les contrasenyes de cada servei:
```env
# Passwords per als usuaris IAM de cada servei   # SECRET (tots)
SC_API_GATEWAY_MINIO_PASSWORD=change_me_sc_api_gateway_password
SC_VIDEO_MANAGER_MINIO_PASSWORD=change_me_sc_video_manager_password
SC_INFERENCE_WORKER_MINIO_PASSWORD=change_me_sc_inference_worker_password
SC_ACTIVE_LEARNER_MINIO_PASSWORD=change_me_sc_active_learner_password
SC_LABEL_STUDIO_MINIO_PASSWORD=change_me_sc_label_studio_password
```

---

## 5. Decisions tècniques

### D1 — Un usuari MinIO per servei (no credencial compartida)
**Decisió:** Cada servei té el seu propi usuari IAM a MinIO en lloc d'un admin compartit.
**Motiu:** Compleix el requisit "polítiques restrictives" de la spec. Un servei comprometit no pot accedir als buckets d'un altre. La credencial `minioadmin` del root només s'usa per a l'init.

### D2 — Script sh amb mc, no Python
**Decisió:** El script d'init és shell (`mc`), no Python.
**Motiu:** `minio/mc:latest` és la imatge oficial per a operacions d'admin de MinIO. No requereix cap dependència addicional i és idiomàtic.

### D3 — Servei `sc-minio-init` separat al compose
**Decisió:** Servei init amb `restart: "no"` que s'executa un cop i para.
**Motiu:** Permet re-executar manualment si cal (`docker compose run sc-minio-init`), sense que el compose el reiniciï indefinidament. El `depends_on: service_healthy` garanteix que MinIO és operatiu.

### D4 — `s3:DeleteObject` restringit
**Decisió:** Només `sc-api-gateway` (per cancel·lació de partits) i `sc-video-manager` (cleanup de frames temporals) reben `DeleteObject`. La resta és `PutObject` i `GetObject`.
**Motiu:** Mínim privilegi. Un worker d'inferència no ha de poder esborrar frames de producció.

---

## 6. Riscos

| Risc | Probabilitat | Impacte | Mitigació |
|------|-------------|---------|-----------|
| `mc ilm add` falla per sintaxi de MinIO vell | Baixa | Lifecycle no actiu | Verificar versió de `minio/mc` compatible amb el server |
| Contrasenya IAM massa curta rebutjada per MinIO | Baixa | Init falla | MinIO requereix mínim 8 chars — `change_me_*` ho compleix |
| Script re-executat crea usuaris duplicats | Mitja | Error a init | `mc admin user add` falla si existeix → afegir check o `|| true` |

---

## 7. Ordre d'implementació

1. Crear `services/sc-object-storage/init/policies/` amb els 5 JSON
2. Crear `services/sc-object-storage/init/create-buckets.sh`
3. Actualitzar `services/sc-object-storage/.env.example` (passwords IAM)
4. Actualitzar `.env.example` dels 5 serveis (MINIO_ACCESS_KEY per servei)
5. Modificar `docker-compose.yml` (afegir `sc-minio-init`)
6. Actualitzar `CLAUDE.md` (estructura de fitxers)

---

## 8. Fora d'abast

- Configuració de MinIO en mode distribuït (clustered) → producció futura
- Backup dels buckets → fora d'abast del projecte acadèmic
- Accés públic/presigned URLs → s'implementarà als routers de l'API (PJM-18+)
