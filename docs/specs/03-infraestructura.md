# Infraestructura

## 1. Política de Logging i Monitorització

Per mantenir la consistència del sistema i facilitar el debugatge, s'aplica una política de **Logging Estructurat** obligatòria a tots els microserveis de Python:

* **Logs JSON:** Cap servei escriurà logs plans. S'utilitzarà un format JSON per permetre el parseig automàtic.
* **Dual Handler:**
  * `INFO`, `WARNING`, `DEBUG`: Enviats a **stdout** en format JSON (recollits per Dozzle).
  * `ERROR`, `CRITICAL`: Enviats a **Sentry** amb el rastreig complet de la pila (stack trace).

**Configuració base obligatòria** a tots els serveis Python:

```python
import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

def setup_logging(service_name: str, sentry_dsn: str = None):
    # Configuració de logs JSON per a stdout
    logging.basicConfig(
        level=logging.INFO,
        format='{"time": "%(asctime)s", "service": "' + service_name + '", "level": "%(levelname)s", "message": "%(message)s"}'
    )
    
    # Integració amb Sentry si el DSN està present
    if sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Captura logs d'info com a breadcrumbs
            event_level=logging.ERROR  # Envia errors com a esdeveniments
        )
        sentry_sdk.init(
            dsn=sentry_dsn,
            integrations=[sentry_logging],
            traces_sample_rate=1.0 # Monitorització de rendiment (performance)
        )
```

Aquest **setup_logging()** s'ha d'executar en l'entrypoint de cada servei (main.py) per assegurar la traçabilitat total des del minut zero del partit.

## 2. Gestió i Retenció de Logs

Per evitar que els logs de Docker omplin el disc, tots els contenidors heretaran una política de rotació comuna definida al docker-compose.yml mitjançant una àncora YAML:

```yaml
x-logging: &default-logging
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "5"

services:
  backend:
    logging: *default-logging
  worker:
    logging: *default-logging
  nginx:
    logging: *default-logging
  # ... aplicar a tots els serveis
```

Amb aquesta configuració cada contenidor ocupa un màxim de 50MB de logs.

## 3. Flux de treball

- **Estratègia de Branques i Versions:**
    - **Branques:** Utilitzarem main per a producció i develop per a noves funcionalitats.
    - **Versionat:** Utilitzarem Semantic Versioning (SemVer) per a les versions de la aplicació. Quan es demani una nova versió (ex: v1.2.0), es crearà un Git Tag o una nova branca de release depenent del canvi respecte a l'anterior.

## 4. Esquema de Xarxa Docker

El sistema defineix **4 xarxes Docker isolades** per garantir el principi de mínim privilegi: cap servei té accés a un altre tret que sigui estrictament necessari per al seu funcionament. 

### Xarxes definides

| Xarxa | Tipus | Propòsit |
| :--- | :--- | :--- |
| `sc-frontend-net` | Bridge (pública) | Única xarxa exposable a l'exterior. Conté el frontend i l'API Gateway com a punt d'entrada. |
| `sc-backend-net` | Bridge (interna) | Comunicació entre l'API Gateway, els workers, Redis, MongoDB i MinIO. Cap port exposat al host tret dels necessaris. |
| `sc-ai-net` | Bridge (interna) | Xarxa aïllada exclusiva per a la inferència GPU. Només `sc-inference-worker` hi té accés directe. |
| `sc-observability-net` | Bridge (interna) | Prometheus i Grafana separats del tràfic de negoci. Dozzle hi accedeix via bridge per poder llegir el socket de Docker. |

### Assignació de serveis per xarxa

| Servei | sc-frontend-net | sc-backend-net | sc-ai-net | sc-observability-net |
| :--- | :---: | :---: | :---: | :---: |
| `sc-frontend` | ✓ | | | |
| `sc-api-gateway` | ✓ (bridge) | ✓ (bridge) | | |
| `sc-video-manager` | | ✓ | | |
| `sc-object-storage` | | ✓ (bridge) | ✓ (bridge) | |
| `sc-redis` | | ✓ (bridge) | ✓ (bridge) | |
| `sc-mongodb` | | ✓ | | |
| `sc-inference-worker` | | | ✓ | |
| `sc-logic-aggregator` | | ✓ | | |
| `sc-active-learner` | | ✓ | | |
| `sc-dozzle` | ✓ (bridge) | | | ✓ (bridge) |
| `sc-prometheus` | | | | ✓ |
| `sc-grafana` | | | | ✓ |
| `sc-label-studio` *(opcional)* | ✓ (bridge) | ✓ (bridge) | | |

Els serveis marcats com a **bridge** pertanyen a dues xarxes simultàniament i actuen com a pont de comunicació controlat entre elles.

### Diagrama de xarxa
 
```
╔═════════════════════════════════════════════════════════════════════╗
║  sc-frontend-net  (pública)                                         ║
║                                                                     ║
║  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  ┌────────┐* ║
║  │ sc-frontend │  │ sc-api-gateway   │  │ sc-dozzle │  │sc-label│  ║
║  │ :3000→host  │  │ :8000→host       │  │:8080→host │  │-studio │  ║
║  └─────────────┘  └────────┬─────────┘  └─────┬─────┘  │:8081  │  ║
║                             │ bridge            │ bridge └───┬────┘  ║
╚════════════════════════════╪══════════════════╪═══════════╪════════╝
                             │                  │           │ bridge
╔════════════════════════════╪══════════════════╪═══════════╪════════╗
║  sc-backend-net  (interna) │                  │           │        ║
║                            │                  │           │        ║
║  ┌─────────────────────────┘                  │           │        ║
║  │                                            │           │        ║
║  ├──────────────┬─────────────┬───────────────┤           │        ║
║  ▼              ▼             ▼               ▼           ▼        ║
║  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐*  ║
║  │sc-video- │ │sc-object-│ │sc-redis│ │sc-mongodb│ │sc-label- │   ║
║  │manager   │ │storage   │ │ :6379  │ │ :27017   │ │studio    │   ║
║  └──────────┘ └────┬─────┘ └───┬────┘ └──────────┘ │(MinIO    │   ║
║                    │ bridge     │ bridge              │ access)  │   ║
║  ┌─────────────┐   │           │                     └──────────┘   ║
║  │sc-logic-    │   │           │                                    ║
║  │aggregator   │   │           │                                    ║
║  └─────────────┘   │           │                                    ║
║  ┌─────────────┐   │           │                                    ║
║  │sc-active-   │   │           │                                    ║
║  │learner      │   │           │                                    ║
║  └─────────────┘   │           │                                    ║
╚════════════════════╪═══════════╪════════════════════════════════════╝
                     │           │
╔════════════════════╪═══════════╪════════════════════════════════════╗
║  sc-ai-net  (interna)          │                                    ║
║                     ▼          ▼                                    ║
║          ┌──────────────┐ ┌──────────┐                              ║
║          │ sc-object-   │ │ sc-redis │                              ║
║          │ storage      │ │          │                              ║
║          └──────┬───────┘ └────┬─────┘                              ║
║                 └──────┬───────┘                                    ║
║                        ▼                                            ║
║            ┌───────────────────────┐                                ║
║            │  sc-inference-worker  │                                ║
║            │  GPU (CUDA) — intern  │                                ║
║            └───────────────────────┘                                ║
╚═════════════════════════════════════════════════════════════════════╝
 
╔═════════════════════════════════════════════════════════════════════╗
║  sc-observability-net  (interna)                                    ║
║                                                                     ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   ║
║  │ sc-prometheus│  │  sc-grafana  │  │  sc-dozzle               │   ║
║  │  :9090       │◄─┤  :3001→host  │  │  + /var/run/docker.sock  │   ║
║  │  (scrapes    │  │              │  │    (read-only)            │   ║
║  │   tots els   │  └──────────────┘  └──────────────────────────┘   ║
║  │   serveis)   │                                                   ║
║  └──────────────┘                                                   ║
╚═════════════════════════════════════════════════════════════════════╝
 
* Servei OPCIONAL — s'aixeca manualment: docker compose up sc-label-studio
  No és dependència de cap altre servei. No s'inicia amb docker compose up.
 
Ports exposats al host:
  :3000  → sc-frontend
  :8000  → sc-api-gateway
  :8080  → sc-dozzle
  :3001  → sc-grafana
  :9000  → sc-object-storage (MinIO, opcional per a gestió directa)
  :8081  → sc-label-studio * (només quan el servei està actiu)
 
La resta de serveis (Redis, MongoDB, workers) NO exposen cap port al host.
 
Flux d'etiquetatge (només quan sc-label-studio està actiu):
  Admin puja vídeo → sc-video-manager trosseja → frames a MinIO (labeling-frames)
  → sc-label-studio llegeix frames de MinIO → etiqueta → exporta dataset a MinIO (datasets)
  → sc-active-learner consumeix el dataset per al proper entrenament
```
### Regles de comunicació
 
- **Frontend → API Gateway:** HTTP REST a través de `sc-frontend-net`. És l'únic canal d'entrada de dades externes al sistema.
- **API Gateway → Backend:** Publica jobs a Redis, llegeix/escriu a MongoDB i puja fitxers a MinIO, tot dins de `sc-backend-net`.
- **Redis i MinIO com a bridges:** Actuen de pont controlat entre `sc-backend-net` i `sc-ai-net`. L'`sc-inference-worker` mai té accés directe a MongoDB ni a l'API Gateway.
- **Inference Worker aïllat:** Només pot llegir tasques de Redis i descarregar/pujar frames de MinIO. No pot accedir a cap altre servei de backend.
- **Observabilitat separada:** Prometheus fa scraping dels endpoints `/metrics` dels serveis de backend a través de la seva xarxa pròpia. Grafana mai toca `sc-backend-net`.
- **Dozzle i el socket Docker:** Dozzle necessita muntar `/var/run/docker.sock` en mode lectura per accedir als logs dels contenidors. S'ha de restringir explícitament a `read-only` al `docker-compose.yml`.
- **Label Studio (opcional):** Quan actiu, accedeix a MinIO via `sc-backend-net` per llegir frames del bucket `labeling-frames` i escriure datasets a `datasets`. La seva UI és accessible al host via `:8081`. S'aixeca manualment amb `docker compose up sc-label-studio`.

## 5. Estratègia de Storage — MinIO (S3)

Tot l'emmagatzematge de fitxers del sistema passa exclusivament per `sc-object-storage` (MinIO). No hi ha carpetes compartides entre contenidors ni volums Docker per a fitxers de dades. Cada servei accedeix als fitxers que necessita descarregant-los de MinIO al seu propi sistema de fitxers efímer.

### Buckets definits
 
| Bucket | Clau d'objecte | Escriu | Llegeix | Retenció |
| :--- | :--- | :--- | :--- | :--- |
| `raw-videos` | `{match_id}/original.mp4` | `sc-api-gateway` | `sc-video-manager` | 30 dies |
| `pending-frames` | `{match_id}/{frame_id}.jpg` | `sc-video-manager` | `sc-inference-worker` | 7 dies |
| `processed-frames` | `{match_id}/{frame_id}_overlay.jpg` | `sc-video-manager` | `sc-video-manager` (muntatge) | 7 dies |
| `processed-videos` | `{match_id}/output.mp4` | `sc-video-manager` | `sc-api-gateway` (download) | Indefinida |
| `feedback-data` | `{match_id}/{frame_id}_crop.jpg` | `sc-inference-worker` | `sc-active-learner` | Indefinida |
| `models` | `yolo/weights/v{N}.pt` · `cnn/weights/v{N}.keras` | `sc-active-learner` | `sc-inference-worker` | Indefinida (versionat) |
| `labeling-videos` | `{session_id}/original.mp4` | `sc-api-gateway` (admin) | `sc-video-manager` | 30 dies |
| `labeling-frames` | `{session_id}/{frame_id}.jpg` | `sc-video-manager` | `sc-label-studio` | 30 dies |
| `datasets` | `yolo/v{N}/` · `cnn/v{N}/` | `sc-label-studio` | `sc-active-learner` | Indefinida (versionat) |

### Convenció de claus
 
- `{match_id}` — UUID del partit generat per MongoDB en el moment de la creació (ex: `a3f1c2d4-7e81-...`).
- `{frame_id}` — Número de frame zero-padded a 6 dígits (ex: `frame_000001`). Garanteix l'ordenació lexicogràfica correcta.
- `{N}` — Versió semàntica del model, incremental (ex: `v1`, `v2`). El model actiu és sempre el de versió més alta disponible al bucket.

### Flux de dades entre buckets
 
1. L'usuari puja el vídeo → `sc-api-gateway` escriu a `raw-videos`.
2. `sc-video-manager` descarrega de `raw-videos`, extreu els frames i els puja a `pending-frames`.
3. `sc-inference-worker` descarrega frames de `pending-frames`, fa la inferència i puja els crops de baixa confiança a `feedback-data`.
4. `sc-video-manager` descarrega els frames de `pending-frames`, dibuixa els overlays i els puja a `processed-frames`.
5. `sc-video-manager` munta el vídeo final des de `processed-frames` i el puja a `processed-videos`.
6. `sc-active-learner` descarrega de `feedback-data`, re-entrena i puja els nous pesos a `models`.
7. `sc-inference-worker` detecta nous pesos a `models` i els carrega per substituir els anteriors.

### Optimització: prefetch buffer a l'inference-worker
 
La descàrrega de frames des de MinIO afegeix una latència de xarxa per frame (~1–5ms en xarxa Docker interna). Per amortitzar-la, `sc-inference-worker` manté un **buffer en memòria de 5–10 frames** prefetchats mentre processa el frame actual. Això elimina el temps d'espera de descàrrega del camí crític de la GPU sense necessitar cap carpeta compartida.
 
El nombre de frames del buffer és configurable via variable d'entorn `PREFETCH_BUFFER_SIZE` (per defecte: `8`).

### Serveis amb accés a MinIO
 
Únicament els serveis següents tenen credencials de MinIO configurades:
 
- `sc-api-gateway` — upload inicial i download del vídeo final.
- `sc-video-manager` — lectura i escriptura de vídeos i frames.
- `sc-inference-worker` — lectura de frames i escriptura de crops de feedback.
- `sc-active-learner` — lectura de feedback i escriptura de nous pesos de model.
- `sc-label-studio` *(opcional)* — lectura de `labeling-frames` i escriptura a `datasets`.
 
Cap altre servei (`sc-redis`, `sc-mongodb`, `sc-logic-aggregator`, etc.) té accés configurat a MinIO.