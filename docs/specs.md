# SmartChrono IP - Especificacions del Projecte

## 1. Visió del projecte

### 1.1 Resum

SmartChrono IP és una eina d'anàlisi de dades esportives dissenyada per automatitzar el **recompte de minuts** jugats per cada jugador en un partit de futbol sala. Mitjançant tècniques de **visió per computador** (Computer Vision), el sistema processa el vídeo d'un partit, identifica els jugadors pel seu dorsal i genera un informe detallat de la participació de cadascun.

### 1.2 Problema i Oportunitat

En el futbol sala modern, la gestió de les rotacions és clau per mantenir la intensitat física. Actualment, el recompte de minuts es fa de forma subjectiva o **manual**, la qual cosa comporta **errors** i una pèrdua de temps per al cos tècnic.
Existeix l'oportunitat d'utilitzar la infraestructura de gravació del Racing Pineda FS per extreure dades objectives que ajudin a l'entrenador a prendre millors decisions basades en la càrrega real de cada jugador.

### 1.3 Objectius Principals

* **Automatització:** Eliminar la necessitat de cronometratge manual durant o després del partit.
* **Identificació Precisa:** Reconèixer els dorsals en condicions de moviment i resolució estàndard.
* **Generació de Reports:** Proporcionar un resum final exportable amb els minuts totals i les franges de temps jugades per cada dorsal.

### 1.4 Definició del Producte

El producte mínim viable serà una aplicació que:

1. Rebi un fitxer de vídeo (`.mp4`).
2. Detecti els jugadors a la pista (utilitzant YOLOv8).
3. Identifiqui el dorsal del jugador quan sigui visible.
4. Calculi el temps de permanència a la pista per a cada ID detectat.
5. Exporti un fitxer de dades (CSV/Excel) amb el resum de minuts.

## 2. Arquitectura tècnica

El sistema SmartChrono IP adopta una arquitectura de microserveis mitjançant contenidors Docker completament desacoblats mitjançant un patró de Cua de Tasques (Producer-Consumer). Això permet que el processament intensiu de vídeo no bloquegi l'API i facilita el re-entrenament del model en paral·lel.

Utilitzarem les ultimes versions suportades de tot, i si hi han components/llibreries també han de ser les últimes versions sense que hi hagin problemes de seguretat.

### 2.1 Stack Tecnològic

* **Backend API:** FastAPI (Python 3.11).
* **AI Engine:** Ultralytics (YOLOv8) + TensorFlow (CNN Dorsals).
* **Message Broker:** Redis (Pub/Sub & List Queues).
* **Database:** MongoDB 6.0+ (NoSQL).
* **Frontend:** React (Vite) + Tailwind CSS.
* **Retraining Engine:** PyTorch/TensorFlow (entrenament asíncron).
* **Video Processing:** FFmpeg + OpenCV.
* **Observabilitat:**
  * **Dozzle:** Visualització de logs de tots els contenidors en temps real.
  * **Sentry:** Rastreig d'errors crítics en producció (SDK integrat en serveis Python).
  * **Prometheus:** Recollida de mètriques (CPU, GPU VRAM, temps de processament per frame).
  * **Grafana:** Dashboard visual de mètriques i estat del sistema.

### 2.2 Definició de Contenidors (Docker Services)

| Servei | Responsabilitat Tècnica | Stack Intern |
| :--- | :--- | :--- |
| **`sc-api-gateway`** | **Punt d'entrada REST**. Gestiona el CRUD de MongoDB, l'autenticació i l'enviament de "Jobs" de processament a Redis. | FastAPI, Uvicorn, Motor (MongoDB Driver) |
| **`sc-video-manager`** | **Ingestió i Muntatge:** Talla el vídeo original en frames (.jpg). Un cop finalitzat el procés, re-munta un nou vídeo sobreposant els bounding boxes i IDs. | FFmpeg, OpenCV, PyAV |
| **`sc-object-storage`**| **S3 Storage:** Servidor d'objectes per guardar frames, vídeos i models de forma centralitzada. | MinIO Server (S3 API) |
| **`sc-inference-worker`** | **Inferència d'IA:** Escolta la cua `pending_frames`. Executa YOLOv8 (jugadors) i la CNN personalitzada (dorsals) utilitzant la GPU (8GB VRAM). | PyTorch, CUDA 12.6, TensorFlow/Keras, Ultralytics |
| **`sc-logic-aggregator`** | **Tracking i Temps:** Rep els JSON de detecció de l'inference-worker. Implementa la lògica de ByteTrack per mantenir la identitat i calcula el temps de permanència a "zona activa". | FilterPy, NumPy, Pandas (Time-series) |
| **`sc-active-learner`** | **Feedback Loop:** Automatitza el fine-tuning dels models. Detecta noves dades a `/feedback` i entrena noves versions dels pesos (.pt / .keras). | Ultralytics Trainer, Scikit-learn |
| **`sc-frontend`** | **Interfície Gràfica:** Mostra per insertar el video, mostra el reproductor de vídeo i el llistat de jugadors. | React 18, Vite, Tailwind CSS, Lucide Icons |
| **`sc-dozzle`** | **Log Viewer:** Interfície web per monitoritzar logs dels contenidors en temps real. | Dozzle Runtime |
| **`sc-prometheus`** | **Metrics Scraper:** Recull mètriques d'ús de recursos i rendiment de l'IA (frames/segon). | Prometheus Server |
| **`sc-grafana`** | **Visualization:** Panells de control per visualitzar mètriques de Prometheus. | Grafana Labs |
| **`sc-redis`** | **Broker & State:** Gestió de cues de tasques (RQ/Celery style) i emmagatzematge temporal de l'estat del processament. | Redis Streams, Pub/Sub |
| **`sc-mongodb`** | **Data Persistence:** Base de dades documental per a la persistència de partits, plantilles de jugadors i estadístiques històriques. | MongoDB Document Store |
| **`sc-label-studio`** | **Etiquetatge de dades:** Interfície web per etiquetar frames de vídeo (bounding boxes de jugadors i dorsals). Servei opcional que només s'aixeca durant sessions d'etiquetatge. S'integra amb MinIO per llegir frames i exportar datasets directament als buckets. | Label Studio, integració S3/MinIO |

### 2.3 Workflow de Dades (Frame-by-Frame Pipeline)

El processament de SmartChrono IP segueix un model asíncron basat en esdeveniments per optimitzar l'ús de la GPU i garantir la persistència de les dades.

#### Fase A: Ingestió i Fragmentació
1. **Upload:** L'usuari puja el fitxer `.mp4` a través del `sc-frontend`. El `sc-api-gateway` el rep i l'emmagatzema al bucket `raw-videos` de **MinIO**.
2. **Trigger:** L'API publica un missatge a la cua de Redis `video_to_process`.
3. **Decomposició:** El `sc-video-manager` descarrega el vídeo, l'analitza amb FFmpeg i extreu cada frame en format `.jpg`. 
4. **Storage:** Cada frame es puja immediatament al bucket `pending-frames` de **MinIO** amb una clau única: `partit_id/frame_000001.jpg`.
5. **Indexing:** Per cada frame pujat, s'afegeix una tasca a la cua de Redis `task_frames`.

#### Fase B: Inferència i Detecció
1. **Consum de tasques:** El `sc-inference-worker` (amb accés a la GPU) extreu els IDs de frame de Redis.
2. **Download & Predict:** - Descarrega el frame des de **MinIO**.
   - Executa **YOLOv8** per localitzar jugadors i la pilota.
   - Per a cada jugador detectat, realitza un *crop* de la zona del dorsal i l'envia a la **CNN**.
3. **Publish:** Els resultats (coordenades, classe, dorsal, confiança) s'envien a la cua `detected_frames_results`.

#### Fase C: Seguiment i Lògica Esportiva
1. **Tracking:** El `sc-logic-aggregator` processa els resultats seqüencialment utilitzant **ByteTrack**. Assigna un `Player_ID` persistent a cada trajectòria.
2. **Càlcul de Minuts:** - Si un `Player_ID` és identificat amb un dorsal (ex: "8") i es manté actiu en pista, el sistema incrementa el seu comptador de temps a **MongoDB**.
   - Es gestionen les "zones mortes" (banqueta) per aturar el cronòmetre automàticament.
3. **Events:** Els esdeveniments especials (chutes, canvis) es guarden a la col·lecció `events` de **MongoDB**.

#### Fase D: Re-muntatge i Feedback (Active Learning)
1. **Video Render:** Un cop finalitzat el partit, el `sc-video-manager` recupera les coordenades de Mongo, descarrega els frames de **MinIO**, dibuixa els *overlays* (caixes i noms) i genera el vídeo final que es guarda al bucket `processed-videos`.
2. **Feedback Loop:** - Aquells frames amb una confiança d'identificació baixa (< 0.6) o marcats manualment per l'usuari es copien al bucket `feedback-data`.
   - El `sc-active-learner` utilitza aquestes imatges per re-entrenar la CNN o el YOLO de forma asíncrona, generant una nova versió del model a `models/weights/`.

### 2.4 Lògica de Cronometratge i Control de Sessió

Aquest mòdul, executat pel `sc-logic-aggregator`, és l'encarregat de transformar les coordenades espacials en temps de joc efectiu. Atès que en categories de base el temps és "a rellotge corregut", el sistema es basa en la presència física dins l'àrea de joc.

**A. Input de l'Usuari (Control des del Frontend)**
Per evitar processar l'escalfament o el temps de descans, l'usuari interaccionarà amb el `sc-frontend` per definir els límits de la sessió:
* **Match Start/End:** Mitjançant el reproductor de vídeo, l'usuari marcarà el *timestamp* exacte de l'inici i la fi del partit. Aquests valors s'enviaran a l'API com a `start_frame` i `end_frame`.
* **Definició de Pista (ROI):** L'usuari dibuixarà sobre un frame de referència el polígon (4 punts) que delimita les línies de banda i fons. Només els jugadors dins d'aquest polígon sumaran minuts.

**B. Algoritme de Presència Efectiva**
El sistema analitzarà cada frame de forma seqüencial dins del rang definit:
1. **Detecció de Posició:** Es pren el punt mig de la base del *bounding box* del jugador com a referència de la seva posició a terra.
2. **Filtre Espacial:** Si el punt està dins del polígon de pista, el jugador es marca com a `IN_GAME`. Si està fora (banqueta) o no es detecta, es marca com a `OFF_COURT`.
3. **Suma de Temps:** Per cada frame on el jugador és `IN_GAME`, s'incrementa el seu comptador individual: 
   $$Temps\_Jugador = \sum \frac{1}{FPS_{video}}$$

**C. Gestió d'Oclusions i Robustesa (Smoothing)**
Per evitar que el cronòmetre s'aturi si un jugador queda tapat momentàniament per un altre o surt de pla uns segons, s'aplicarà una **Histeresi de Persistència**:
* **Buffer de Desaparició:** Si un jugador identificat desapareix o és marcat com `OFF_COURT`, el sistema esperarà un marge de **3 segons** (configurable) abans d'aturar el seu cronòmetre. 
* **Re-identificació:** Si el `Player_ID` reapareix dins de la pista abans d'exhaurir el buffer, el sistema omplirà el buit temporal automàticament, considerant que el jugador mai ha abandonat el terreny de joc.
* **Confirmació de Canvi:** Un jugador només deixarà de sumar minuts de forma definitiva quan sigui detectat fora del polígon o no aparegui en el tracking durant més de 3 segons consecutius.

### 2.5 Política de Logging i Monitorització

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

### 2.6 Gestió i Retenció de Logs

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

### 2.7 Flux de treball

- **Estratègia de Branques i Versions:**
    - **Branques:** Utilitzarem main per a producció i develop per a noves funcionalitats.
    - **Versionat:** Utilitzarem Semantic Versioning (SemVer) per a les versions de la aplicació. Quan es demani una nova versió (ex: v1.2.0), es crearà un Git Tag o una nova branca de release depenent del canvi respecte a l'anterior.

### 2.8 Esquema de Xarxa Docker

El sistema defineix **4 xarxes Docker isolades** per garantir el principi de mínim privilegi: cap servei té accés a un altre tret que sigui estrictament necessari per al seu funcionament. 

#### Xarxes definides

| Xarxa | Tipus | Propòsit |
| :--- | :--- | :--- |
| `sc-frontend-net` | Bridge (pública) | Única xarxa exposable a l'exterior. Conté el frontend i l'API Gateway com a punt d'entrada. |
| `sc-backend-net` | Bridge (interna) | Comunicació entre l'API Gateway, els workers, Redis, MongoDB i MinIO. Cap port exposat al host tret dels necessaris. |
| `sc-ai-net` | Bridge (interna) | Xarxa aïllada exclusiva per a la inferència GPU. Només `sc-inference-worker` hi té accés directe. |
| `sc-observability-net` | Bridge (interna) | Prometheus i Grafana separats del tràfic de negoci. Dozzle hi accedeix via bridge per poder llegir el socket de Docker. |

#### Assignació de serveis per xarxa

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

#### Diagrama de xarxa
 
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
#### Regles de comunicació
 
- **Frontend → API Gateway:** HTTP REST a través de `sc-frontend-net`. És l'únic canal d'entrada de dades externes al sistema.
- **API Gateway → Backend:** Publica jobs a Redis, llegeix/escriu a MongoDB i puja fitxers a MinIO, tot dins de `sc-backend-net`.
- **Redis i MinIO com a bridges:** Actuen de pont controlat entre `sc-backend-net` i `sc-ai-net`. L'`sc-inference-worker` mai té accés directe a MongoDB ni a l'API Gateway.
- **Inference Worker aïllat:** Només pot llegir tasques de Redis i descarregar/pujar frames de MinIO. No pot accedir a cap altre servei de backend.
- **Observabilitat separada:** Prometheus fa scraping dels endpoints `/metrics` dels serveis de backend a través de la seva xarxa pròpia. Grafana mai toca `sc-backend-net`.
- **Dozzle i el socket Docker:** Dozzle necessita muntar `/var/run/docker.sock` en mode lectura per accedir als logs dels contenidors. S'ha de restringir explícitament a `read-only` al `docker-compose.yml`.
- **Label Studio (opcional):** Quan actiu, accedeix a MinIO via `sc-backend-net` per llegir frames del bucket `labeling-frames` i escriure datasets a `datasets`. La seva UI és accessible al host via `:8081`. S'aixeca manualment amb `docker compose up sc-label-studio`.

### 2.9 Estratègia de Storage — MinIO (S3)

Tot l'emmagatzematge de fitxers del sistema passa exclusivament per `sc-object-storage` (MinIO). No hi ha carpetes compartides entre contenidors ni volums Docker per a fitxers de dades. Cada servei accedeix als fitxers que necessita descarregant-los de MinIO al seu propi sistema de fitxers efímer.

#### Buckets definits
 
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

#### Convenció de claus
 
- `{match_id}` — UUID del partit generat per MongoDB en el moment de la creació (ex: `a3f1c2d4-7e81-...`).
- `{frame_id}` — Número de frame zero-padded a 6 dígits (ex: `frame_000001`). Garanteix l'ordenació lexicogràfica correcta.
- `{N}` — Versió semàntica del model, incremental (ex: `v1`, `v2`). El model actiu és sempre el de versió més alta disponible al bucket.

#### Flux de dades entre buckets
 
1. L'usuari puja el vídeo → `sc-api-gateway` escriu a `raw-videos`.
2. `sc-video-manager` descarrega de `raw-videos`, extreu els frames i els puja a `pending-frames`.
3. `sc-inference-worker` descarrega frames de `pending-frames`, fa la inferència i puja els crops de baixa confiança a `feedback-data`.
4. `sc-video-manager` descarrega els frames de `pending-frames`, dibuixa els overlays i els puja a `processed-frames`.
5. `sc-video-manager` munta el vídeo final des de `processed-frames` i el puja a `processed-videos`.
6. `sc-active-learner` descarrega de `feedback-data`, re-entrena i puja els nous pesos a `models`.
7. `sc-inference-worker` detecta nous pesos a `models` i els carrega per substituir els anteriors.

#### Optimització: prefetch buffer a l'inference-worker
 
La descàrrega de frames des de MinIO afegeix una latència de xarxa per frame (~1–5ms en xarxa Docker interna). Per amortitzar-la, `sc-inference-worker` manté un **buffer en memòria de 5–10 frames** prefetchats mentre processa el frame actual. Això elimina el temps d'espera de descàrrega del camí crític de la GPU sense necessitar cap carpeta compartida.
 
El nombre de frames del buffer és configurable via variable d'entorn `PREFETCH_BUFFER_SIZE` (per defecte: `8`).

#### Serveis amb accés a MinIO
 
Únicament els serveis següents tenen credencials de MinIO configurades:
 
- `sc-api-gateway` — upload inicial i download del vídeo final.
- `sc-video-manager` — lectura i escriptura de vídeos i frames.
- `sc-inference-worker` — lectura de frames i escriptura de crops de feedback.
- `sc-active-learner` — lectura de feedback i escriptura de nous pesos de model.
- `sc-label-studio` *(opcional)* — lectura de `labeling-frames` i escriptura a `datasets`.
 
Cap altre servei (`sc-redis`, `sc-mongodb`, `sc-logic-aggregator`, etc.) té accés configurat a MinIO.

### 2.10 Seguretat i Autenticació (JWT)

- **Estratègia:** Doble token amb Access Token de vida curta i Refresh Token persistent.
    - **Access Token:** JWT signat (HS256) amb vida de 15 minuts. S'envia a cada petició via capçalera `Authorization: Bearer`. El payload conté `user_id`, `roles[]` i `team_id`. Mai conté dades sensibles.
    - **Refresh Token:** Token opac de 7 dies emmagatzemat en cookie `HttpOnly` + `Secure` + `SameSite=Strict`. JavaScript no hi pot accedir mai (protecció XSS).
    - **Refresh Token Rotation:** Cada crida a `/auth/refresh` invalida el token anterior i n'emet un de nou. Si un token robat intenta fer refresh un cop ja ha estat rotat, el sistema el detecta com a reutilització i invalida tota la sessió immediatament.
    - **Blacklist de sessions:** Els Refresh Tokens actius es registren a Redis amb TTL de 7 dies. En fer logout o revocar una sessió, el token s'afegeix a la blacklist i queda inutilitzable immediatament.
    - **Endpoints d'autenticació:**
        - `POST /auth/login` → emet ambdós tokens.
        - `POST /auth/refresh` → valida cookie, comprova blacklist Redis, invalida token antic, emet nou Refresh Token + nou Access Token.
        - `POST /auth/logout` → invalida Refresh Token a Redis.
    - **Frontend (React):** L'Access Token es guarda únicament en memòria (Zustand), mai a `localStorage`. El refresc és automàtic i transparent per a l'usuari mitjançant un interceptor d'Axios.

- **Secret de signatura JWT:**
    - `JWT_SECRET` és una variable d'entorn injectada en temps d'execució. Mai no pot estar hardcoded al codi ni commited al repositori.
    - Mínim 256 bits d'entropia (32 bytes aleatoris, generats amb `openssl rand -hex 32`).
    - Gestió: variable en fitxer `.env` per a entorns locals (exclòs de `.gitignore`), i Docker Secret o variable d'entorn xifrada per a producció.
    - Exemple de variables d'entorn requerides a `sc-api-gateway`:
        ```env
        JWT_SECRET=<mínim 32 bytes aleatoris>
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
        JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
        ```

- **Autenticació entre microserveis (Internal API Key):**
    - Els serveis interns (`sc-logic-aggregator`, `sc-inference-worker`, `sc-video-manager`, `sc-active-learner`) no accepten peticions sense autenticació, ni tan sols des de la xarxa Docker interna.
    - Cada servei intern valida una capçalera `X-Internal-API-Key` a totes les peticions rebudes.
    - La clau es comparteix entre serveis via variable d'entorn `INTERNAL_API_KEY`, generada amb el mateix procediment que `JWT_SECRET`.
    - `sc-api-gateway` és l'únic servei autoritzat a cridar els serveis interns directament. La resta de serveis es comuniquen exclusivament a través de Redis (cues) o MinIO (fitxers), sense cridades HTTP directes entre ells.
    - Exemple de capçalera per a comunicació interna:
        ```
        X-Internal-API-Key: <INTERNAL_API_KEY>
        ```

### 2.11 Esquema de Base de Dades (MongoDB)
 
El sistema utilitza 4 col·leccions. La instal·lació és per club, però el sistema suporta múltiples categories (Aleví A, Infantil B...). Els documents players i matches inclouen team_id per separar les dades per categoria.

#### Col·lecció `players` — Plantilla de jugadors
 
Document persistent i reutilitzable entre partits. Es crea i manté manualment des del frontend.
 
```json
{
  "_id":        "ObjectId",
  "team_id":    "ObjectId",
  "dorsal":     9,
  "name":       "Pau Garcia",
  "position":   "base",
  "active":     true,
  "created_at": "ISODate"
}
```

- `team_id` és una FK lògica cap a `teams._id` de `sc-app-db`. Obligatori.
- `position` accepta els valors: `base`, `aler`, `pivot`.
- `active: false` desactiva el jugador sense eliminar-lo (historial preservat).
- **Índex:** compost `team_id + dorsal` (unique) — un dorsal és únic dins d'un equip, però dos equips poden tenir el mateix dorsal.

#### Col·lecció `matches` — Partits
 
Document central de cada sessió de processament. Conté tota la metadata del vídeo i la configuració de la sessió.
 
```json
{
  "_id":          "ObjectId",
  "team_id":      "ObjectId",
  "title":        "Lliga J12 vs Joventut",
  "date":         "ISODate",
  "status":       "processing",
  "video_raw":    "a3f1c2d4/original.mp4",
  "video_output": "a3f1c2d4/output.mp4",
  "fps":          25,
  "start_frame":  1500,
  "end_frame":    138000,
  "roi_polygon":  [
    { "x": 120, "y": 80 },
    { "x": 1800, "y": 80 },
    { "x": 1800, "y": 1000 },
    { "x": 120, "y": 1000 }
  ],
  "created_at":   "ISODate",
  "updated_at":   "ISODate"
}
```

- `team_id` és una FK lògica cap a `teams._id`. Obligatori.
- `status` accepta els valors: `pending`, `processing`, `frames_ready`, `done`, `error`.
  - `pending` — el partit s'ha creat però el vídeo encara no s'ha processat.
  - `processing` — `sc-video-manager` ha rebut el job i està extraient frames.
  - `frames_ready` — tots els frames estan a MinIO i `sc-inference-worker` els està processant.
  - `done` — el pipeline ha finalitzat i el vídeo de sortida està disponible.
  - `error` — ha fallat algun pas del pipeline.
- `video_raw` i `video_output` contenen la clau MinIO (sense bucket), no una URL absoluta.
- `roi_polygon` és l'array de 4 punts `{x, y}` en píxels que defineix la zona activa de joc.
- `start_frame` i `end_frame` marquen el rang de frames a processar (descarten escalfament i descans).
- **Índexos:** `team_id`, `team_id + status`, `team_id + date` (desc), `status`, `date`.

#### Col·lecció `match_players` — Jugadors per partit
 
Intersecció entre un partit i un jugador. Aquí viuen els minuts jugats, la confiança de detecció i l'historial de presència en pista. Es crea un document per cada jugador actiu quan comença el processament d'un partit.
 
```json
{
  "_id":            "ObjectId",
  "match_id":       "ObjectId",
  "player_id":      "ObjectId",
  "seconds_played": 1842,
  "confidence_avg": 0.87,
  "track_ids":      [3, 7, 15],
  "intervals": [
    { "in": 1500,  "out": 45000, "src": "auto" },
    { "in": 47200, "out": 98000, "src": "auto" },
    { "in": 98500, "out": 99000, "src": "manual" }
  ],
  "status":           "IN_GAME",
  "last_seen_frame":  99000,
  "updated_at":       "ISODate"
}
```
 
- `seconds_played` és el valor acumulat final. Es calcula sumant la durada de tots els `intervals`.
- `track_ids` conté tots els IDs de ByteTrack que el sistema ha associat a aquest jugador al llarg del partit (un jugador pot tenir múltiples track_ids per re-identificacions).
- `intervals` és l'historial complet de períodes en pista: `in` i `out` són números de frame, `src` indica si l'interval l'ha generat el sistema automàticament (`auto`) o l'ha corregit un usuari (`manual`).
- `status` reflecteix l'estat en temps real durant el processament: `IN_GAME` o `OFF_COURT`.
- **Índexos:** `match_id`, compost `match_id + player_id` (unique).

#### Col·lecció `events` — Esdeveniments del partit
 
Registre immutable de tots els esdeveniments discrets: entrades i sortides de pista, substitucions i frames marcats per al feedback del model. És el log d'auditoria del partit.
 
```json
{
  "_id":          "ObjectId",
  "match_id":     "ObjectId",
  "player_id":    "ObjectId",
  "type":         "enter",
  "frame_number": 1500,
  "timestamp_s":  60.0,
  "confidence":   0.91,
  "source":       "auto",
  "created_at":   "ISODate"
}
```
 
- `type` accepta els valors: `enter`, `exit`, `substitution`, `feedback_flagged`, `low_confidence`.
- `player_id` és nullable: els esdeveniments de tipus `low_confidence` o `feedback_flagged` poden no tenir jugador associat si el dorsal no s'ha pogut identificar.
- `timestamp_s` és el segon de vídeo relatiu a `start_frame` (no al frame absolut).
- `source` indica si l'esdeveniment l'ha generat el sistema (`auto`) o l'usuari (`manual`).
- **Índexos:** `match_id + type`, `match_id + frame_number`.

#### Resum de relacions
 
| Relació | Tipus | Nota |
| :--- | :--- | :--- |
| `players` → `match_players` | 1 : N | Un jugador pot aparèixer a molts partits |
| `matches` → `match_players` | 1 : N | Un partit té un document per cada jugador actiu |
| `matches` → `events` | 1 : N | Un partit genera múltiples esdeveniments |
| `players` → `events` | 0..1 : N | Un esdeveniment pot no tenir jugador identificat |

### 2.12 Gestió d'Usuaris — Arquitectura de Doble Base de Dades
 
El sistema separa **identitat** i **negoci** en dues bases de dades MongoDB lògicament independents dins del mateix contenidor `sc-mongodb`. Si `sc-app-db` es veiés compromesa, un atacant obtindria dades esportives però cap credencial. Si `sc-auth-db` es veiés compromesa, no tindria accés a cap dada de partits ni jugadors.

| Base de dades | Contingut | Qui hi accedeix |
| :--- | :--- | :--- |
| `sc-auth-db` | Credencials, rols, sessions (refresh tokens) | Únicament `sc-api-gateway` en login/refresh/logout |
| `sc-app-db` | Equips, perfils, partits, jugadors, estadístiques | `sc-api-gateway` i serveis interns (via Redis/MinIO) |

L'API Gateway utilitza dos clients Motor independents: `auth_db` i `app_db`. Cap servei té accés creuat entre les dues bases de dades.

#### `sc-auth-db` — Col·lecció `users`
 
```json
{
  "_id":           "ObjectId",
  "email":         "pau.garcia@clubbadalona.cat",
  "password_hash": "$2b$12$...",
  "role":          "coach",
  "active":        true,
  "created_at":    "ISODate",
  "last_login":    "ISODate"
}
```
 
- `role` accepta: `admin`, `coach`, `assistant`, `player`.
- `password_hash` utilitza bcrypt amb cost factor 12 mínim.
- Cap camp d'aquest document fa referència a `sc-app-db`.
- **Índex:** `email` (unique).

#### `sc-auth-db` — Col·lecció `refresh_tokens`
 
```json
{
  "_id":        "ObjectId",
  "user_id":    "ObjectId",
  "token_hash": "sha256:abc123...",
  "expires_at": "ISODate"
}
```
 
- `token_hash` emmagatzema el hash SHA-256 del token opac, mai el token en clar.
- `expires_at` té un **TTL index** de MongoDB que elimina automàticament els tokens expirats.
- Amb la Refresh Token Rotation (vegeu punt 2.10), cada crida a `/auth/refresh` elimina el document anterior i en crea un de nou.

#### `sc-app-db` — Col·lecció `teams` (categories)
 
Representa cada categoria del club (Aleví A, Infantil B, etc.).
 
```json
{
  "_id":       "ObjectId",
  "name":      "Aleví A",
  "category":  "aleví",
  "coach_id":  "ObjectId",
  "active":    true
}
```
 
- `coach_id` és una **FK lògica** cap a `users._id` de `sc-auth-db`. MongoDB no enforça aquesta referència; la consistència la gestiona l'API Gateway.
- Un entrenador (`coach_id`) pot aparèixer a múltiples documents `teams`.
- Una categoria té exactament un entrenador.
- **Índexos:** `coach_id`, `category`.

#### `sc-app-db` — Col·lecció `user_profiles`
 
Estén cada usuari amb les dades de negoci necessàries per al control d'accés i la navegació.
 
```json
{
  "_id":          "ObjectId",
  "user_id":      "ObjectId",
  "display_name": "Pau Garcia",
  "team_ids":     ["ObjectId", "ObjectId"],
  "player_id":    null
}
```
 
- `user_id` és la FK lògica que connecta amb `users._id` de `sc-auth-db`.
- `team_ids` conté els equips als quals l'usuari té accés:
  - **admin:** array amb tots els `team_ids` del club (o `[]` per indicar accés total).
  - **coach:** els equips que entrena.
  - **assistant:** els equips als quals ha estat assignat.
  - **player:** l'equip al qual pertany.
- `player_id` és nullable i només s'omple per al rol `player`, apuntant al document `players` de `sc-app-db`.
- **Índex:** `user_id` (unique).

#### Flux d'autenticació i autorització
 
1. **Login:** L'API consulta `sc-auth-db` per verificar email i password (bcrypt). Si és correcte, carrega el `user_profile` de `sc-app-db` per obtenir `team_ids` i `player_id`.
2. **Emissió del JWT:** El payload del token inclou `user_id`, `role` i `team_ids`. D'aquesta manera, cada petició posterior és autocontinguda.
3. **Peticions normals:** L'API valida el JWT localment (sense consultar cap base de dades) i filtra les queries de `sc-app-db` pels `team_ids` extrets del token.
4. **`sc-auth-db` en repòs:** Un cop emès el JWT, `sc-auth-db` no es torna a consultar fins al proper login, refresh o logout.

#### Regles d'accés per rol
 
| Rol | Veu partits de | Pot crear partits | Pot editar jugadors |
| :--- | :--- | :--- | :--- |
| `admin` | Tots els equips | Sí | Sí |
| `coach` | Els seus equips | Sí | Sí (els seus equips) |
| `assistant` | Els seus equips | No | No |
| `player` | El seu equip | No | No |

Amb `team_id` a `players` i `matches`, les queries de l'API ara filtren per `team_id` en lloc de confiar únicament en els `team_ids` del JWT:
 
- **Coach/assistant/player:** `{ team_id: { $in: token.team_ids } }` — veu només els jugadors i partits dels seus equips.
- **Admin:** sense filtre de `team_id` — veu tot.
 
Aquesta és la regla que `sc-api-gateway` ha d'aplicar a tots els endpoints de `players` i `matches`.

### 2.13 Variables d'Entorn

Cada servei té el seu propi fitxer `.env` ubicat a `services/{nom-servei}/.env`. Tots els fitxers `.env` estan exclosos del repositori via `.gitignore`. El repositori inclou un fitxer `.env.example` per a cada servei amb els noms de les variables i valors d'exemple segurs per a desenvolupament local.

**Convenció de noms:**
- Majúscules amb separador `_`.
- Prefix del servei per a variables compartides que apareixen a més d'un contenidor (ex: `REDIS_HOST`, `MONGO_URI`).
- Les variables que contenen secrets reals (claus, contrasenyes) s'indiquen amb el comentari `# SECRET`.

#### `services/sc-api-gateway/.env`
 
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

#### `services/sc-video-manager/.env`
 
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

#### `services/sc-inference-worker/.env`
 
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

#### `services/sc-logic-aggregator/.env`
 
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

#### `services/sc-active-learner/.env`
 
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

#### `services/sc-frontend/.env`
 
```env
# API
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
 
# Entorn
VITE_ENV=development
```
 
---
 
#### `services/sc-mongodb/.env`
 
```env
MONGO_INITDB_ROOT_USERNAME=admin         # SECRET
MONGO_INITDB_ROOT_PASSWORD=admin         # SECRET
 
# Seed — primer usuari admin de l'aplicació
ADMIN_EMAIL=admin@smartchrono.local      # SECRET
ADMIN_PASSWORD=admin1234                 # SECRET
ADMIN_DISPLAY_NAME=Administrador
```

#### `services/sc-redis/.env`
 
```env
REDIS_PASSWORD=                      # buit en dev, obligatori en prod # SECRET
```
 
---
 
#### `services/sc-object-storage/.env`
 
```env
MINIO_ROOT_USER=minioadmin           # SECRET
MINIO_ROOT_PASSWORD=minioadmin       # SECRET
```
 
---
 
#### `services/sc-grafana/.env`
 
```env
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin     # SECRET
GF_SERVER_HTTP_PORT=3001
```

#### `services/sc-label-studio/.env`
 
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

#### Variables compartides entre serveis
 
Les variables següents apareixen a més d'un `.env` i han de tenir el mateix valor a tots els serveis on apareguin. En entorns locals es copien manualment; en producció s'injecten via Docker Secrets o el sistema de secrets de l'orquestrador.
 
| Variable | Serveis | Descripció |
| :--- | :--- | :--- |
| `INTERNAL_API_KEY` | api-gateway, video-manager, inference-worker, logic-aggregator, active-learner | Clau de comunicació entre microserveis |
| `REDIS_HOST` / `REDIS_PORT` | api-gateway, video-manager, inference-worker, logic-aggregator | Adreça del broker Redis |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | api-gateway, video-manager, inference-worker, active-learner | Credencials d'accés a MinIO |
| `SENTRY_DSN` | Tots els serveis Python | DSN de Sentry per a reporting d'errors |

### 2.14 Decisions Tècniques per a la Implementació

#### Healthchecks i ordre d'arrencada (docker-compose)

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

#### Convenció d'endpoints REST
 
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

#### Format de respostes de l'API
 
**Resposta correcta:** format lliure segons el recurs, definit per cada endpoint. FastAPI serialitza automàticament els models Pydantic.
 
**Resposta d'error:** format estàndard FastAPI sense modificacions.

Els errors 500 mai exposen stack traces al client. El stack trace complet s'envia únicament a Sentry (vegeu punt 2.5).

#### Política CORS
 
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

### 2.15 Protocol de Missatges de les Cues Redis

Tots els missatges publicats a les cues Redis són objectes JSON serialitzats. Cap servei pot publicar un missatge sense els camps obligatoris definits aquí.

#### Cua `video_to_process` — Publicador: `sc-api-gateway` · Consumidor: `sc-video-manager`

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

#### Cua `task_frames` — Publicador: `sc-video-manager` · Consumidor: `sc-inference-worker`
 
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

#### Cua `detected_frames_results` — Publicador: `sc-inference-worker` · Consumidor: `sc-logic-aggregator`
 
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

#### Regles generals
 
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

### 2.16 `sc-video-manager` — Arquitectura Worker Pur
 
`sc-video-manager` és un **worker pur basat en Redis**. No exposa cap endpoint HTTP ni port al host. Tota la comunicació amb la resta del sistema es fa exclusivament a través de la cua `video_to_process` de Redis (entrada) i de MongoDB (escriptura d'estat).

#### Per què worker pur i no servidor HTTP
 
- L'`sc-api-gateway` ja és l'únic punt d'entrada HTTP del sistema. Afegir un servidor HTTP a `sc-video-manager` crearia un segon punt d'entrada que hauria d'autenticar-se, mantenir-se i monitoritzar-se innecessàriament.
- Redis com a broker ja proporciona el canal de comunicació. Publicar un missatge a `video_to_process` és suficient per desencadenar qualsevol tipus de feina.
- Un worker que cau i es reinicia simplement reprèn la cua des d'on estava. Un servidor HTTP perd les peticions en vol.

#### Cicle de vida del worker
 
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

#### Estats que escriu a MongoDB
 
| Moment | Camp actualitzat | Valor |
| :--- | :--- | :--- |
| Inici de processament de partit | `matches.status` | `"processing"` |
| Tots els frames extrets i pujats | `matches.status` | `"frames_ready"` |
| Error durant l'extracció | `matches.status` | `"error"` |

Per a feines de tipus `process_labeling` no s'escriu res a MongoDB — no hi ha cap document de partit associat.
 
#### Gestió d'errors
 
Si qualsevol pas falla (descàrrega de MinIO, FFmpeg, upload de frame), el worker:
1. Actualitza `matches.status → "error"` a MongoDB (si és un `process_match`).
2. Registra l'error complet a Sentry amb el `match_id` o `session_id` com a context.
3. **No reencua el missatge** — el descarta i torna a escoltar la cua.
4. Continua processant el proper missatge normalment.

### 2.17 Protocol de Promoció de Models
 
Quan `sc-active-learner` entrena un nou model i supera les mètriques d'acceptació (vegeu punt 4.5), publica un missatge a la cua `model_promoted` de Redis. `sc-inference-worker` escolta aquesta cua i carrega el nou model en calent sense reiniciar el contenidor.
 
---
 
#### Cua `model_promoted` — Publicador: `sc-active-learner` · Consumidor: `sc-inference-worker`
 
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
 
#### Flux complet de promoció
 
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
 
#### Comportament de `sc-inference-worker` durant la càrrega
 
- El worker **no atura** el processament de frames mentre descarrega el nou model. Continua usant el model anterior fins que la càrrega és completa.
- La substitució és atòmica: el nou model es carrega en memòria completament abans de substituir el punter a l'anterior.
- Si la descàrrega o càrrega falla, el worker manté el model anterior i registra l'error a Sentry. **No reintenta** — el model antic segueix operatiu.
 
#### Models que no superen el llindar
 
Si el nou model no supera les mètriques d'acceptació, `sc-active-learner` el guarda igualment a MinIO com a candidat però amb un prefix `candidate/`:
 
```
models/yolo/weights/v2.pt          ← actiu (ha superat el llindar)
models/yolo/candidate/v3.pt        ← candidat (no ha superat, pendent revisió manual)
```
 
Els candidats mai s'envien a `model_promoted` i `sc-inference-worker` mai els carrega automàticament.

## 3. Flux de Treball i Gestió del Projecte
 
### 3.1 Metodologia
 
El projecte segueix una metodologia **Scrum lleugera** adaptada a un equip de 2 persones. Els sprints tenen una durada d'**1 setmana** i comencen cada dilluns. Tota la gestió de tasques es fa a **Jira**, amb integració MCP per permetre a Claude Code crear, actualitzar i tancar tickets directament des del flux de desenvolupament.
 
**Principi fonamental:** des de la **setmana 1** hi ha d'haver alguna cosa visible i funcional. No es construeix tota la infraestructura en silenci per després mostrar-ho — cada sprint ha de poder ser demostrat.

### 3.2 Integració MCP amb Jira
 
Claude Code té accés al MCP de Jira de l'organització. Això permet:
 
- Crear tickets i épiques automàticament quan s'inicia una nova funcionalitat.
- Moure tickets a `In Progress` quan comença la implementació.
- Tancar tickets amb referència al commit o PR quan es completa una tasca.
- Consultar l'estat del sprint actual per prioritzar la feina.
 
**Regla:** Claude Code ha d'actualitzar el ticket corresponent a Jira **abans de començar** la implementació (mou a `In Progress`) i **en acabar** (mou a `Done` amb nota del que s'ha fet). Mai deixar tickets en `To Do` si la feina ja ha començat.

### 3.3 Estructura de Tickets
 
**Camps obligatoris a cada ticket:**
 
| Camp | Valors |
| :--- | :--- |
| Prioritat | `Alta` · `Mitja` · `Baixa` |
| Etiqueta | `backend` · `frontend` · `ai` · `infra` · `docs` |
| Assignat | Persona responsable |
| Sprint | Sprint actiu |
 
**Estats del tauler:**
```
To Do → In Progress → In Review → Done
```
 
**Convenció de títols:**
```
[ETIQUETA] Descripció breu en infinitiu
Ex: [backend] Implementar endpoint POST /api/v1/matches
Ex: [infra] Configurar docker-compose amb healthchecks
Ex: [frontend] Crear formulari d'upload de vídeo
```

### 3.4 Épiques del Projecte
 
Tots els tickets pertanyen a una d'aquestes épiques:
 
| Èpica | Descripció |
| :--- | :--- |
| `EP-01 Infraestructura Base` | Docker, xarxes, variables d'entorn, healthchecks |
| `EP-02 Autenticació` | Login, JWT, rols, gestió d'usuaris |
| `EP-03 Gestió de Partits` | CRUD matches, upload vídeo, ROI, configuració sessió |
| `EP-04 Pipeline d'IA` | Inference worker, ByteTrack, cronometratge, logic aggregator |
| `EP-05 Frontend` | Totes les pantalles i components visuals |
| `EP-06 Active Learning` | Feedback loop, re-entrenament, gestió de models |
| `EP-07 Observabilitat` | Prometheus, Grafana, Dozzle, Sentry |

### 3.5 Planificació de Sprints
 
La planificació segueix el principi de **visible des del dia 1**: els primers sprints prioritzen tenir una interfície navegable, una API responent i l'eina d'etiquetatge funcional — la base sense la qual la IA no pot existir.
 
#### Sprint 1 — Fonaments visibles + Label Studio operatiu
**Objectiu demostrable:** pots obrir el navegador, fer login com a admin, i entrar a Label Studio per començar a etiquetar frames des de MinIO. L'API respon a `/health`.
 
Tickets:
- `[infra]` Crear `docker-compose.yml` amb tots els serveis, xarxes i healthchecks — **Alta**
- `[infra]` Configurar fitxers `.env.example` per a tots els serveis — **Alta**
- `[infra]` Configurar MinIO amb tots els buckets i polítiques d'accés — **Alta**
- `[infra]` Configurar `sc-label-studio` amb integració S3/MinIO (lectura de `labeling-frames`, escriptura a `datasets`) — **Alta**
- `[backend]` Implementar `GET /health` i estructura base de FastAPI — **Alta**
- `[backend]` Implementar `POST /auth/login` i `POST /auth/refresh` amb JWT — **Alta**
- `[frontend]` Crear pantalla de login amb Vite + Tailwind + crida real a l'API — **Alta**
 
**Nota:** Al final del Sprint 1 ja es pot començar a etiquetar vídeos en paral·lel al desenvolupament dels sprints següents. Això és crític perquè el dataset estigui llest quan arribi el Sprint 4.

#### Sprint 2 — Gestió de jugadors, equips i pipeline d'etiquetatge
**Objectiu demostrable:** pots crear jugadors a la plantilla, veure'ls llistats, i pujar un vídeo des del frontend per trossejar-lo automàticament i tenir els frames disponibles a Label Studio.
 
Tickets:
- `[backend]` CRUD complet `GET/POST/PATCH /api/v1/players` — **Alta**
- `[backend]` CRUD complet `GET/POST/PATCH /api/v1/teams` — **Alta**
- `[backend]` Implementar middleware d'autorització per rols — **Alta**
- `[backend]` Endpoint d'upload de vídeo per a etiquetatge → `labeling-videos` de MinIO — **Alta**
- `[ai]` Implementar trossejament automàtic de vídeo d'etiquetatge (1 frame/2s → `labeling-frames`) — **Alta**
- `[frontend]` Pantalla de gestió de plantilla (llistat + formulari crear/editar jugador) — **Alta**
- `[frontend]` Secció d'etiquetatge a l'admin: upload de vídeo + link a Label Studio — **Alta**
 
**Nota:** Al final del Sprint 2 el flux complet d'etiquetatge és operatiu. L'equip pot etiquetar frames en paral·lel mentre es desenvolupa la resta del sistema.

#### Sprint 3 — Creació de partits i upload de vídeo
**Objectiu demostrable:** pots crear un partit, pujar un vídeo i veure'l a la llista de partits amb estat `pending`.
 
Tickets:
- `[backend]` CRUD `GET/POST /api/v1/matches` + upload a MinIO — **Alta**
- `[backend]` Endpoint per definir `start_frame`, `end_frame` i `roi_polygon` — **Alta**
- `[frontend]` Pantalla de creació de partit amb formulari i upload de vídeo — **Alta**
- `[frontend]` Llistat de partits amb estat i data — **Alta**
- `[frontend]` Pantalla de gestió d'equips — **Mitja**
 
#### Sprint 4 — Pipeline d'IA (fase 1): frames i inferència
**Objectiu demostrable:** pots iniciar el processament d'un partit i veure per Dozzle com els frames s'extreuen i la GPU treballa.
 
**Prerequisit:** el dataset YOLO ha d'estar etiquetat i exportat a `datasets` de MinIO (feina feta en paral·lel des del Sprint 1).
 
Tickets:
- `[infra]` Configurar Redis amb cues `video_to_process` i `task_frames` — **Alta**
- `[ai]` Implementar `sc-video-manager`: extracció de frames a MinIO — **Alta**
- `[ai]` Entrenar YOLO v1 sobre el dataset etiquetat i pujar pesos a `models/yolo/weights/v1.pt` — **Alta**
- `[ai]` Implementar `sc-inference-worker`: consum de cua + YOLOv8 + CNN — **Alta**
- `[backend]` Endpoint `POST /api/v1/matches/{id}/process` per iniciar pipeline — **Alta**
- `[infra]` Configurar Dozzle per visualitzar logs en temps real — **Mitja**
 
#### Sprint 5 — Pipeline d'IA (fase 2): tracking i cronometratge
**Objectiu demostrable:** pots veure els minuts jugats per cada jugador actualitzant-se a MongoDB mentre el pipeline processa.
 
Tickets:
- `[ai]` Implementar `sc-logic-aggregator`: ByteTrack + lògica ROI + histeresi — **Alta**
- `[ai]` Càlcul de `seconds_played` i escriptura d'intervals a `match_players` — **Alta**
- `[backend]` Endpoint `GET /api/v1/matches/{id}/players` amb minuts per jugador — **Alta**
- `[frontend]` Pantalla de detall de partit amb llistat de jugadors i minuts — **Alta**
 
#### Sprint 6 — Vídeo de sortida i resultats finals
**Objectiu demostrable:** en acabar el processament, pots descarregar el vídeo amb els overlays i exportar un CSV amb els minuts.
 
Tickets:
- `[ai]` Implementar muntatge de vídeo final amb overlays a `sc-video-manager` — **Alta**
- `[backend]` Endpoint `GET /api/v1/matches/{id}/export` — CSV amb minuts per jugador — **Alta**
- `[frontend]` Botó de descàrrega de vídeo processat i exportació CSV — **Alta**
- `[frontend]` Pantalla de resultats finals del partit — **Mitja**
 
#### Sprint 7 — Observabilitat i Active Learning
**Objectiu demostrable:** Grafana mostra mètriques de la GPU i el sistema pot iniciar un re-entrenament automàtic.
 
Tickets:
- `[infra]` Configurar Prometheus + Grafana amb dashboard de mètriques GPU/CPU — **Mitja**
- `[ai]` Implementar `sc-active-learner`: detecció de feedback i re-entrenament — **Mitja**
- `[backend]` Endpoint per marcar frames com a feedback manual — **Mitja**
- `[infra]` Configurar Sentry en tots els serveis Python — **Mitja**
 
#### Sprint 8 — Poliment i estabilitat
**Objectiu demostrable:** el sistema complet funciona d'extrem a extrem sense errors coneguts. Llest per a revisió acadèmica.
 
Tickets:
- `[frontend]` Revisió UX general: missatges d'error, estats de càrrega, responsive — **Alta**
- `[backend]` Tests d'integració dels endpoints principals — **Alta**
- `[docs]` Documentació de l'API (FastAPI OpenAPI auto-generat) — **Mitja**
- `[infra]` Revisió de seguretat: secrets, CORS, headers HTTP — **Alta**

### 3.6 Directiva de Treball amb Claude Code
 
Abans de fer qualsevol canvi significatiu (nou mòdul, refactorització, canvi d'esquema de BD, nou servei Docker), s'ha de seguir obligatòriament aquest protocol:
 
1. **Consultar Jira** via MCP per identificar el ticket actiu del sprint actual.
2. **Moure el ticket a `In Progress`** abans d'escriure cap línia de codi.
3. **Crear un pla d'implementació** en format Markdown a `/docs/implementation-plans/` amb el format de nom `YYYY-MM-DD_nom-del-canvi.md` descrivint els fitxers afectats, les decisions tècniques i els riscos potencials.
4. **Esperar confirmació explícita** abans d'implementar.
5. **Implementar per fases verificables**, no tot d'un cop.
6. **Reportar al final de cada fase** què s'ha fet, què s'ha canviat i si cal reiniciar algun servei.
7. **Moure el ticket a `Done`** amb una nota breu del que s'ha implementat.
 
Aquesta directiva s'aplica sempre, independentment de com estigui formulada la petició.

## 4. Estratègia d'Entrenament dels Models d'IA

### 4.0 Condicions Reals de Gravació

Les imatges del pavelló revelen condicions específiques que condicionen directament l'estratègia d'entrenament:

| Condició | Observació | Impacte |
| :--- | :--- | :--- |
| **Angle de càmera** | Elevat des de la graderia, no cenital pur. Perspectiva en diagonal | Els jugadors al fons apareixen més petits que els del davant. El YOLO ha d'aprendre aquesta variació d'escala |
| **Distorsió òptica** | Lleu efecte ull de peix visible a les cantonades | Cal aplicar correcció de distorsió (undistort) com a pas de preprocessat abans de la inferència |
| **Backlight** | Finestres grans al fons generen contrallum fort | Els jugadors a la meitat del camp apareixen en semisombra. L'augmentació de dades ha d'incloure variacions de llum agressives |
| **Mida del dorsal** | Jugadors propers: dorsal llegible. Jugadors al fons: dorsal de 10–15px d'alçada | La CNN ha de ser robusta a dorsals de molt baixa resolució. `INPUT_SIZE` s'ha d'ajustar a la mida real dels crops |
| **Resolució de la càmera habitual** | Qualitat moderada (càmera d'acció) | La majoria del dataset serà d'aquesta qualitat — el model ha d'estar entrenat principalment sobre això |
| **Càmera professional** | 1 sol partit disponible, resolució superior | Útil com a conjunt de validació o per generar crops d'alta qualitat per a casos difícils |
 
**Conseqüència crítica:** el dataset d'entrenament ha de reflectir la qualitat real de la càmera habitual, **no** la càmera professional. Un model entrenat majoritàriament sobre imatges d'alta qualitat fallarà en producció.

### 4.1 Visió General
 
El sistema utilitza **dos models independents** amb responsabilitats i estratègies d'entrenament diferenciades:
 
| Model | Responsabilitat | Arquitectura | Input |
| :--- | :--- | :--- | :--- |
| **YOLO (detecció)** | Localitzar jugadors a la pista i generar bounding boxes | YOLOv8 (fine-tuned) | Frame complet |
| **CNN (dorsals)** | Llegir el número del dorsal a partir d'un crop del jugador | CNN personalitzada | Crop de la zona dorsal |
 
**Principi fonamental:** cap dels dos models s'entrena sobre la identitat dels jugadors ni sobre persones concretes. El YOLO detecta la classe `player` de forma genèrica, i la CNN llegeix un número d'1 a 99. Això fa el sistema transferible entre categories, temporades i altures de jugadors sense necessitat de re-entrenament complet.

### 4.2 Model 1 — YOLO: Detecció de Jugadors

#### Objectiu

Detectar tots els jugadors del nostre equip dins del frame i retornar els seus bounding boxes. **No cal distingir jugadors individuals** — això ho fa ByteTrack. Sí cal filtrar per equipació (el nostre equip vs. rival i àrbitres).

#### Estratègia: Fine-tuning sobre YOLOv8 preentrenat
 
No entrenant des de zero. YOLOv8 preentrenat sobre COCO ja sap detectar persones amb alta precisió. El fine-tuning serveix per especialitzar-lo en:
 
- **Angle cenital/picat** de càmera fixa al sostre — perspectiva molt diferent a les imatges de COCO.
- **Filtratge per equipació** — aprendre a distingir la nostra samarreta dels rivals i àrbitres pel color i patró.
- **Escala de jugadors** pròpia del nostre pavelló (resolució i distància constants).

#### Dataset per al YOLO

**Font principal:** vídeos gravats al club amb la càmera fixa.

Procés d'extracció i etiquetatge:
1. L'admin puja el vídeo a etiquetar des del frontend (secció d'etiquetatge) → `sc-api-gateway` el puja al bucket `labeling-videos` de MinIO.
2. El sistema llança `sc-video-manager` per trossejar el vídeo a intervals regulars (1 frame cada 2 segons = ~1.800 frames per hora) i els puja al bucket `labeling-frames` de MinIO.
3. **Label Studio** (`sc-label-studio`, servei opcional) llegeix els frames directament des del bucket `labeling-frames` via la integració S3/MinIO nativa. No cal moure ni descarregar res manualment.
4. L'etiquetador marca els bounding boxes amb dues classes: `player_own` (el nostre equip) i `other` (rivals, àrbitres).
5. Label Studio exporta el dataset en format YOLO directament al bucket `datasets` de MinIO, llest per ser consumit per `sc-active-learner`.

**Volum mínim recomanat:**
- 3.000–5.000 frames etiquetats per obtenir un model robust.
- Amb 1 frame cada 2 segons d'1 hora de vídeo → ~1.800 frames per partit. Amb 2-3 partits etiquetats ja es pot fer un primer entrenament viable.

**Preprocessat obligatori abans d'etiquetar i entrenar:**
- Correcció de distorsió d'ull de peix (`cv2.undistort`) amb els paràmetres de la càmera habitual. Calibrar usant les línies rectes de la pista com a referència.

**Augmentació de dades** (via Roboflow o Albumentations):
- Flip horitzontal (la pista és simètrica).
- Variacions de brillantor i contrast **agressives** — simular el backlight de les finestres del fons (jugadors en semisombra).
- Soroll gaussià i reducció de resolució deliberada — simular la qualitat real de la càmera habitual.
- Variació d'escala — jugadors propers (grans) i jugadors al fons (petits a causa de la perspectiva).
- **No** rotar ni fer flip vertical — la càmera és fixa i l'angle sempre és el mateix.

#### Configuració d'entrenament
 
```yaml
# yolo_finetune_config.yaml
model: yolov8m.pt          # Base preentrenada (medium — bon balanç velocitat/precisió)
data: dataset/yolo/data.yaml
epochs: 50
imgsz: 1280                # Resolució alta per càmera fixa cenital
batch: 16
lr0: 0.001
freeze: 10                 # Congela les primeres 10 capes (extractor de features COCO)
classes: 2                 # player_own, other
device: 0                  # GPU
```

**Mètriques d'acceptació:** mAP@0.5 > 0.85 sobre el conjunt de validació.

### 4.3 Model 2 — CNN: Reconeixement de Dorsals

#### Objectiu
Donat un crop de la zona del dorsal d'un jugador, retornar el número (1–99) amb una puntuació de confiança. Si la confiança és inferior a `INFERENCE_CONFIDENCE_THRESHOLD=0.6`, el resultat es descarta i el crop s'afegeix a `feedback-data` per al re-entrenament.

#### Per què una CNN pròpia i no OCR genèric?

L'OCR genèric (Tesseract, EasyOCR) falla en aquest domini per diverses raons: el dorsal apareix en moviment i parcialment desenfocats, la font és específica de l'equipació del club, hi ha oclusió parcial freqüent, i l'angle de la càmera fixa genera distorsió de perspectiva. Una CNN entrenada específicament sobre crops del nostre club aprèn exactament aquestes condicions.

#### Estratègia: classificació de 99 classes (1–99)

El problema es tracta com una **classificació multiclasse** (99 classes) i no com a OCR seqüencial. Això simplifica molt l'arquitectura i és suficient per al rang 1–99.

**Arquitectura base recomanada:** MobileNetV3-Small o EfficientNet-B0 (lleugers, ràpids, bons per a imatges petites).

#### Dataset per a la CNN

**Problema principal:** els crops de dorsals de vídeos reals son petits (~64×64px), moguts i parcialment tapats. Cal un dataset gran i variat.

**Estratègia en 3 capes:**
 
**Capa 1 — Dades sintètiques (punt de partida ràpid):**
Generar imatges sintètiques de dorsals amb la mateixa font, colors i estil de l'equipació del club. Script Python amb PIL/Pillow:
- Fons del color de la samarreta del club.
- Números 1–99 amb la font real de l'equipació.
- Augmentació: rotació ±15°, perspectiva, soroll, desenfoc de moviment, oclusió parcial aleatòria.
- Generar 500–1.000 imatges per classe = 50.000–100.000 imatges sintètiques totals.

Això permet tenir un model base funcional **sense necessitat d'etiquetar res manualment** al principi.

**Capa 2 — Crops reals dels vídeos del club (millora de qualitat):**
Extreure crops reals de dorsals dels vídeos existents usant el YOLO ja entrenat. Etiquetar el número de dorsal de cada crop.
- Objectiu: 200–500 crops reals per dorsal actiu (els dorsals que realment fan servir els jugadors del club).
- No cal cobrir tots els números 1–99 amb dades reals — els sintètics cobreixen la cua llarga.

**Capa 3 — Active Learning continu (millora automàtica):**
Els crops amb confiança < 0.6 s'acumulen a `feedback-data` de MinIO. Quan `TRAINING_MIN_SAMPLES=50` nous crops estan disponibles, `sc-active-learner` llança un fine-tuning automàtic i genera una nova versió del model. Vegeu punt 2.9 i 2.3 Fase D.

#### Configuració d'entrenament
 
```python
# cnn_training_config.py
BASE_MODEL = "efficientnet_b0"  # Preentrenat ImageNet
NUM_CLASSES = 99                # Dorsals 1–99
INPUT_SIZE = (48, 48)           # Ajustat a la mida real dels crops (dorsal ~10-15px al fons)
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0005
FREEZE_BACKBONE = True          # Primera fase: entrenar només el cap classificador
UNFREEZE_AFTER_EPOCH = 15       # Segona fase: fine-tuning complet
DROPOUT = 0.3
```

**Nota sobre la càmera professional:** el partit gravat amb la càmera professional (alta resolució) s'usa exclusivament com a **conjunt de validació** — mai per entrenar. Permet mesurar el límit superior de precisió del model en condicions ideals i detectar si el model generalitza bé o sobreajusta a la qualitat baixa.

**Mètriques d'acceptació:** accuracy Top-1 > 0.80 i Top-3 > 0.92 sobre el conjunt de validació de la càmera habitual. S'accepta un llindar lleugerament inferior al teòric donades les condicions reals de llum i resolució.

### 4.4 Pipeline d'Etiquetatge Recomanat
 
Per aprofitar els vídeos existents del club de forma eficient:
 
```
Vídeos del club
      │
      ▼
Extracció de frames (1 frame / 2s)
      │
      ▼
Etiquetatge YOLO (Roboflow)        ← ~2-3 dies de feina manual
      │
      ▼
Entrenament YOLO v1
      │
      ▼
YOLO detecta jugadors automàticament en nous vídeos
      │
      ▼
Extracció automàtica de crops de dorsals
      │
      ▼
Etiquetatge CNN (només el número)  ← molt més ràpid, crops petits
      │
      ▼
Entrenament CNN v1 (sintètic + real)
      │
      ▼
Sistema en producció → Active Learning automàtic
```

### 4.5 Gestió de Versions dels Models
 
Els models entrenats es guarden al bucket `models` de MinIO amb versionat incremental (vegeu punt 2.9). El model actiu és sempre el de versió més alta disponible.
 
**Política de promoció:** un nou model generat per `sc-active-learner` no substitueix l'actiu automàticament. Primer es valida sobre un conjunt de test fix (`models/eval/test_set/`) i només es promou si supera les mètriques d'acceptació definides als punts 4.2 i 4.3. Si no les supera, es guarda igualment com a versió candidata per a revisió manual.
 
```
models/
├── yolo/
│   ├── weights/
│   │   ├── v1.pt       ← actiu
│   │   ├── v2.pt       ← candidat pendent de validació
│   └── eval/
│       └── test_set/   ← conjunt de validació fix (mai s'usa per entrenar)
├── cnn/
│   ├── weights/
│   │   ├── v1.keras    ← actiu
│   │   └── v2.keras
│   └── eval/
│       └── test_set/
```

### 4.6 Consideracions de Privacitat
 
Els vídeos de partits contenen imatges de menors d'edat (categories de base). Cal tenir en compte:
 
- Els vídeos d'entrenament **mai surten del servidor local** del club — no es pugen a serveis externs com Roboflow Cloud ni Google Colab.
- L'etiquetatge es fa en local amb eines auto-hostatjades (Label Studio).
- El dataset final (crops de dorsals) no conté cares ni és identificable — conté únicament retalls de samarretes amb números.

## 5. Documentació d'Endpoints

El fitxer /docs/endpoints.md ha de mantenir-se actualitzat cada vegada que s'afegeixi o modifiqui un endpoint. És responsabilitat de Claude Code actualitzar-lo com a part de qualsevol pla d'implementació que afecti l'API. Un exemple de com funcionaria el fitxer és el següent:

```markdown
# Endpoints del Backend

> Base URL: `https://[domini]/api/v1`

## Titol
| Mètode | Ruta | Rol mínim | Descripció |
|--------|------|-----------|------------|
| POST | /auth/login | públic | Emet Access Token + Refresh Token cookie |
```

# 6. Documentació de decisions

Utilitzarem aquesta plantilla per saber les decissions que es van prendre.

```markdown
# Decisions Tècniques

## Format
### [DATA] Títol
- **Context:** Per què calia decidir
- **Decisió:** Què s'ha triat  
- **Alternativa descartada:** Què i per què
```

## 7. Estructura de Directoris del Repositori

El projecte és un **monorepo** amb tots els serveis al mateix repositori. L'arrel conté els serveis Docker a `services/`, les eines d'entrenament manual a `training_pipeline/` (fora de Docker) i la documentació a `docs/`.

### 7.1 Estructura General
 
```
smartchrono-ip/
├── docker-compose.yml
├── .gitignore
├── README.md
│
├── docs/
│   ├── specs.md                          ← aquest document
│   ├── endpoints.md                      ← actualitzat per Claude Code
│   ├── decisions.md                      ← registre de decisions tècniques
│   └── implementation-plans/
│       └── YYYY-MM-DD_nom-del-canvi.md   ← plans de Claude Code
│
├── services/
│   ├── sc-api-gateway/
│   ├── sc-video-manager/
│   ├── sc-inference-worker/
│   ├── sc-logic-aggregator/
│   ├── sc-active-learner/
│   ├── sc-frontend/
│   ├── sc-label-studio/
│   ├── sc-mongodb/
│   ├── sc-redis/
│   ├── sc-object-storage/
│   ├── sc-prometheus/
│   ├── sc-grafana/
│   └── sc-dozzle/
│
└── training_pipeline/                    ← eina externa, fora de Docker
    ├── config/
    ├── frames/
    ├── label_studio_data/
    ├── label_studio_export/
    ├── yolo_dataset/
    ├── runs/
    ├── tasks/
    ├── preparar_dades.py
    ├── organitzar_dataset_yolo.py
    ├── entrenar_yolo.py
    └── verificar_sistema.py
```

### 7.2 Estructura Interna dels Serveis Python (FastAPI — 3 capes)
 
Tots els serveis Python (`sc-api-gateway`, `sc-video-manager`, `sc-inference-worker`, `sc-logic-aggregator`, `sc-active-learner`) segueixen la mateixa estructura interna de **3 capes**: esquema, repositori i servei.
 
```
services/sc-api-gateway/
├── Dockerfile
├── .env                    ← exclòs de git
├── .env.example
├── requirements.txt
└── app/
    ├── main.py             ← entrypoint: setup_logging(), app FastAPI, routers
    ├── config.py           ← lectura de variables d'entorn (pydantic Settings)
    ├── dependencies.py     ← dependències compartides (auth, db clients)
    │
    ├── schemas/            ← CAPA 1: models Pydantic (request/response)
    │   ├── __init__.py
    │   ├── match.py
    │   ├── player.py
    │   ├── auth.py
    │   └── ...
    │
    ├── repositories/       ← CAPA 2: accés a dades (MongoDB, Redis, MinIO)
    │   ├── __init__.py
    │   ├── match_repository.py
    │   ├── player_repository.py
    │   ├── auth_repository.py
    │   └── ...
    │
    ├── services/           ← CAPA 3: lògica de negoci
    │   ├── __init__.py
    │   ├── match_service.py
    │   ├── player_service.py
    │   ├── auth_service.py
    │   └── ...
    │
    └── routers/            ← endpoints FastAPI (criden services/)
        ├── __init__.py
        ├── matches.py
        ├── players.py
        ├── auth.py
        └── ...
```

**Regles de les 3 capes:**
 
| Capa | Responsabilitat | Pot importar |
| :--- | :--- | :--- |
| `schemas/` | Definir l'estructura de les dades d'entrada i sortida (Pydantic). Cap lògica. | — |
| `repositories/` | Totes les operacions de base de dades, Redis i MinIO. Cap lògica de negoci. | `schemas/` |
| `services/` | Lògica de negoci pura. Orquestra repositoris. | `schemas/`, `repositories/` |
| `routers/` | Rebre peticions HTTP, validar amb schemas, cridar services, retornar respostes. | `schemas/`, `services/` |

Els `routers/` mai criden directament els `repositories/`. Tot passa per `services/`.

### 7.3 Estructura dels Serveis No-Python
 
#### `services/sc-frontend/`
 
```
services/sc-frontend/
├── Dockerfile
├── .env
├── .env.example
├── package.json
├── vite.config.ts
├── tailwind.config.ts
├── tsconfig.json
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── store/              ← Zustand (access token en memòria)
    ├── api/                ← clients Axios + interceptor de refresh
    ├── components/         ← components reutilitzables
    ├── pages/              ← pantalles (Login, Matches, Players...)
    ├── hooks/              ← custom hooks
    └── types/              ← tipus TypeScript
```

#### `services/sc-mongodb/`
 
```
services/sc-mongodb/
├── .env
├── .env.example
└── init/
    └── 01-init.js          ← script d'inicialització: crea índexos i usuaris
```

El script d'inicialització crea els índexos i el primer usuari admin llegint les variables d'entorn. S'executa automàticament quan MongoDB arrenca per primera vegada (carpeta `init/` muntada com a volum).

```javascript
// Connexió a sc-auth-db
const authDb = db.getSiblingDB('sc-auth-db');
 
// Índexos sc-auth-db
authDb.users.createIndex({ email: 1 }, { unique: true });
authDb.refresh_tokens.createIndex({ expires_at: 1 }, { expireAfterSeconds: 0 });
authDb.refresh_tokens.createIndex({ user_id: 1 });
 
// Connexió a sc-app-db
const appDb = db.getSiblingDB('sc-app-db');
 
// Índexos sc-app-db
// players — índex compost team_id + dorsal (unique)
appDb.players.createIndex({ team_id: 1, dorsal: 1 }, { unique: true });
 
// matches — índexos nous amb team_id
appDb.matches.createIndex({ team_id: 1 });
appDb.matches.createIndex({ team_id: 1, status: 1 });
appDb.matches.createIndex({ team_id: 1, date: -1 });
appDb.matches.createIndex({ status: 1 });
appDb.matches.createIndex({ date: -1 });

appDb.match_players.createIndex({ match_id: 1 });
appDb.match_players.createIndex({ match_id: 1, player_id: 1 }, { unique: true });
appDb.events.createIndex({ match_id: 1, type: 1 });
appDb.events.createIndex({ match_id: 1, frame_number: 1 });
appDb.teams.createIndex({ coach_id: 1 });
appDb.teams.createIndex({ category: 1 });
appDb.user_profiles.createIndex({ user_id: 1 }, { unique: true });
 
// Seed: primer usuari admin
// Les variables d'entorn s'injecten via docker-compose (env_file)
const adminEmail = process.env.ADMIN_EMAIL;
const adminPassword = process.env.ADMIN_PASSWORD;
const adminDisplayName = process.env.ADMIN_DISPLAY_NAME || 'Administrador';
 
const existingAdmin = authDb.users.findOne({ email: adminEmail });
if (!existingAdmin) {
  // NOTA: el password_hash el genera sc-api-gateway en el primer login.
  // Aquí guardem el password en clar temporalment amb un flag force_reset: true.
  // En el primer login, l'API detecta force_reset, fa el hash bcrypt i l'actualitza.
  authDb.users.insertOne({
    email: adminEmail,
    password_hash: adminPassword,
    force_reset: true,
    role: 'admin',
    active: true,
    created_at: new Date(),
    last_login: null
  });
 
  const adminUser = authDb.users.findOne({ email: adminEmail });
 
  appDb.user_profiles.insertOne({
    user_id: adminUser._id,
    display_name: adminDisplayName,
    team_ids: [],
    player_id: null
  });
 
  print('Seed: usuari admin creat → ' + adminEmail);
} else {
  print('Seed: usuari admin ja existeix, omès.');
}
```

**Comportament de `force_reset`:**
- El flag `force_reset: true` indica que el `password_hash` conté el password en clar (provisional).
- En el primer `POST /auth/login`, `sc-api-gateway` detecta `force_reset: true`, fa el hash bcrypt del password rebut, actualitza el document i elimina el flag.
- A partir d'aquest moment el funcionament és idèntic a qualsevol altre usuari.
- Si `force_reset` és `false` o absent, el login segueix el flux normal de bcrypt.

#### `services/sc-object-storage/`
 
```
services/sc-object-storage/
├── .env
├── .env.example
└── init/
    └── create-buckets.sh   ← crea tots els buckets en l'arrencada inicial
```

#### `services/sc-prometheus/`
 
```
services/sc-prometheus/
└── prometheus.yml          ← configuració de scraping (targets de tots els serveis)
```
 
#### `services/sc-grafana/`
 
```
services/sc-grafana/
├── .env
├── .env.example
└── provisioning/
    ├── datasources/
    │   └── prometheus.yml
    └── dashboards/
        └── smartchrono.json
```
 
#### `services/sc-label-studio/`
 
```
services/sc-label-studio/
├── .env
├── .env.example
└── init/
    └── setup-project.sh    ← crea el projecte d'etiquetatge i connecta MinIO
```

### 7.4 `training_pipeline/` — Eina d'Entrenament Inicial (fora de Docker)
 
Aquesta carpeta conté els scripts per a l'entrenament inicial dels models (YOLO v1 i CNN v1). S'executa manualment en local, fora del sistema Docker, amb accés directe a MinIO per descarregar datasets i pujar pesos entrenats.

```
training_pipeline/
├── README.md               ← instruccions d'ús pas a pas
├── requirements.txt        ← dependències Python (ultralytics, torch, etc.)
├── config/
│   ├── yolo_finetune.yaml  ← configuració d'entrenament YOLO (vegeu punt 4.2)
│   └── cnn_training.py     ← configuració d'entrenament CNN (vegeu punt 4.3)
│
├── preparar_dades.py       ← descarrega dataset de MinIO i prepara estructura
├── organitzar_dataset_yolo.py  ← converteix export de Label Studio a format YOLO
├── entrenar_yolo.py        ← fine-tuning YOLOv8, puja v1.pt a MinIO
├── entrenar_cnn.py         ← entrenament EfficientNet-B0, puja v1.keras a MinIO
├── verificar_sistema.py    ← valida que els models funcionen sobre frames reals
│
├── frames/                 ← frames temporals descarregats de MinIO (ignorats per git)
├── label_studio_data/      ← dades de Label Studio (ignorades per git)
├── label_studio_export/    ← exports de Label Studio (ignorats per git)
├── yolo_dataset/           ← dataset en format YOLO preparat (ignorat per git)
└── runs/                   ← resultats d'entrenament locals (ignorats per git)
```
 
**Flux d'ús:**
1. `python preparar_dades.py` — descarrega el dataset exportat de MinIO (`datasets/yolo/v1/`)
2. `python organitzar_dataset_yolo.py` — converteix a format YOLO
3. `python entrenar_yolo.py` — entrena i puja `models/yolo/weights/v1.pt` a MinIO
4. `python entrenar_cnn.py` — entrena i puja `models/cnn/weights/v1.keras` a MinIO
5. `python verificar_sistema.py` — valida els models sobre frames de test

### 7.5 `.gitignore` Global
 
```gitignore
# Secrets
**/.env
!**/.env.example
 
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/
 
# Node
node_modules/
dist/
.next/
 
# Training pipeline (dades locals, mai al repositori)
training_pipeline/frames/
training_pipeline/label_studio_data/
training_pipeline/label_studio_export/
training_pipeline/yolo_dataset/
training_pipeline/runs/
training_pipeline/tasks/
 
# Models (pesos entrenats, van a MinIO no al repo)
*.pt
*.keras
*.onnx
 
# Misc
.DS_Store
*.log
```

### 7.6 Convencions de Codi
 
- **Python:** tots els serveis usen `ruff` per a linting i formatació.
- **TypeScript/React:** `eslint` + `prettier`.
- **Imports:** absoluts des de l'arrel del paquet (`from app.services.match_service import ...`), mai relatius amb `..`.
- **Noms de fitxers:** `snake_case` per a Python, `PascalCase` per a components React, `kebab-case` per a fitxers de configuració.