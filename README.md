# SmartChrono IP

Sistema d'anàlisi tàctica de futbol sala mitjançant visió per computador i intel·ligència artificial generativa.

Projecte de final de cicle — CE IABD, Institut Sa Palomera, Blanes · 2025–2026  
Autors: Pau Miró Fabregas · Isaac Ruiz García  
Tutors: S. Rovira · F. Barragan · R. Ventura · M. Floriach

---

## Descripció

SmartChrono IP processa vídeos de partits de futbol sala i genera informes tàctics automàtics. El sistema detecta els jugadors en cada fotograma amb el model **RT-DETR Large**, classifica cada detecció per equip mitjançant un classificador **HSV** calibrat per l'entrenador, calcula estadístiques tàctiques (possessió, zones de pressió, ritme de joc) i sintetitza els resultats en un informe en català generat per un **LLM local Ollama qwen2.5:3b**, sense cap dependència de serveis externs.

---

## Arquitectura

El sistema s'organitza en microserveis orquestrats amb Docker Compose. La comunicació entre serveis es realitza exclusivament via cues Redis i fitxers a MinIO; cap servei intern exposa HTTP directe excepte `sc-api-gateway`.

```
sc-frontend (React 18 + Vite)
      |
      v  HTTP/REST
sc-api-gateway (FastAPI)
      |
      +---> Redis (cues asíncrones)
      |           |
      |           +---> sc-video-manager (FFmpeg, extracció de frames)
      |           +---> sc-inference-worker (RT-DETR + HSV)
      |           +---> sc-logic-aggregator (estadístiques + Ollama)
      |
      +---> sc-mongodb (sc-auth-db + sc-app-db)
      +---> sc-object-storage (MinIO, 9 buckets)
```

### Serveis

| Servei | Tecnologia | Funció |
|---|---|---|
| `sc-api-gateway` | FastAPI 3 capes, port 8000 | Punt d'entrada HTTP, autenticació JWT, coordinació |
| `sc-frontend` | React 18 + Vite + Tailwind, port 3000 | Interfície web de l'entrenador |
| `sc-video-manager` | Python + FFmpeg, worker Redis | Extracció de frames amb ROI i rang de temps |
| `sc-inference-worker` | Python + Ultralytics RT-DETR, worker Redis | Detecció de jugadors i classificació per color |
| `sc-logic-aggregator` | Python + Ollama, worker Redis | Agregació d'estadístiques i generació d'informes |
| `sc-mongodb` | MongoDB 7, ports interns | Dues bases de dades: `sc-auth-db` i `sc-app-db` |
| `sc-object-storage` | MinIO, port 9000/9001 | Emmagatzematge de vídeos, frames i models |
| `sc-redis` | Redis 7, port intern | Cues asíncrones entre serveis |
| `sc-ollama` | Ollama, port intern | LLM local qwen2.5:3b per a informes tàctics |
| `sc-label-studio` | Label Studio, port 8081 | Etiquetatge semi-automàtic (opcional) |

---

## Requisits previs

- Docker Desktop 4.x o superior
- Docker Compose v2
- GPU NVIDIA amb drivers compatibles amb CUDA 12.6 (per a `sc-inference-worker`)
- 16 GB RAM recomanats
- 20 GB d'espai en disc lliure

---

## Posada en marxa

**1. Clonar el repositori**

```bash
git clone <url-del-repositori>
cd ProjecteIABDPau-Isaac
```

**2. Configurar les variables d'entorn**

Cada servei disposa d'un fitxer `.env.example` a la seva carpeta. Copiar i emplenar:

```bash
cp services/sc-api-gateway/.env.example services/sc-api-gateway/.env
cp services/sc-mongodb/.env.example services/sc-mongodb/.env
cp services/sc-object-storage/.env.example services/sc-object-storage/.env
cp services/sc-redis/.env.example services/sc-redis/.env
cp services/sc-frontend/.env.example services/sc-frontend/.env
# (repetir per a la resta de serveis)
```

**3. Descarregar el model RT-DETR**

El model `rtdetr-l.pt` s'ha de pujar manualment a MinIO al bucket `models` amb la clau `yolo/active/rtdetr-l.pt` després de la primera arrencada. Consultar `docs/specs/02-logica-ia.md` per als detalls.

**4. Aixecar el sistema**

```bash
docker compose up -d
```

La interfície web és accessible a `http://localhost:3000`.  
La consola de MinIO és accessible a `http://localhost:9001`.

---

## Flux d'ús

### Processar un partit

1. Accedir a `http://localhost:3000` i iniciar sessió.
2. Crear un nou partit i pujar el vídeo `.mp4`.
3. Iniciar el processament. El sistema extrau els frames, detecta els jugadors, classifica per equip i genera l'informe tàctic.

### Etiquetatge i millora del model (opcional)

1. Aixecar Label Studio: `docker compose up -d sc-label-studio`
2. Accedir a `http://localhost:8081`.
3. Els frames de vídeos de entrenament s'envien automàticament a Label Studio via la cua `video_to_process` amb `job_type: process_labeling`.
4. Revisar les pre-anotacions generades per RF-DETR Small, corregir i exportar.
5. Pujar el dataset a MinIO al bucket `datasets` per al reentrament.

---

## Estructura del repositori

```
ProjecteIABDPau-Isaac/
├── CLAUDE.md                        Instruccions per a Claude Code
├── docker-compose.yml               Orquestració de tots els serveis
├── docs/
│   ├── specs/                       Especificacions tècniques per àmbit
│   │   ├── 01-arquitectura.md
│   │   ├── 02-logica-ia.md
│   │   ├── 03-infraestructura.md
│   │   ├── 04-seguretat-bd.md
│   │   ├── 05-config.md
│   │   ├── 06-projecte.md
│   │   └── 07-estructura.md
│   ├── specs.md                     Especificació completa unificada
│   ├── endpoints.md                 Registre de tots els endpoints de l'API
│   ├── decisions.md                 Decisions tècniques documentades
│   └── implementation-plans/        Plans d'implementació per ticket Jira
├── services/
│   ├── sc-api-gateway/              FastAPI — porta d'entrada HTTP
│   ├── sc-frontend/                 React 18 — interfície web
│   ├── sc-video-manager/            Worker FFmpeg — extracció de frames
│   ├── sc-inference-worker/         Worker RT-DETR + HSV
│   ├── sc-logic-aggregator/         Worker estadístiques + Ollama
│   ├── sc-mongodb/                  MongoDB — init i configuració
│   ├── sc-redis/                    Redis — configuració
│   ├── sc-object-storage/           MinIO — buckets, polítiques IAM
│   ├── sc-label-studio/             Label Studio — etiquetatge (opcional)
│   └── sc-grafana/                  Grafana — observabilitat
└── training_pipeline/               Scripts locals d'entrenament YOLO
```

---

## Documentació d'entrega

La documentació formal del projecte es distribueix en un fitxer ZIP independent i **no forma part del repositori**. El ZIP conté:

| Fitxer | Contingut |
|---|---|
| `SmartChrono_IP_Memoria.docx` | Memòria tècnica completa |
| `SmartChrono_IP_Infografia.html` | Infografia A4 (obrir al navegador per imprimir) |
| `SmartChrono_IP_Resum.html` | Resum executiu A4 (obrir al navegador per imprimir) |
| `SmartChrono_IP_Demo.mp4` | Vídeo de demostració del sistema |

---

## Llicència

Projecte acadèmic. Tots els drets reservats — Institut Sa Palomera, Blanes 2025–2026.
