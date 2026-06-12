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

### Generar un informe tàctic

El procés es guia en tres passos des de la interfície web (`http://localhost:3000`).

**Pas 1 — Puja el vídeo**

Accedir a la secció de nou partit. Seleccionar el fitxer `.mp4` del partit (arrossegar o fer clic) i donar-li un títol. Fer clic a "Puja el vídeo" per iniciar la càrrega al servidor.

**Pas 2 — Definir la ROI**

Un cop el vídeo s'ha pujat, el sistema mostra el primer fotograma. Cal marcar els 4 vèrtexs del polígon que delimita el camp de joc, fent clic sobre les quatre cantonades en ordre. Això exclou del processament marcadors, banquetes i zones no rellevants. Els vèrtexs es poden arrossegar per ajustar la posició i desfer o reiniciar si cal.

**Pas 3 — Seleccionar el rang de temps**

A continuació, el sistema mostra el vídeo complet amb un control de línia de temps. Cal arrossegar els dos extrems per definir el segon d'inici i el segon de fi del fragment que es vol analitzar (per exemple, el primer temps, una represa o un fragment concret).

**Processament**

Fer clic a "Iniciar processament". El pipeline s'executa de manera asíncrona:

- `sc-video-manager` extrau els frames dins del rang i la ROI indicats.
- `sc-inference-worker` detecta els jugadors en cada frame amb RT-DETR Large.
- `sc-logic-aggregator` consolida les deteccions, calcula les estadístiques tàctiques i genera l'informe en català amb Ollama qwen2.5:3b.

La interfície mostra el progrés en temps real. Quan el processament finalitza, redirigeix automàticament a la pàgina de resultats amb l'informe tàctic complet.

---

### Etiquetatge i millora del model (opcional)

El flux d'etiquetatge és accessible des del menú d'administrador del frontend i permet generar pre-anotacions automàtiques per a nous vídeos d'entrenament.

**Pas 1 — Pujar el vídeo d'entrenament**

A la pàgina d'etiquetatge, seleccionar el fitxer `.mp4` de referència i configurar l'interval entre frames (per defecte, 1 frame cada 2 segons). Fer clic a "Pujar vídeo". El sistema extreu els frames de manera asíncrona i confirma quan estan disponibles.

**Pas 2 — Seleccionar el color de samarreta**

Un cop extrets els frames, la interfície mostra fotogrames representatius del vídeo (al 10%, 25%, 50%, 75% i 90% de la durada). Cal passar el cursor per sobre de la samarreta d'un jugador de l'equip que s'etiquetarà i fer clic per confirmar la selecció. El sistema captura el valor HSV d'aquella zona. Opcionalment, es pot ajustar la tolerància de color (valor entre 10 i 60) per adaptar-se a variacions d'il·luminació.

**Pas 3 — Iniciar la pre-anotació**

Fer clic a "Iniciar etiquetatge". El sistema envia tots els frames a `sc-inference-worker`, que aplica RF-DETR Small filtrat pel color HSV seleccionat per generar bounding boxes automàtiques. La interfície mostra el progrés de la pre-anotació en temps real.

**Pas 4 — Revisar a Label Studio**

Quan la pre-anotació finalitza, fer clic a "Obrir Label Studio" (accessible a `http://localhost:8081`). Les tasques apareixen preomplertes amb les deteccions automàtiques. Cal revisar-les, corregir les bounding boxes incorrectes i validar les correctes. Un cop completada la revisió, exportar les anotacions en format YOLO des de Label Studio i pujar el dataset resultant al bucket `datasets` de MinIO per al reentrament del model.

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
