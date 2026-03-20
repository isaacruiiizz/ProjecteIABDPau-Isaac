# Estructura de Directoris del Repositori

El projecte és un **monorepo** amb tots els serveis al mateix repositori. L'arrel conté els serveis Docker a `services/`, les eines d'entrenament manual a `training_pipeline/` (fora de Docker) i la documentació a `docs/`.

## 1. Estructura General
 
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

## 2. Estructura Interna dels Serveis Python (FastAPI — 3 capes)
 
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

## 3. Estructura dels Serveis No-Python
 
### `services/sc-frontend/`
 
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

### `services/sc-mongodb/`
 
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

### `services/sc-object-storage/`
 
```
services/sc-object-storage/
├── .env
├── .env.example
└── init/
    └── create-buckets.sh   ← crea tots els buckets en l'arrencada inicial
```

### `services/sc-prometheus/`
 
```
services/sc-prometheus/
└── prometheus.yml          ← configuració de scraping (targets de tots els serveis)
```
 
### `services/sc-grafana/`
 
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
 
### `services/sc-label-studio/`
 
```
services/sc-label-studio/
├── .env
├── .env.example
└── init/
    └── setup-project.sh    ← crea el projecte d'etiquetatge i connecta MinIO
```

## 4. `training_pipeline/` — Eina d'Entrenament Inicial (fora de Docker)
 
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

## 5. `.gitignore` Global
 
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

## 6. Convencions de Codi
 
- **Python:** tots els serveis usen `ruff` per a linting i formatació.
- **TypeScript/React:** `eslint` + `prettier`.
- **Imports:** absoluts des de l'arrel del paquet (`from app.services.match_service import ...`), mai relatius amb `..`.
- **Noms de fitxers:** `snake_case` per a Python, `PascalCase` per a components React, `kebab-case` per a fitxers de configuració.