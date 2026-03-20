# Flux de Treball i Gestió del Projecte
 
## 1. Metodologia
 
El projecte segueix una metodologia **Scrum lleugera** adaptada a un equip de 2 persones. Els sprints tenen una durada d'**1 setmana** i comencen cada dilluns. Tota la gestió de tasques es fa a **Jira**, amb integració MCP per permetre a Claude Code crear, actualitzar i tancar tickets directament des del flux de desenvolupament.
 
**Principi fonamental:** des de la **setmana 1** hi ha d'haver alguna cosa visible i funcional. No es construeix tota la infraestructura en silenci per després mostrar-ho — cada sprint ha de poder ser demostrat.

## 2. Integració MCP amb Jira
 
Claude Code té accés al MCP de Jira de l'organització. Això permet:
 
- Crear tickets i épiques automàticament quan s'inicia una nova funcionalitat.
- Moure tickets a `In Progress` quan comença la implementació.
- Tancar tickets amb referència al commit o PR quan es completa una tasca.
- Consultar l'estat del sprint actual per prioritzar la feina.
 
**Regla:** Claude Code ha d'actualitzar el ticket corresponent a Jira **abans de començar** la implementació (mou a `In Progress`) i **en acabar** (mou a `Done` amb nota del que s'ha fet). Mai deixar tickets en `To Do` si la feina ja ha començat.

## 3. Estructura de Tickets
 
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

## 4. Épiques del Projecte
 
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

## 5. Planificació de Sprints
 
La planificació segueix el principi de **visible des del dia 1**: els primers sprints prioritzen tenir una interfície navegable, una API responent i l'eina d'etiquetatge funcional — la base sense la qual la IA no pot existir.
 
### Sprint 1 — Fonaments visibles + Label Studio operatiu
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

### Sprint 2 — Gestió de jugadors, equips i pipeline d'etiquetatge
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

### Sprint 3 — Creació de partits i upload de vídeo
**Objectiu demostrable:** pots crear un partit, pujar un vídeo i veure'l a la llista de partits amb estat `pending`.
 
Tickets:
- `[backend]` CRUD `GET/POST /api/v1/matches` + upload a MinIO — **Alta**
- `[backend]` Endpoint per definir `start_frame`, `end_frame` i `roi_polygon` — **Alta**
- `[frontend]` Pantalla de creació de partit amb formulari i upload de vídeo — **Alta**
- `[frontend]` Llistat de partits amb estat i data — **Alta**
- `[frontend]` Pantalla de gestió d'equips — **Mitja**
 
### Sprint 4 — Pipeline d'IA (fase 1): frames i inferència
**Objectiu demostrable:** pots iniciar el processament d'un partit i veure per Dozzle com els frames s'extreuen i la GPU treballa.
 
**Prerequisit:** el dataset YOLO ha d'estar etiquetat i exportat a `datasets` de MinIO (feina feta en paral·lel des del Sprint 1).
 
Tickets:
- `[infra]` Configurar Redis amb cues `video_to_process` i `task_frames` — **Alta**
- `[ai]` Implementar `sc-video-manager`: extracció de frames a MinIO — **Alta**
- `[ai]` Entrenar YOLO v1 sobre el dataset etiquetat i pujar pesos a `models/yolo/weights/v1.pt` — **Alta**
- `[ai]` Implementar `sc-inference-worker`: consum de cua + YOLOv8 + CNN — **Alta**
- `[backend]` Endpoint `POST /api/v1/matches/{id}/process` per iniciar pipeline — **Alta**
- `[infra]` Configurar Dozzle per visualitzar logs en temps real — **Mitja**
 
### Sprint 5 — Pipeline d'IA (fase 2): tracking i cronometratge
**Objectiu demostrable:** pots veure els minuts jugats per cada jugador actualitzant-se a MongoDB mentre el pipeline processa.
 
Tickets:
- `[ai]` Implementar `sc-logic-aggregator`: ByteTrack + lògica ROI + histeresi — **Alta**
- `[ai]` Càlcul de `seconds_played` i escriptura d'intervals a `match_players` — **Alta**
- `[backend]` Endpoint `GET /api/v1/matches/{id}/players` amb minuts per jugador — **Alta**
- `[frontend]` Pantalla de detall de partit amb llistat de jugadors i minuts — **Alta**
 
### Sprint 6 — Vídeo de sortida i resultats finals
**Objectiu demostrable:** en acabar el processament, pots descarregar el vídeo amb els overlays i exportar un CSV amb els minuts.
 
Tickets:
- `[ai]` Implementar muntatge de vídeo final amb overlays a `sc-video-manager` — **Alta**
- `[backend]` Endpoint `GET /api/v1/matches/{id}/export` — CSV amb minuts per jugador — **Alta**
- `[frontend]` Botó de descàrrega de vídeo processat i exportació CSV — **Alta**
- `[frontend]` Pantalla de resultats finals del partit — **Mitja**
 
### Sprint 7 — Observabilitat i Active Learning
**Objectiu demostrable:** Grafana mostra mètriques de la GPU i el sistema pot iniciar un re-entrenament automàtic.
 
Tickets:
- `[infra]` Configurar Prometheus + Grafana amb dashboard de mètriques GPU/CPU — **Mitja**
- `[ai]` Implementar `sc-active-learner`: detecció de feedback i re-entrenament — **Mitja**
- `[backend]` Endpoint per marcar frames com a feedback manual — **Mitja**
- `[infra]` Configurar Sentry en tots els serveis Python — **Mitja**
 
### Sprint 8 — Poliment i estabilitat
**Objectiu demostrable:** el sistema complet funciona d'extrem a extrem sense errors coneguts. Llest per a revisió acadèmica.
 
Tickets:
- `[frontend]` Revisió UX general: missatges d'error, estats de càrrega, responsive — **Alta**
- `[backend]` Tests d'integració dels endpoints principals — **Alta**
- `[docs]` Documentació de l'API (FastAPI OpenAPI auto-generat) — **Mitja**
- `[infra]` Revisió de seguretat: secrets, CORS, headers HTTP — **Alta**

## 6. Directiva de Treball amb Claude Code
 
Abans de fer qualsevol canvi significatiu (nou mòdul, refactorització, canvi d'esquema de BD, nou servei Docker), s'ha de seguir obligatòriament aquest protocol:
 
1. **Consultar Jira** via MCP per identificar el ticket actiu del sprint actual.
2. **Moure el ticket a `In Progress`** abans d'escriure cap línia de codi.
3. **Crear un pla d'implementació** en format Markdown a `/docs/implementation-plans/` amb el format de nom `YYYY-MM-DD_nom-del-canvi.md` descrivint els fitxers afectats, les decisions tècniques i els riscos potencials.
4. **Esperar confirmació explícita** abans d'implementar.
5. **Implementar per fases verificables**, no tot d'un cop.
6. **Reportar al final de cada fase** què s'ha fet, què s'ha canviat i si cal reiniciar algun servei.
7. **Moure el ticket a `Done`** amb una nota breu del que s'ha implementat.
 
Aquesta directiva s'aplica sempre, independentment de com estigui formulada la petició.