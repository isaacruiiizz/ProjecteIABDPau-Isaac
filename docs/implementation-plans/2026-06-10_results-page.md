# PJM-69 — ResultsPage: vídeo processat descarregable

**Estat:** Completat ✓ (2026-06-10)
**Data:** 2026-06-10

## Objectiu
Crear una ResultsPage que mostri el vídeo processat descarregable un cop el partit té `status=done`. Accessible des de MatchesPage amb un botó "Veure resultats".

## Canvis

### Backend (sc-api-gateway)
- `schemas/matches.py` → + `MatchDetail`
- `services/match_service.py` → + `get_match_detail()` amb presigned URL de MinIO
- `routers/matches.py` → + `GET /api/v1/matches/{match_id}`

### Frontend (sc-frontend)
- `api/matches.ts` → + `MatchDetail` + `getMatch()`
- `pages/ResultsPage.tsx` → nou fitxer amb hero gradient + download card + stats
- `pages/MatchesPage.tsx` → botó "Veure resultats" per `status=done`
- `App.tsx` → ruta `/matches/:id/results`
