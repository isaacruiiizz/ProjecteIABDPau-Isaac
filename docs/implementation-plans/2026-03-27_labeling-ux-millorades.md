# Pla d'implementació — Millores UX LabelingPage

**Data:** 2026-03-27
**Estat:** En Procés

---

## Objectiu

Quatre millores sobre la pàgina d'etiquetatge:

1. **Lupa eyedropper en temps real** — en lloc de clicar directament al canvas, mostrar una lupa circular que segueix el cursor amb zoom 4×, previsualitza el color en temps real i selecciona al clic.
2. **Precisió de color** — mostrejar la mitjana d'un bloc 5×5 píxels en lloc d'un sol píxel per reduir artefactes JPEG.
3. **Barres de progrés** — (a) durant l'extracció de frames (retry polling); (b) durant la pre-anotació (polling `ls-stats`).
4. **Stats Label Studio + botó netejar** — mostrar quantes tasques hi ha i quantes estan anotades; botó per esborrar-les totes.

---

## Fases

### Fase 1 — Backend: sc-api-gateway [~30 línies]

**1.1** `app/config.py` → afegir `LABEL_STUDIO_URL` + `LABEL_STUDIO_API_TOKEN`

**1.2** `app/services/labeling_service.py` → afegir:
- `get_ls_stats()` → crida `GET /api/projects/1/` a LS → retorna `{total, annotated}`
- `clear_ls_tasks()` → crida `POST /api/dm/actions?id=delete_tasks&project=1` → retorna `{deleted}`

**1.3** `app/routers/labeling.py` → afegir:
- `GET /api/v1/labeling/ls-stats` → `{total_tasks, annotated_tasks}`
- `DELETE /api/v1/labeling/ls-tasks` → `{deleted}`

**1.4** `app/schemas/labeling.py` → afegir `LsStatsResponse`, `LsTasksClearResponse`

**1.5** `.env` + `.env.example` → afegir `LABEL_STUDIO_URL` + `LABEL_STUDIO_API_TOKEN`

### Fase 2 — Frontend: sc-frontend [~200 línies]

**2.1** `api/labeling.ts` → afegir `getLsStats()` + `clearLsTasks()`

**2.2** `LabelingPage.tsx` — canvis:
- **Lupa**: `hoverRgb`, `lensPos` state + `onMouseMove` + overlay circular amb mini-canvas
- **Color accuracy**: `getImageData(x-2, y-2, 5, 5)` + mitjana RGB
- **Barra progrés vídeo**: durant retry (retryCount / maxRetries)
- **Barra progrés etiquetatge**: `pollingRef` actiu post-startResult, crida `getLsStats()` cada 5 s
- **Stats LS**: widget al final amb `total_tasks`, `annotated_tasks`, botó `Netejar`

---

## Decisió de disseny

- La lupa s'implementa com a `<div>` absolutament posicionat dins el contenidor del canvas. Usa un `<canvas>` intern de 80×80 px que dibuixa un crop 20×20 px de la imatge real (zoom 4×). Una franja inferior mostra el color actual.
- El polling de ls-stats s'atura quan `annotated_tasks === total_tasks && total_tasks > 0`, o als 5 minuts.
- Les crides a Label Studio des de l'API Gateway es fan amb `httpx.AsyncClient` (no requests) per no bloquejar l'event loop.
