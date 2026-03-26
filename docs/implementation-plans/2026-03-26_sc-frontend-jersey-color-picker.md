# Pla d'implementació — Selector de color de samarreta des del frontend

**Data:** 2026-03-26
**Estat:** Completat ✓ (2026-03-26)

---

## Problema

El color de la samarreta del propi equip (`JERSEY_OWN_COLOR_HSV`) es configura avui com a variable d'entorn estàtica. Això és insuficient perquè:

- Un mateix equip pot portar samarretes de colors molt diferents (primera equipació, alternativa, tercera)
- La il·luminació del camp canvia el to percebut entre partits (indoor vs outdoor, llum artificial vs solar)
- L'usuari no pot saber quin valor HSV exacte cal fins que veu el vídeo real

**Nota important:** Els vídeos d'etiquetatge **no es guarden a MongoDB** i no tenen `match_id`. El pipeline d'etiquetatge és independent del pipeline de partits i tota la seva informació viatja via Redis + MinIO.

---

## Solució

Afegir a la pàgina d'etiquetatge (`LabelingPage`) un **eyedropper interactiu** sobre un frame del vídeo:

1. Es mostra un frame representatiu del vídeo (ja extret per `sc-video-manager` a MinIO `labeling-frames/`)
2. L'usuari passa el cursor per sobre d'un jugador del seu equip i fa clic
3. El frontend llegeix el color RGB del píxel clicat i el converteix a HSV
4. Es mostra una previsualització del color seleccionat
5. L'usuari confirma i el color s'inclou com a camp del missatge Redis quan s'inicia el pipeline

---

## Disseny tècnic

### Per què Redis i no MongoDB

Els vídeos d'etiquetatge no tenen persistència a MongoDB — el seu cicle de vida és: upload → frames → inferència → Label Studio. Guardar el color a MongoDB requeriria crear una col·lecció `labeling_jobs` només per a aquest propòsit.

La solució correcta és **incloure `jersey_own_color_hsv` directament al payload de Redis** quan l'usuari llança el pipeline. El missatge és autocontingut: el worker té tot el context que necessita sense fer cap consulta addicional. Aquest és el patró estàndard de les cues de tasques (Celery, BullMQ, Sidekiq).

Fallback: si `jersey_own_color_hsv` és `null` al payload, `sc-inference-worker` usarà la variable d'entorn `JERSEY_OWN_COLOR_HSV`.

### Frontend (`sc-frontend`)

- **`LabelingPage.tsx`** → afegir secció "Color de samarreta" amb:
  - Canvas HTML5 que renderitza el frame seleccionat (signed URL via API)
  - **Navegació entre frames representatius:** botons prev/next que salten entre el 10%, 25%, 50%, 75% i 90% del total de frames extrets. El primer frame s'evita deliberadament perquè sol ser ajust de càmera o pista buida
  - L'API retorna el `total_frames` del vídeo per calcular els índexos dels salts
  - Event listener `onClick` → llegeix píxel del canvas amb `getImageData`, converteix RGB → HSV
  - Component de previsualització del color (quadre de color + valors HSV numèrics)
  - El color seleccionat s'emmagatzema a l'estat local de la pàgina
  - Slider o input numèric per ajustar `jersey_color_threshold` (distància màxima HSV, default: 30, rang recomanat: 10–60). Permet afinar si el detector classifica massa o massa poc jugadors com a propis
  - Botó "Iniciar etiquetatge" → envia `POST /api/v1/labeling/start` amb `{ "video_key": "...", "jersey_own_color_hsv": "120,80,150", "jersey_color_threshold": 30 }`

- **Conversió RGB → HSV:** implementació pura JS/TS, sense dependències externes

### Backend (`sc-api-gateway`)

- **`GET /api/v1/labeling/frame?video_key=...&frame_number=N`** → retorna signed URL de MinIO per al frame `N` de `labeling-frames/{video_key}/` + camp `total_frames` perquè el frontend calculi els salts representatius (10%, 25%, 50%, 75%, 90%)
- **`POST /api/v1/labeling/start`** → accepta `video_key` + `jersey_own_color_hsv` opcional; publica a Redis `labeling_frames_to_infer` amb el color inclòs al payload

### Redis — payload actualitzat

```json
{
  "video_key": "abc123",
  "frame_id": "frame_000001",
  "minio_bucket": "labeling-frames",
  "minio_key": "abc123/frame_000001.jpg",
  "frame_number": 1,
  "timestamp_s": 0.0,
  "jersey_own_color_hsv": "120,80,150",
  "jersey_color_threshold": 30
}
```

`jersey_own_color_hsv` és opcional al payload. Si és `null` o absent, `sc-inference-worker` fa fallback a la variable d'entorn.

---

## Flux complet

```
LabelingPage
    ↓ GET /api/v1/labeling/frame?video_key=abc123
sc-api-gateway → MinIO signed URL → frontend carrega frame al canvas
    ↓ usuari clica sobre jugador del seu equip
Frontend llegeix píxel RGB → converteix a HSV → previsualitza color
    ↓ usuari confirma color + prem "Iniciar etiquetatge"
POST /api/v1/labeling/start { video_key, jersey_own_color_hsv: "120,80,150" }
    ↓
sc-api-gateway publica a Redis: labeling_frames_to_infer (un missatge per frame, color inclòs)
    ↓
sc-inference-worker processa cada frame usant jersey_own_color_hsv del payload
    ↓
Prediccions enviades a Label Studio via API
```

---

## Tickets a crear (quan s'implementi)

| Ticket | Descripció | Etiqueta |
|--------|------------|---------|
| PJM-XX | `[frontend] Implementar eyedropper de color de samarreta a LabelingPage` | `frontend` |
| PJM-XX | `[backend] Endpoint GET /labeling/frame + POST /labeling/start amb jersey_own_color_hsv` | `backend` |
| PJM-XX | `[ai] sc-inference-worker llegeix jersey_own_color_hsv del payload Redis (fallback env var)` | `ai` |

---

## Dependències

- Requereix que `sc-video-manager` hagi extret almenys el primer frame a `labeling-frames/{video_key}/`
- No depèn de MongoDB ni del pipeline de partits — és completament independent
