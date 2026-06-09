# PJM-77 — Classificació equip per color samarreta + millora visual bboxes

**Estat:** En Procés
**Data:** 2026-06-09
**Ticket:** PJM-77

---

## Diagnòstic del codi actual

| Fitxer | Estat |
|--------|-------|
| `sc-inference-worker/app/utils/jersey_classifier.py` | Existeix però usa **mediana HSV** (poc robust) i crop al top 40% (inclou cap/cabell) |
| `sc-inference-worker/app/workers/inference_worker.py` | Quan `JERSEY_OWN_COLOR_HSV` és buit marca **tots** com `player_own` |
| `sc-logic-aggregator/app/services/aggregator_service.py` | Línies de 2px, text sense fons → poc llegible en 4K |

---

## Fase 1 — Millorar `jersey_classifier.py`

**Problema actual:**
- Crop: top 40% del bbox → inclou cabell/cara (confon el color)
- Mètode: mediana H,S,V → no distingeix el color dominant d'un fons

**Nova implementació:**
```
1. Crop 15–60% vertical del bbox → zona de torç pura (evita cap i pantalons)
2. Convertir a HSV
3. Filtrar: S >= 60 AND V >= 40 (descartar fons, pell, negre, blanc)
4. Histograma del canal H (0–180) sobre els píxels filtrats
5. Pic del histograma = hue dominant de la samarreta
6. Comparació circular H-only: |h_det - h_own| <= threshold → player_own
```

**Per què H-only vs. distància HSV completa:**
S i V canvien molt amb la llum (ombres, sol directe). El hue és estable:
el blau continua sent blau en ombra.

---

## Fase 2 — Petita correcció a `inference_worker.py`

- Passar imatge en BGR directament al classifier (eliminar conversió RGB innecessària)
- Quan `JERSEY_OWN_COLOR_HSV` buit → `class_name: "person"` (cian, no classificat) en lloc de forçar `player_own`

---

## Fase 3 — Millora visual a `aggregator_service.py`

### Gruix i font adaptatius a la resolució del vídeo
```python
line_thick = max(2, vid_w // 960)    # 4px en 4K, 2px en 1080p
font_scale = max(0.5, vid_w / 2000)  # ~1.9 en 4K, ~1.0 en 1080p
```

### Label amb fons ple
```
┌─────────────┐
│ Own  0.92   │  <- rectangle ple del color de l'equip + text blanc
└─────────────┘
│             │  <- bbox del jugador (4px)
│             │
└─────────────┘
```

---

## Fitxers que canvien

| Fitxer | Canvi |
|--------|-------|
| `services/sc-inference-worker/app/utils/jersey_classifier.py` | Reescriure algorisme (H histogram + torso crop) |
| `services/sc-inference-worker/app/workers/inference_worker.py` | BGR directe, fallback "person" |
| `services/sc-logic-aggregator/app/services/aggregator_service.py` | Gruix + font adaptatius, label amb fons ple |

---

## Referència de hues (OpenCV, 0–180)

| Color samarreta | JERSEY_OWN_COLOR_HSV |
|-----------------|----------------------|
| Vermell         | 0 (o 170)            |
| Taronja         | 10                   |
| Groc            | 25                   |
| Verd            | 60                   |
| Blau cel        | 100                  |
| Blau fosc       | 115                  |
| Lila            | 130                  |

`JERSEY_COLOR_THRESHOLD=25` (±25° de hue) funciona bé per a la majoria
de condicions d'il·luminació exterior.
