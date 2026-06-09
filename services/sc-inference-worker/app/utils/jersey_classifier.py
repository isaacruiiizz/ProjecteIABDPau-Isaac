"""
Classifica cada detecció com 'player_own' o 'player_other' analitzant el hue
dominant de la zona del torç del bounding box (samarreta).

Algoritme:
  1. Crop 15–60% vertical del bbox (torç, evitant cap i pantalons)
  2. Convertir a HSV
  3. Filtrar píxels amb S < MIN_SAT o V < MIN_VAL (ignora fons/pell/blanc/negre)
  4. Histograma del canal H → pic = color dominant de la samarreta
  5. Distància circular H-only vs. own_hue <= threshold → player_own
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TORSO_TOP    = 0.20   # saltem cap i coll (20% superior del bbox)
_TORSO_BOTTOM = 0.50   # tallem a meitat del cos (evita terra i cames)
_MIN_SAT      = 90     # filtre alt: el terra/parquet té S<90, les samarretes solen tenir S>90
_MIN_VAL      = 50     # ignora zones molt fosques


def _parse_own_hue(hsv_str: str) -> int | None:
    """
    Accepta 'H' o 'H,S,V' -> retorna el hue (0-180).
    Retorna None si buit o invalid.
    """
    if not hsv_str or not hsv_str.strip():
        return None
    try:
        return int(hsv_str.strip().split(",")[0])
    except ValueError:
        logger.warning('{"event":"jersey_classifier_bad_config","value":"%s"}', hsv_str)
        return None


def _dominant_hue(crop_bgr: np.ndarray) -> int | None:
    """
    Retorna el hue dominant (0-180 OpenCV) del crop.
    Filtra pixels poc saturats i molt foscos.
    Retorna None si no hi ha prou pixels informatius.
    """
    if crop_bgr.size == 0:
        return None

    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= _MIN_SAT) & (hsv[:, :, 2] >= _MIN_VAL)

    if mask.sum() < 10:
        return None

    hist = cv2.calcHist([hsv[:, :, 0]], [0], mask.astype(np.uint8), [180], [0, 180])
    return int(np.argmax(hist))


def _hue_distance(h_a: int, h_b: int) -> int:
    """Distancia circular entre dos hues (domini 0-180)."""
    d = abs(h_a - h_b)
    return min(d, 180 - d)


def classify(
    image: np.ndarray,
    detections: list[dict],
    own_color_hsv_str: str,
    threshold: int = 25,
) -> list[dict]:
    """
    Assigna 'player_own' o 'player_other' a cada deteccio.

    Args:
        image:             imatge BGR (H, W, 3)
        detections:        dicts amb x1, y1, x2, y2 normalitzats [0..1]
        own_color_hsv_str: 'H' o 'H,S,V' del color propi. Buit -> tot 'player_own'
        threshold:         diferencia maxima de hue per considerar 'player_own'

    Returns:
        Mateixa llista amb camp 'label' afegit ('player_own' | 'player_other')
    """
    own_hue = _parse_own_hue(own_color_hsv_str)

    if own_hue is None:
        for det in detections:
            det["label"] = "player_own"
        return detections

    h_img, w_img = image.shape[:2]

    for det in detections:
        x1 = int(det["x1"] * w_img)
        y1 = int(det["y1"] * h_img)
        x2 = int(det["x2"] * w_img)
        y2 = int(det["y2"] * h_img)

        box_h    = max(1, y2 - y1)
        torso_y1 = y1 + int(box_h * _TORSO_TOP)
        torso_y2 = y1 + int(box_h * _TORSO_BOTTOM)
        crop     = image[torso_y1:torso_y2, x1:x2]

        dom_hue = _dominant_hue(crop)

        if dom_hue is None:
            det["label"] = "player_other"
            continue

        det["label"] = (
            "player_own" if _hue_distance(dom_hue, own_hue) <= threshold
            else "player_other"
        )

    return detections
