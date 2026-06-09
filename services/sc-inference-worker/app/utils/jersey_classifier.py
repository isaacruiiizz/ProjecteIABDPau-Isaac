"""
Classifica cada detecció com 'player_own' o 'player_other' analitzant el hue
dominant de la samarreta, amb background subtraction per bbox.

Algoritme:
  1. Mostreig dels 4 cantons del bbox → hue dominant del terra (background)
  2. Crop 20-55% vertical del bbox (zona del torç)
  3. Convertir a HSV; filtrar S < MIN_SAT o V < MIN_VAL
  4. Excloure píxels amb hue similar al terra (evita contaminació parquet/gespa)
  5. Histograma del canal H restant → pic = color dominant de la samarreta
  6. Distancia circular H-only vs. own_hue <= threshold → player_own
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TORSO_TOP    = 0.20
_TORSO_BOTTOM = 0.55
_MIN_SAT      = 80     # saturaciò mínima (el terra filtrat per BG subtract ja no contamina)
_MIN_VAL      = 50
_BG_CORNER_PX = 8      # px dels cantons del bbox per mostrejar el terra
_BG_HUE_TOL   = 12     # excloure píxels a menys de 12° del hue del terra


def _parse_own_hue(hsv_str: str) -> int | None:
    if not hsv_str or not hsv_str.strip():
        return None
    try:
        return int(hsv_str.strip().split(",")[0])
    except ValueError:
        logger.warning('{"event":"jersey_classifier_bad_config","value":"%s"}', hsv_str)
        return None


def _sample_bg_hue(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int | None:
    """
    Mostra els 4 cantons del bbox per estimar el color del terra.
    Retorna el hue dominant dels cantons, o None si no n'hi ha prou pixels.
    """
    h, w = image.shape[:2]
    p = _BG_CORNER_PX
    corners = [
        image[max(0, y1):min(h, y1+p),    max(0, x1):min(w, x1+p)],     # sup-esq
        image[max(0, y1):min(h, y1+p),    max(0, x2-p):min(w, x2)],     # sup-dre
        image[max(0, y2-p):min(h, y2),    max(0, x1):min(w, x1+p)],     # inf-esq
        image[max(0, y2-p):min(h, y2),    max(0, x2-p):min(w, x2)],     # inf-dre
    ]
    patches = [c for c in corners if c.size > 0]
    if not patches:
        return None

    combined = np.concatenate([c.reshape(-1, 3) for c in patches], axis=0)
    combined_img = combined.reshape(1, -1, 3).astype(np.uint8)
    hsv   = cv2.cvtColor(combined_img, cv2.COLOR_BGR2HSV)[0]
    mask  = (hsv[:, 1] >= 30) & (hsv[:, 2] >= 50)
    if mask.sum() < 5:
        return None
    hist = cv2.calcHist([hsv[:, 0][mask]], [0], None, [180], [0, 180]).flatten()
    return int(np.argmax(hist))


def _dominant_hue(crop_bgr: np.ndarray, bg_hue: int | None) -> int | None:
    """
    Hue dominant del crop excloent els píxels similars al terra (bg_hue).
    """
    if crop_bgr.size == 0:
        return None

    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= _MIN_SAT) & (hsv[:, :, 2] >= _MIN_VAL)

    if bg_hue is not None:
        h_ch  = hsv[:, :, 0].astype(np.int16)
        diff  = np.abs(h_ch - bg_hue)
        circ  = np.minimum(diff, 180 - diff)
        mask  = mask & (circ > _BG_HUE_TOL)

    if mask.sum() < 8:
        return None

    hist = cv2.calcHist([hsv[:, :, 0]], [0], mask.astype(np.uint8), [180], [0, 180])
    return int(np.argmax(hist))


def _hue_distance(h_a: int, h_b: int) -> int:
    d = abs(h_a - h_b)
    return min(d, 180 - d)


def classify(
    image: np.ndarray,
    detections: list[dict],
    own_color_hsv_str: str,
    threshold: int = 15,
) -> list[dict]:
    """
    Assigna 'player_own' o 'player_other' a cada deteccio.

    Args:
        image:             imatge BGR (H, W, 3)
        detections:        dicts amb x1, y1, x2, y2 normalitzats [0..1]
        own_color_hsv_str: 'H' o 'H,S,V' del color propi. Buit -> tot 'player_own'
        threshold:         diferencia maxima de hue per considerar 'player_own'
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

        # 1. Estima el color del terra als cantons del bbox
        bg_hue = _sample_bg_hue(image, x1, y1, x2, y2)

        # 2. Crop de torç
        box_h    = max(1, y2 - y1)
        torso_y1 = y1 + int(box_h * _TORSO_TOP)
        torso_y2 = y1 + int(box_h * _TORSO_BOTTOM)
        crop     = image[torso_y1:torso_y2, x1:x2]

        # 3. Hue dominant sense el terra
        dom_hue = _dominant_hue(crop, bg_hue)

        if dom_hue is None:
            det["label"] = "player_other"
            continue

        det["label"] = (
            "player_own" if _hue_distance(dom_hue, own_hue) <= threshold
            else "player_other"
        )

    return detections
