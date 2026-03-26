"""
Classifica cada detecció com 'player_own' o 'others' analitzant el color
dominant de la franja superior del bounding box (samarreta).

Si JERSEY_OWN_COLOR_HSV no està configurat, totes les deteccions reben
'player_own' com a fallback.
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fracció superior del BB que s'analitza (evita shorts/cames/gespa)
_JERSEY_CROP_RATIO = 0.40


def _parse_hsv_config(hsv_str: str) -> tuple[int, int, int] | None:
    """Parseja '120,80,150' → (120, 80, 150). Retorna None si buit o invàlid."""
    if not hsv_str or not hsv_str.strip():
        return None
    try:
        parts = [int(v.strip()) for v in hsv_str.split(",")]
        if len(parts) != 3:
            return None
        return tuple(parts)  # type: ignore[return-value]
    except ValueError:
        logger.warning('{"event":"jersey_classifier_bad_config","value":"%s"}', hsv_str)
        return None


def _dominant_hsv(crop: np.ndarray) -> tuple[float, float, float]:
    """Retorna el color dominant d'un crop en espai HSV (mitjana ponderada)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    # Màscara per ignorar píxels molt foscos (ombres/gespa)
    mask = hsv[:, :, 2] > 30
    if mask.sum() == 0:
        return (0.0, 0.0, 0.0)
    h = float(np.median(hsv[:, :, 0][mask]))
    s = float(np.median(hsv[:, :, 1][mask]))
    v = float(np.median(hsv[:, :, 2][mask]))
    return (h, s, v)


def _hsv_distance(a: tuple, b: tuple) -> float:
    """Distància euclidiana en espai HSV (pes doble a la component H)."""
    dh = min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0]))  # H és circular [0..180]
    ds = abs(a[1] - b[1])
    dv = abs(a[2] - b[2])
    return float(np.sqrt((dh * 2) ** 2 + ds ** 2 + dv ** 2))


def classify(
    image: np.ndarray,
    detections: list[dict],
    own_color_hsv_str: str,
    threshold: int = 30,
) -> list[dict]:
    """
    Assigna 'player_own' o 'others' a cada detecció basant-se en el color
    de la samarreta (franja superior del BB).

    Args:
        image:            imatge RGB en format numpy (H, W, 3)
        detections:       llista de dicts amb x1, y1, x2, y2 normalitzats [0..1]
        own_color_hsv_str: string "H,S,V" del color propi. Buit → tot 'player_own'
        threshold:        distància màxima HSV per considerar 'player_own'

    Returns:
        Mateixa llista amb camp 'label' afegit ('player_own' | 'others')
    """
    own_color = _parse_hsv_config(own_color_hsv_str)

    if own_color is None:
        for det in detections:
            det["label"] = "player_own"
        return detections

    h_img, w_img = image.shape[:2]

    for det in detections:
        x1 = int(det["x1"] * w_img)
        y1 = int(det["y1"] * h_img)
        x2 = int(det["x2"] * w_img)
        y2 = int(det["y2"] * h_img)

        # Retalla el 40% superior del BB (zona de la samarreta)
        jersey_y2 = y1 + max(1, int((y2 - y1) * _JERSEY_CROP_RATIO))
        crop = image[y1:jersey_y2, x1:x2]

        if crop.size == 0:
            det["label"] = "player_own"
            continue

        dominant = _dominant_hsv(crop)
        distance = _hsv_distance(dominant, own_color)
        det["label"] = "player_own" if distance <= threshold else "others"

    return detections
