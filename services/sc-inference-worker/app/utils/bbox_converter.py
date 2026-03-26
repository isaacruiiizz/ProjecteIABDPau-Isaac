"""
Conversió de coordenades YOLO normalitzades a format Label Studio percentatge.

YOLO:          (x1, y1, x2, y2) normalitzat [0..1]
Label Studio:  (x%, y%, width%, height%) relatiu a la imatge [0..100]
"""


def yolo_to_label_studio(x1: float, y1: float, x2: float, y2: float) -> dict:
    """
    Converteix un bounding box de format YOLO normalitzat a format Label Studio.

    Args:
        x1, y1: cantonada superior esquerra normalitzada [0..1]
        x2, y2: cantonada inferior dreta normalitzada [0..1]

    Returns:
        Dict amb x, y, width, height en percentatge [0..100]
    """
    return {
        "x": x1 * 100,
        "y": y1 * 100,
        "width": (x2 - x1) * 100,
        "height": (y2 - y1) * 100,
    }
