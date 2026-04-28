"""
Servei de càrrega i inferència RF-DETR Small.

Model base: RF-DETR Small pre-entrenat amb COCO (80 classes).
Classe 0 = 'person' — la que usem per detectar jugadors.
Confiança baixa (0.3) per pre-anotació: millor tenir de més que de menys.
"""
import io
import logging
import os

import numpy as np
import rfdetr.detr as _rfdetr_detr
from PIL import Image
from rfdetr import RFDETRSmall
from rfdetr.assets.model_weights import download_pretrain_weights as _orig_download_pretrain_weights


def _download_if_missing(pretrain_weights: str, redownload: bool = False, validate_md5: bool = True) -> None:
    """Salta la descàrrega si el fitxer ja existeix localment (evita dependència de xarxa)."""
    if os.path.exists(pretrain_weights):
        return
    _orig_download_pretrain_weights(pretrain_weights, redownload=False, validate_md5=validate_md5)


# rfdetr.detr crida download_pretrain_weights amb redownload=True ignorant el fitxer local.
# Aquest patch fa que comprovi primer si existeix abans d'intentar descarregar.
_rfdetr_detr.download_pretrain_weights = _download_if_missing

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


class RFDETRService:
    def __init__(self, model_path: str | None = None, confidence: float = 0.3):
        self._confidence = confidence
        logger.info('{"event":"rfdetr_loading","model":"%s"}', model_path or "auto-download")
        if model_path:
            self._model = RFDETRSmall(pretrain_weights=model_path)
        else:
            self._model = RFDETRSmall()
        logger.info('{"event":"rfdetr_loaded"}')

    def predict(self, image_bytes: bytes) -> list[dict]:
        """
        Executa inferència sobre una imatge i retorna les deteccions de persones.

        Returns:
            Llista de dicts: [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
                               "confidence": 0.87, "class_id": 0}]
            Coordenades normalitzades [0..1]. Llista buida si cap detecció.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size

        detections = self._model.predict(image, threshold=self._confidence)

        results = []
        for i in range(len(detections.xyxy)):
            class_id = int(detections.class_id[i])
            if class_id != _PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = detections.xyxy[i]
            results.append({
                "x1": float(x1) / w,
                "y1": float(y1) / h,
                "x2": float(x2) / w,
                "y2": float(y2) / h,
                "confidence": float(detections.confidence[i]),
                "class_id": class_id,
            })

        logger.debug('{"event":"rfdetr_predict_done","detections":%d}', len(results))
        return results

    def get_image_array(self, image_bytes: bytes) -> np.ndarray:
        """Retorna la imatge com a numpy array RGB (per a jersey_classifier)."""
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
