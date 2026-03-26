"""
Servei de càrrega i inferència YOLOv8n.

Model base: YOLOv8n pre-entrenat amb COCO (80 classes).
Classe 0 = 'person' — la que usem per detectar jugadors.
Confiança baixa (0.3) per pre-anotació: millor tenir de més que de menys.
"""
import io
import logging

import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_PERSON_CLASS_ID = 0


class YoloService:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3):
        """
        Carrega el model YOLO.

        Args:
            model_path: ruta local o nom de model Ultralytics
                        ('yolov8n.pt' → descarrega automàticament si no existeix)
            confidence: llindar mínim de confiança per filtrar deteccions
        """
        self._confidence = confidence
        logger.info('{"event":"yolo_loading","model":"%s"}', model_path)
        self._model = YOLO(model_path)
        logger.info('{"event":"yolo_loaded","model":"%s"}', model_path)

    def predict(self, image_bytes: bytes) -> list[dict]:
        """
        Executa inferència sobre una imatge i retorna les deteccions de persones.

        Args:
            image_bytes: bytes de la imatge (JPEG/PNG)

        Returns:
            Llista de dicts: [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
                               "confidence": 0.87, "class_id": 0}]
            Coordenades normalitzades [0..1]. Llista buida si cap detecció.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)

        results = self._model(img_array, conf=self._confidence, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id != _PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(box.conf[0]),
                    "class_id": class_id,
                })

        logger.debug(
            '{"event":"yolo_predict_done","detections":%d}', len(detections)
        )
        return detections

    def get_image_array(self, image_bytes: bytes) -> np.ndarray:
        """Retorna la imatge com a numpy array RGB (per a jersey_classifier)."""
        return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
