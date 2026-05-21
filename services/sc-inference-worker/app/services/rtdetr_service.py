import logging

import cv2
import numpy as np
from ultralytics import RTDETR

logger = logging.getLogger(__name__)

_SHARPEN_KERNEL = np.array([
    [ 0, -0.3,  0],
    [-0.3,  2.2, -0.3],
    [ 0, -0.3,  0],
], dtype=np.float32)

# En el model base COCO, 'person' és la classe 0
_COCO_PERSON_CLASS = 0


class RTDETRService:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        device: str = "cpu",
        clahe: bool = True,
        sharpen: bool = True,
    ) -> None:
        self._confidence = confidence
        self._device = device
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        self._sharpen = sharpen

        logger.info('{"event":"rtdetr_loading","model":"%s","device":"%s"}', model_path, device)
        self._model = RTDETR(model_path)

        # Model base COCO: filtrar només la classe 'person' (0)
        # Model entrenat: totes les classes
        self._is_base_model = "person" in self._model.names.values()
        logger.info('{"event":"rtdetr_loaded","base_model":%s}', str(self._is_base_model).lower())

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if self._clahe is not None:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        if self._sharpen:
            bgr = cv2.filter2D(bgr, -1, _SHARPEN_KERNEL)
        return bgr

    def predict(self, image_bytes: bytes) -> list[dict]:
        arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            logger.warning('{"event":"rtdetr_decode_failed"}')
            return []

        arr = self._preprocess(arr)

        # Model base: filtra persona (classe 0 COCO); model entrenat: tot
        classes_filter = [_COCO_PERSON_CLASS] if self._is_base_model else None

        results = self._model.predict(
            arr,
            conf=self._confidence,
            imgsz=640 if self._is_base_model else 480,
            iou=0.45,
            classes=classes_filter,
            device=self._device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxyn = box.xyxyn[0].tolist()
                detections.append({
                    "x1": xyxyn[0],
                    "y1": xyxyn[1],
                    "x2": xyxyn[2],
                    "y2": xyxyn[3],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": "person" if self._is_base_model else r.names[int(box.cls[0])],
                })

        logger.debug('{"event":"rtdetr_predict_done","detections":%d}', len(detections))
        return detections

    def get_image_array(self, image_bytes: bytes) -> np.ndarray:
        arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        arr = self._preprocess(arr)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
