"""
Client de l'API de Label Studio per a pre-anotació.
"""
import logging

import requests

from app.utils.bbox_converter import yolo_to_label_studio

logger = logging.getLogger(__name__)

_MODEL_VERSION = "yolov8n-base-v0"


class LabelStudioService:
    def __init__(self, base_url: str, api_token: str, project_id: int):
        self._base_url = base_url.rstrip("/")
        self._project_id = project_id
        self._headers = {
            "Authorization": f"Token {api_token}",
            "Content-Type": "application/json",
        }

    def get_task_by_frame(self, session_id: str, frame_name: str) -> int | None:
        """
        Cerca el task_id de Label Studio corresponent a un frame.

        Filtra per project_id i per frame_name dins la URL de la imatge.
        Retorna None si el task encara no existeix (sync en curs).
        """
        url = f"{self._base_url}/api/tasks/"
        params = {
            "project": self._project_id,
            "data__image__contains": frame_name,
        }
        try:
            resp = requests.get(url, headers=self._headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            tasks = data.get("tasks", data) if isinstance(data, dict) else data
            # Filtra per session_id dins la URL de la imatge
            for task in tasks:
                image_url = task.get("data", {}).get("image", "")
                if session_id in image_url:
                    return int(task["id"])
            return None
        except requests.RequestException as exc:
            logger.warning(
                '{"event":"ls_get_task_error","frame":"%s","error":"%s"}',
                frame_name, str(exc)
            )
            return None

    def post_prediction(self, task_id: int, detections: list[dict]) -> bool:
        """
        Publica les prediccions de YOLOv8n a Label Studio per a un task.

        Args:
            task_id:    ID del task a Label Studio
            detections: llista de dicts amb x1,y1,x2,y2 normalitzats i 'label'

        Returns:
            True si s'ha publicat correctament, False si error
        """
        if not detections:
            return True

        results = []
        for det in detections:
            coords = yolo_to_label_studio(det["x1"], det["y1"], det["x2"], det["y2"])
            results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": coords["x"],
                    "y": coords["y"],
                    "width": coords["width"],
                    "height": coords["height"],
                    "rotation": 0,
                    "rectanglelabels": [det.get("label", "player_own")],
                },
            })

        avg_score = sum(d["confidence"] for d in detections) / len(detections)
        payload = {
            "task": task_id,
            "model_version": _MODEL_VERSION,
            "score": round(avg_score, 4),
            "result": results,
        }

        url = f"{self._base_url}/api/predictions/"
        try:
            resp = requests.post(url, headers=self._headers, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(
                '{"event":"ls_prediction_posted","task_id":%d,"detections":%d}',
                task_id, len(detections)
            )
            return True
        except requests.RequestException as exc:
            logger.error(
                '{"event":"ls_prediction_error","task_id":%d,"error":"%s"}',
                task_id, str(exc)
            )
            return False
