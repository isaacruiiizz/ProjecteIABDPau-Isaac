"""
Client de l'API de Label Studio per a pre-anotació.
"""
import base64
import logging

import requests

from app.utils.bbox_converter import yolo_to_label_studio

logger = logging.getLogger(__name__)

_MODEL_VERSION = "rfdetr-small-base-v0"


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

        Label Studio guarda la imatge com a presign URL:
          /tasks/{N}/presign/?fileuri=BASE64(s3://labeling-frames/{session_id}/{frame_name})
        El filtre data__image__contains no funciona en LS 1.14.x, per la qual cosa
        paginam totes les tasques del projecte i filtrem client-side.
        Retorna None si el task encara no existeix (sync en curs).
        """
        s3_uri = f"s3://labeling-frames/{session_id}/{frame_name}"
        b64_target = base64.b64encode(s3_uri.encode()).decode()

        url = f"{self._base_url}/api/tasks/"
        page = 1
        page_size = 200
        try:
            while True:
                params = {"project": self._project_id, "page": page, "page_size": page_size}
                resp = requests.get(url, headers=self._headers, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                tasks = data.get("tasks", data) if isinstance(data, dict) else data
                if not tasks:
                    return None
                target_suffix = f"{session_id}/{frame_name}"
                for task in tasks:
                    image_val = task.get("data", {}).get("image", "")
                    # Format 1: /tasks/N/presign/?fileuri=BASE64(s3://labeling-frames/...)
                    if "fileuri=" in image_val:
                        b64 = image_val.split("fileuri=")[-1]
                        if b64 == b64_target:
                            return int(task["id"])
                    # Format 2: http://HOST/labeling-frames/SESSION/frame.jpg
                    elif image_val.endswith(target_suffix):
                        return int(task["id"])
                total = data.get("total", 0) if isinstance(data, dict) else 0
                if page * page_size >= total:
                    return None
                page += 1
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
