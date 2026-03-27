"""
Worker de pre-anotació per al pipeline d'etiquetatge.

Flux:
  1. BLPOP labeling_frames_to_infer
  2. Descarrega frame de MinIO (labeling-frames)
  3. Inferència YOLOv8n → deteccions de persones
  4. Classifica player_own / others per color de samarreta
  5. Cerca task_id a Label Studio (retry si sync no ha acabat)
  6. Publica predicció via Label Studio API

Màxim MAX_RETRIES reencuaments per frame. Si el task no apareix, es descarta.
"""
import json
import logging
import threading
import time

import redis
from minio import Minio

from app.config import settings
from app.services.label_studio_service import LabelStudioService
from app.services.yolo_service import YoloService
from app.utils import jersey_classifier

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_S = 2


def _build_redis_client() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD or None,
        decode_responses=False,
    )


def _build_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_USE_SSL,
        region="us-east-1",  # evita la crida GetBucketLocation (no inclosa a la política IAM)
    )


def run(stop_event: threading.Event) -> None:
    """
    Bucle principal del labeling worker. S'executa en un thread separat.
    Para quan stop_event és set.
    """
    logger.info('{"event":"labeling_worker_start"}')

    redis_client = _build_redis_client()
    minio_client = _build_minio_client()

    # Descarrega el model de MinIO a /tmp per evitar dependència de xarxa externa
    model_local_path = "/tmp/yolov8n.pt"
    logger.info('{"event":"yolo_model_download_start","bucket":"%s","key":"%s"}',
                settings.MINIO_BUCKET_MODELS, settings.MINIO_MODEL_KEY)
    minio_client.fget_object(
        settings.MINIO_BUCKET_MODELS,
        settings.MINIO_MODEL_KEY,
        model_local_path,
    )
    logger.info('{"event":"yolo_model_download_done","path":"%s"}', model_local_path)

    yolo_service = YoloService(
        model_path=model_local_path,
        confidence=settings.INFERENCE_LABELING_CONFIDENCE,
    )
    ls_service = LabelStudioService(
        base_url=settings.LABEL_STUDIO_URL,
        api_token=settings.LABEL_STUDIO_API_TOKEN,
        project_id=settings.LABEL_STUDIO_PROJECT_ID,
    )

    while not stop_event.is_set():
        raw = redis_client.blpop(settings.REDIS_QUEUE_LABELING, timeout=5)
        if raw is None:
            continue

        try:
            payload = json.loads(raw[1])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error('{"event":"labeling_worker_bad_payload","error":"%s"}', str(exc))
            continue

        session_id: str = payload.get("session_id", "")
        minio_key: str = payload.get("minio_key", "")
        frame_name: str = payload.get("frame_name", "")
        retry_count: int = payload.get("_retry", 0)

        logger.info(
            '{"event":"labeling_frame_received","session_id":"%s","frame":"%s","retry":%d}',
            session_id, frame_name, retry_count
        )

        # 1. Descarrega el frame de MinIO
        try:
            response = minio_client.get_object(
                settings.MINIO_BUCKET_LABELING_FRAMES, minio_key
            )
            image_bytes = response.read()
            response.close()
            response.release_conn()
        except Exception as exc:
            logger.error(
                '{"event":"labeling_minio_error","key":"%s","error":"%s"}',
                minio_key, str(exc)
            )
            continue

        # 2. Inferència YOLOv8n
        detections = yolo_service.predict(image_bytes)
        if not detections:
            logger.info(
                '{"event":"labeling_no_detections","frame":"%s"}', frame_name
            )
            continue

        # 3. Classifica player_own / others per color de samarreta
        # Prioritat: valor del payload (configurat per l'usuari via frontend)
        # Fallback: variable d'entorn JERSEY_OWN_COLOR_HSV
        jersey_color = payload.get("jersey_own_color_hsv") or settings.JERSEY_OWN_COLOR_HSV
        jersey_threshold = int(payload.get("jersey_color_threshold") or settings.JERSEY_COLOR_THRESHOLD)

        img_array = yolo_service.get_image_array(image_bytes)
        detections = jersey_classifier.classify(
            image=img_array,
            detections=detections,
            own_color_hsv_str=jersey_color,
            threshold=jersey_threshold,
        )

        # 4. Cerca task_id a Label Studio
        task_id = ls_service.get_task_by_frame(session_id, frame_name)

        if task_id is None:
            if retry_count < MAX_RETRIES:
                logger.info(
                    '{"event":"labeling_task_not_found_retry","frame":"%s","retry":%d}',
                    frame_name, retry_count + 1
                )
                time.sleep(RETRY_DELAY_S)
                payload["_retry"] = retry_count + 1
                redis_client.lpush(
                    settings.REDIS_QUEUE_LABELING, json.dumps(payload)
                )
            else:
                logger.warning(
                    '{"event":"labeling_task_not_found_discard","frame":"%s"}',
                    frame_name
                )
            continue

        # 5. Publica predicció a Label Studio
        ls_service.post_prediction(task_id, detections)

    logger.info('{"event":"labeling_worker_stop"}')
