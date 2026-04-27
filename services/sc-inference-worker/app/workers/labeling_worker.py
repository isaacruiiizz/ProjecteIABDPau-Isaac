"""
Worker de pre-anotació per al pipeline d'etiquetatge.

Flux:
  1. BLPOP labeling_frames_to_infer
  2. Descarrega frame de MinIO (labeling-frames)
  3. Inferència RF-DETR Small → deteccions de persones
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
from minio.error import S3Error

from app.config import settings
from app.services.label_studio_service import LabelStudioService
from app.services.rfdetr_service import RFDETRService
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


def _load_model_path(minio_client: Minio) -> str | None:
    """
    Intenta descarregar els pesos RF-DETR des de MinIO.
    Si el fitxer no existeix retorna None perquè RFDETRService
    faci l'auto-descàrrega des de HuggingFace.
    """
    local_path = "/tmp/rf-detr-small.pth"
    try:
        logger.info('{"event":"rfdetr_model_download_start","bucket":"%s","key":"%s"}',
                    settings.MINIO_BUCKET_MODELS, settings.MINIO_MODEL_KEY)
        minio_client.fget_object(
            settings.MINIO_BUCKET_MODELS,
            settings.MINIO_MODEL_KEY,
            local_path,
        )
        logger.info('{"event":"rfdetr_model_download_done","path":"%s"}', local_path)
        return local_path
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            logger.info('{"event":"rfdetr_model_not_in_minio","fallback":"huggingface_autodownload"}')
            return None
        raise


def run(stop_event: threading.Event) -> None:
    """
    Bucle principal del labeling worker. S'executa en un thread separat.
    Para quan stop_event és set.
    """
    logger.info('{"event":"labeling_worker_start"}')

    redis_client = _build_redis_client()
    minio_client = _build_minio_client()

    model_path = _load_model_path(minio_client)
    rfdetr_service = RFDETRService(
        model_path=model_path,
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

        # 2. Inferència RF-DETR Small
        detections = rfdetr_service.predict(image_bytes)
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

        img_array = rfdetr_service.get_image_array(image_bytes)
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
