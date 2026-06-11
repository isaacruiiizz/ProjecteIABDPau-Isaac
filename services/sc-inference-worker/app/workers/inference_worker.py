import json
import logging
import threading

import cv2
import numpy as np
import redis
from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.services.rtdetr_service import RTDETRService
from app.utils import jersey_classifier

logger = logging.getLogger(__name__)

_MODEL_LOCAL_PATH = "/tmp/rtdetr_best.pt"


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
        region="us-east-1",
    )


def _resolve_model(minio_client: Minio) -> str:
    """
    Si RTDETR_MODEL_KEY conté '/' és un path de MinIO → descarregar.
    Si és un nom tipus 'rtdetr-l.pt' → Ultralytics auto-descàrrega.
    """
    key = settings.RTDETR_MODEL_KEY
    if "/" in key:
        logger.info('{"event":"rtdetr_model_download_start","key":"%s"}', key)
        minio_client.fget_object(settings.MINIO_BUCKET_MODELS, key, _MODEL_LOCAL_PATH)
        logger.info('{"event":"rtdetr_model_download_done"}')
        return _MODEL_LOCAL_PATH
    logger.info('{"event":"rtdetr_model_ultralytics_auto","model":"%s"}', key)
    return key


def _build_service(model_path: str) -> RTDETRService:
    return RTDETRService(
        model_path=model_path,
        confidence=settings.RTDETR_CONFIDENCE,
        device=settings.INFERENCE_DEVICE,
        clahe=settings.INFERENCE_CLAHE,
        sharpen=settings.INFERENCE_SHARPEN,
    )


def _apply_jersey_classifier(image_bytes: bytes, detections: list[dict]) -> list[dict]:
    """Classifica player_own / player_other per color de samarreta (HSV dominant)."""
    if not detections:
        return detections

    if not settings.JERSEY_OWN_COLOR_HSV or not settings.JERSEY_OWN_COLOR_HSV.strip():
        for d in detections:
            d["class_name"] = "person"
        return detections

    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return detections

    classified = jersey_classifier.classify(
        image=arr,
        detections=detections,
        own_color_hsv_str=settings.JERSEY_OWN_COLOR_HSV,
        threshold=settings.JERSEY_COLOR_THRESHOLD,
    )
    for d in classified:
        d["class_name"] = d.pop("label")
    return classified


def _process_frame(
    payload: dict,
    minio_client: Minio,
    service: RTDETRService,
    redis_client: redis.Redis,
) -> None:
    match_id: str = payload["match_id"]
    frame_id: str = payload["frame_id"]
    minio_bucket: str = payload["minio_bucket"]
    minio_key: str = payload["minio_key"]
    frame_number: int = payload["frame_number"]
    timestamp_s: float = payload["timestamp_s"]

    try:
        response = minio_client.get_object(minio_bucket, minio_key)
        image_bytes = response.read()
        response.close()
        response.release_conn()
    except S3Error as exc:
        logger.error('{"event":"inference_minio_error","key":"%s","error":"%s"}', minio_key, str(exc))
        return

    detections = service.predict(image_bytes)
    detections = _apply_jersey_classifier(image_bytes, detections)

    result = {
        "match_id": match_id,
        "frame_id": frame_id,
        "frame_number": frame_number,
        "timestamp_s": timestamp_s,
        "detections": detections,
    }
    redis_client.rpush(settings.REDIS_QUEUE_RESULTS, json.dumps(result))

    own = sum(1 for d in detections if d["class_name"] == "player_own")
    logger.info(
        '{"event":"inference_frame_done","match_id":"%s","frame_number":%d,"total":%d,"player_own":%d}',
        match_id, frame_number, len(detections), own,
    )


def run(stop_event: threading.Event) -> None:
    logger.info('{"event":"inference_worker_start"}')

    redis_client = _build_redis_client()
    minio_client = _build_minio_client()

    model_path = _resolve_model(minio_client)
    service = _build_service(model_path)

    queues = [settings.REDIS_QUEUE_FRAMES, settings.REDIS_QUEUE_MODEL_PROMOTED]

    while not stop_event.is_set():
        try:
            raw = redis_client.blpop(queues, timeout=5)
        except (redis.exceptions.TimeoutError, redis.exceptions.ConnectionError) as exc:
            logger.warning('{"event":"inference_redis_reconnect","error":"%s"}', str(exc))
            try:
                redis_client = _build_redis_client()
            except Exception:
                pass
            continue
        if raw is None:
            continue

        queue, data = raw

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            logger.error('{"event":"inference_worker_bad_payload","error":"%s"}', str(exc))
            continue

        if queue == settings.REDIS_QUEUE_MODEL_PROMOTED.encode():
            logger.info('{"event":"inference_model_promoted_reload"}')
            try:
                model_path = _resolve_model(minio_client)
                service = _build_service(model_path)
            except Exception as exc:
                logger.error('{"event":"inference_model_reload_failed","error":"%s"}', str(exc))
            continue

        try:
            _process_frame(payload, minio_client, service, redis_client)
        except Exception as exc:
            logger.error(
                '{"event":"inference_frame_error","frame_id":"%s","error":"%s"}',
                payload.get("frame_id", "?"), str(exc),
            )

    logger.info('{"event":"inference_worker_stop"}')
