import json
import logging
import threading

import redis
from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.services.rtdetr_service import RTDETRService

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


def _download_model(minio_client: Minio) -> None:
    logger.info(
        '{"event":"rtdetr_model_download_start","bucket":"%s","key":"%s"}',
        settings.MINIO_BUCKET_MODELS, settings.RTDETR_MODEL_KEY,
    )
    minio_client.fget_object(
        settings.MINIO_BUCKET_MODELS,
        settings.RTDETR_MODEL_KEY,
        _MODEL_LOCAL_PATH,
    )
    logger.info('{"event":"rtdetr_model_download_done","path":"%s"}', _MODEL_LOCAL_PATH)


def _build_service() -> RTDETRService:
    return RTDETRService(
        model_path=_MODEL_LOCAL_PATH,
        confidence=settings.RTDETR_CONFIDENCE,
        device=settings.INFERENCE_DEVICE,
        clahe=settings.INFERENCE_CLAHE,
        sharpen=settings.INFERENCE_SHARPEN,
    )


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
        logger.error(
            '{"event":"inference_minio_error","key":"%s","error":"%s"}',
            minio_key, str(exc),
        )
        return

    detections = service.predict(image_bytes)

    result = {
        "match_id": match_id,
        "frame_id": frame_id,
        "frame_number": frame_number,
        "timestamp_s": timestamp_s,
        "detections": detections,
    }
    redis_client.rpush(settings.REDIS_QUEUE_RESULTS, json.dumps(result))

    logger.info(
        '{"event":"inference_frame_done","match_id":"%s","frame_number":%d,"detections":%d}',
        match_id, frame_number, len(detections),
    )


def run(stop_event: threading.Event) -> None:
    logger.info('{"event":"inference_worker_start"}')

    redis_client = _build_redis_client()
    minio_client = _build_minio_client()

    _download_model(minio_client)
    service = _build_service()

    queues = [settings.REDIS_QUEUE_FRAMES, settings.REDIS_QUEUE_MODEL_PROMOTED]

    while not stop_event.is_set():
        raw = redis_client.blpop(queues, timeout=5)
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
                _download_model(minio_client)
                service = _build_service()
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
