import asyncio
import json
import logging
from urllib.parse import urlparse
from uuid import uuid4

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

BUCKET_LABELING_VIDEOS = "labeling-videos"
BUCKET_LABELING_FRAMES = "labeling-frames"
QUEUE_VIDEO = "video_to_process"
QUEUE_LABELING_INFER = "labeling_frames_to_infer"
FRAME_PRESIGNED_EXPIRY_S = 3600


def _list_frame_keys_sync(video_key: str, s3) -> list[str]:
    """Llista totes les claus de frames d'una sessió (paginació per >1000 frames)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket=BUCKET_LABELING_FRAMES,
        Prefix=f"{video_key}/",
    )
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".jpg"):
                keys.append(obj["Key"])
    return sorted(keys)


def _presigned_url_sync(key: str, s3) -> str:
    """Genera una URL pre-signada per a un frame. Reescriu l'host si MINIO_PUBLIC_URL és configurat."""
    url: str = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_LABELING_FRAMES, "Key": key},
        ExpiresIn=FRAME_PRESIGNED_EXPIRY_S,
    )
    if settings.MINIO_PUBLIC_URL:
        parsed = urlparse(url)
        internal = f"{parsed.scheme}://{parsed.netloc}"
        url = url.replace(internal, settings.MINIO_PUBLIC_URL.rstrip("/"), 1)
    return url


async def upload_labeling_video(
    file_bytes: bytes,
    frame_interval: int,
    s3,
    redis: aioredis.Redis,
) -> dict:
    session_id = str(uuid4())
    minio_key = f"{session_id}/original.mp4"

    await asyncio.to_thread(
        s3.put_object,
        Bucket=BUCKET_LABELING_VIDEOS,
        Key=minio_key,
        Body=file_bytes,
        ContentType="video/mp4",
    )
    logger.info("Vídeo pujat a MinIO: bucket=%s key=%s", BUCKET_LABELING_VIDEOS, minio_key)

    payload = json.dumps({
        "job_type": "process_labeling",
        "session_id": session_id,
        "minio_bucket": BUCKET_LABELING_VIDEOS,
        "minio_key": minio_key,
        "frame_interval": frame_interval,
    })
    await redis.rpush(QUEUE_VIDEO, payload)
    logger.info("Missatge publicat a Redis: queue=%s session_id=%s", QUEUE_VIDEO, session_id)

    return {"session_id": session_id, "minio_key": minio_key}


async def get_labeling_frame(video_key: str, frame_number: int, s3) -> dict:
    """
    Retorna la URL pre-signada d'un frame i el total de frames de la sessió.

    Args:
        video_key:    session_id del vídeo d'etiquetatge
        frame_number: índex 1-based del frame sol·licitat
        s3:           client boto3 S3

    Returns:
        {"frame_url": "<presigned_url>", "total_frames": N}
    """
    frame_keys = await asyncio.to_thread(_list_frame_keys_sync, video_key, s3)
    total_frames = len(frame_keys)

    if total_frames == 0:
        return {"frame_url": "", "total_frames": 0}

    # Clamp frame_number a [1, total_frames]
    idx = max(0, min(frame_number - 1, total_frames - 1))
    frame_key = frame_keys[idx]

    frame_url = await asyncio.to_thread(_presigned_url_sync, frame_key, s3)

    return {"frame_url": frame_url, "total_frames": total_frames}


async def start_labeling(
    video_key: str,
    jersey_own_color_hsv: str | None,
    jersey_color_threshold: int,
    s3,
    redis: aioredis.Redis,
) -> dict:
    """
    Llista els frames extrets i publica un missatge per frame a labeling_frames_to_infer.
    Inclou jersey_own_color_hsv i jersey_color_threshold al payload perquè
    sc-inference-worker els usi sense consultar cap altra font.

    Returns:
        {"status": "queued", "frames_queued": N}
    """
    frame_keys = await asyncio.to_thread(_list_frame_keys_sync, video_key, s3)

    if not frame_keys:
        logger.warning("start_labeling: cap frame trobat per video_key=%s", video_key)
        return {"status": "no_frames", "frames_queued": 0}

    for i, key in enumerate(frame_keys, start=1):
        frame_name = key.split("/")[-1]
        msg = json.dumps({
            "session_id": video_key,
            "minio_key": key,
            "frame_name": frame_name,
            "minio_bucket": BUCKET_LABELING_FRAMES,
            "frame_number": i,
            "jersey_own_color_hsv": jersey_own_color_hsv,
            "jersey_color_threshold": jersey_color_threshold,
        })
        await redis.rpush(QUEUE_LABELING_INFER, msg)

    logger.info(
        "start_labeling: %d frames encuats per video_key=%s color=%s",
        len(frame_keys), video_key, jersey_own_color_hsv,
    )
    return {"status": "queued", "frames_queued": len(frame_keys)}
