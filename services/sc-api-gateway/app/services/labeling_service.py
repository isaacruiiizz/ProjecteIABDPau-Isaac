import asyncio
import json
import logging
from uuid import uuid4

import redis.asyncio as aioredis

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
    Retorna la URL relativa del proxy de l'API gateway per al frame i el total de frames.

    La URL apunta a GET /api/v1/labeling/frame-img?video_key=...&frame_number=N
    perquè el frontend la carregui via apiClient (amb JWT), evitant problemes CORS
    en accedir directament a MinIO des del navegador.

    Returns:
        {"frame_url": "/api/v1/labeling/frame-img?video_key=...&frame_number=N", "total_frames": N}
    """
    frame_keys = await asyncio.to_thread(_list_frame_keys_sync, video_key, s3)
    total_frames = len(frame_keys)

    frame_url = (
        f"/api/v1/labeling/frame-img?video_key={video_key}&frame_number={frame_number}"
    )
    return {"frame_url": frame_url, "total_frames": total_frames}


def _get_frame_bytes_sync(video_key: str, frame_number: int, s3) -> bytes | None:
    """Descarrega els bytes d'un frame concret de MinIO."""
    keys = _list_frame_keys_sync(video_key, s3)
    if not keys:
        return None
    idx = max(0, min(frame_number - 1, len(keys) - 1))
    key = keys[idx]
    response = s3.get_object(Bucket=BUCKET_LABELING_FRAMES, Key=key)
    return response["Body"].read()


async def get_labeling_frame_bytes(video_key: str, frame_number: int, s3) -> bytes | None:
    """Retorna els bytes JPEG d'un frame per al proxy de l'API gateway."""
    return await asyncio.to_thread(_get_frame_bytes_sync, video_key, frame_number, s3)


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
