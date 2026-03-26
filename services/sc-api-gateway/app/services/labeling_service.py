import asyncio
import json
import logging
from uuid import uuid4

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

BUCKET = "labeling-videos"
QUEUE = "video_to_process"


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
        Bucket=BUCKET,
        Key=minio_key,
        Body=file_bytes,
        ContentType="video/mp4",
    )
    logger.info("Vídeo pujat a MinIO: bucket=%s key=%s", BUCKET, minio_key)

    payload = json.dumps({
        "job_type": "process_labeling",
        "session_id": session_id,
        "minio_bucket": BUCKET,
        "minio_key": minio_key,
        "frame_interval": frame_interval,
    })
    await redis.rpush(QUEUE, payload)
    logger.info(
        "Missatge publicat a Redis: queue=%s session_id=%s", QUEUE, session_id
    )

    return {"session_id": session_id, "minio_key": minio_key}