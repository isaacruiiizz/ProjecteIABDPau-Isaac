import asyncio
import json
import logging
from datetime import datetime, timezone

import boto3
from bson import ObjectId

from app.config import settings
from app.repositories import match_repository

logger = logging.getLogger(__name__)

BUCKET_RAW    = "raw-videos"
BUCKET_OUTPUT = "processed-videos"
QUEUE_VIDEO   = "video_to_process"


async def upload_match(
    file_bytes: bytes,
    title: str,
    user_id: str,
    s3,
    db,
) -> dict:
    match_id = str(ObjectId())
    minio_key = f"{match_id}/original.mp4"

    await asyncio.to_thread(
        s3.put_object,
        Bucket=BUCKET_RAW,
        Key=minio_key,
        Body=file_bytes,
        ContentType="video/mp4",
    )
    logger.info("Vídeo pujat a MinIO: %s/%s", BUCKET_RAW, minio_key)

    now = datetime.now(timezone.utc)
    await match_repository.create_match(db, {
        "_id":          ObjectId(match_id),
        "user_id":      user_id,
        "title":        title,
        "date":         now,
        "status":       "pending",
        "video_raw":    minio_key,
        "video_output": None,
        "fps":          None,
        "start_frame":  None,
        "end_frame":    None,
        "roi_polygon":  [],
        "created_at":   now,
        "updated_at":   now,
    })
    logger.info("Partit creat a MongoDB: %s", match_id)

    return {"match_id": match_id, "status": "pending"}


async def list_matches(user_id: str, db) -> list[dict]:
    docs = await match_repository.list_matches_by_user(db, user_id)
    return [
        {
            "match_id":      str(d["_id"]),
            "title":         d["title"],
            "status":        d["status"],
            "created_at":    d["created_at"],
            "start_seconds": d.get("start_seconds"),
            "end_seconds":   d.get("end_seconds"),
            "has_roi":       len(d.get("roi_polygon") or []) > 0,
        }
        for d in docs
    ]


async def process_match(match_id: str, redis, db) -> dict:
    doc = await match_repository.get_match_by_id(db, match_id)
    if doc is None:
        raise ValueError("Partit no trobat")
    if doc["status"] == "done":
        raise RuntimeError("El partit ja està completat")
    if not doc.get("video_raw"):
        raise RuntimeError("El partit no té cap vídeo pujat")

    if doc["status"] == "processing":
        for key in [
            f"frames:{match_id}:meta",
            f"frames:{match_id}:results",
            f"frames:{match_id}:total",
            f"frames:{match_id}:rendering",
        ]:
            await redis.delete(key)
        logger.info("process_match: claus Redis netejades per reprocessat match_id=%s", match_id)

    await match_repository.update_match_status(db, match_id, "processing")

    payload = json.dumps({
        "job_type":      "process_match",
        "match_id":      match_id,
        "minio_bucket":  BUCKET_RAW,
        "minio_key":     doc["video_raw"],
        "roi_polygon":   doc.get("roi_polygon") or [],
        "start_seconds": doc.get("start_seconds") or 0.0,
        "end_seconds":   doc.get("end_seconds"),
    })
    await redis.rpush(QUEUE_VIDEO, payload)
    logger.info("process_match: missatge publicat match_id=%s", match_id)

    return {"match_id": match_id, "status": "processing"}


async def delete_match(match_id: str, user_id: str, db, s3) -> None:
    doc = await match_repository.get_match_by_id(db, match_id)
    if doc is None or str(doc.get("user_id")) != user_id:
        raise ValueError("Partit no trobat")

    deleted = await match_repository.delete_match(db, match_id, user_id)
    if not deleted:
        raise ValueError("Partit no trobat")

    for bucket, key in [
        (BUCKET_RAW,    doc.get("video_raw")),
        (BUCKET_OUTPUT, doc.get("output_video")),
    ]:
        if key:
            try:
                await asyncio.to_thread(s3.delete_object, Bucket=bucket, Key=key)
            except Exception:
                pass


async def get_match_detail(match_id: str, user_id: str, db) -> dict:
    doc = await match_repository.get_match_by_id(db, match_id)
    if doc is None or str(doc.get("user_id")) != user_id:
        raise ValueError("Partit no trobat")

    download_url = None
    if doc["status"] == "done" and doc.get("output_video"):
        public_base = (settings.MINIO_PUBLIC_URL or
                       f"http{'s' if settings.MINIO_USE_SSL else ''}://{settings.MINIO_ENDPOINT}")
        presign_client = boto3.client(
            "s3",
            endpoint_url=public_base,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
        )
        download_url = await asyncio.to_thread(
            presign_client.generate_presigned_url,
            "get_object",
            Params={"Bucket": BUCKET_OUTPUT, "Key": doc["output_video"]},
            ExpiresIn=300,
        )

    return {
        "match_id":      str(doc["_id"]),
        "title":         doc["title"],
        "status":        doc["status"],
        "created_at":    doc["created_at"],
        "start_seconds": doc.get("start_seconds"),
        "end_seconds":   doc.get("end_seconds"),
        "download_url":  download_url,
    }


async def update_config(
    match_id: str,
    roi_polygon: list,
    start_seconds: float,
    end_seconds: float,
    db,
) -> dict:
    doc = await match_repository.update_match_config(db, match_id, {
        "roi_polygon":   [{"x": p.x, "y": p.y} for p in roi_polygon],
        "start_seconds": start_seconds,
        "end_seconds":   end_seconds,
    })
    if doc is None:
        raise ValueError("Partit no trobat")
    return {
        "match_id":      str(doc["_id"]),
        "roi_polygon":   doc["roi_polygon"],
        "start_seconds": doc["start_seconds"],
        "end_seconds":   doc["end_seconds"],
    }
