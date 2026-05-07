import asyncio
import logging
from datetime import datetime, timezone

from bson import ObjectId

from app.repositories import match_repository

logger = logging.getLogger(__name__)

BUCKET_RAW = "raw-videos"


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
