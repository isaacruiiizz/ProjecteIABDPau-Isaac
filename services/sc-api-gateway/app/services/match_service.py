import asyncio
import json
import logging
from datetime import datetime, timezone

import boto3
import httpx
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
        "ai_stats":      doc.get("ai_stats"),
        "ai_report":     doc.get("ai_report"),
    }


async def refine_ai_report(match_id: str, user_id: str, user_context: str, db) -> str:
    doc = await match_repository.get_match_by_id(db, match_id)
    if doc is None or str(doc.get("user_id")) != user_id:
        raise ValueError("Partit no trobat")
    if doc["status"] != "done":
        raise RuntimeError("El partit encara no s'ha processat")

    stats   = doc.get("ai_stats") or {}
    initial = doc.get("ai_report") or "No disponible."

    def _fmt(key: str, default="—") -> str:
        v = stats.get(key)
        return str(v) if v is not None else default

    prompt = (
        "Ets un analista tàctic de futbol expert. Disposes de les dades objectives d'un vídeo "
        "processat per visió artificial i del context aportat per l'entrenador. "
        "Has de generar una anàlisi tàctica professional, detallada i visual en català.\n\n"
        "--- ESTADÍSTIQUES TÈCNIQUES (IA) ---\n"
        f"• Frames analitzats: {_fmt('total_frames')}\n"
        f"• Durada analitzada: {_fmt('duration_s')} s\n"
        f"• Jugadors detectats de mitjana/frame: {_fmt('avg_players_per_frame')}\n"
        f"• Jugadors propis de mitjana: {_fmt('avg_own_per_frame')} ({_fmt('pct_own')}%)\n"
        f"• Jugadors rivals de mitjana: {_fmt('avg_other_per_frame')}\n"
        f"• Confiança de detecció: {float(stats['avg_confidence']) * 100:.0f}%\n"
        f"• Pic de densitat: {_fmt('max_density_count')} jugadors al segon {_fmt('max_density_time_s')}\n\n"
        "--- ANÀLISI AUTOMÀTICA INICIAL ---\n"
        f"{initial}\n\n"
        "--- CONTEXT DE L'ENTRENADOR ---\n"
        f"{user_context}\n\n"
        "Ara escriu l'anàlisi tàctica detallada. Estructura-la amb seccions en negreta (**Títol:**) "
        "i llistes amb guions (- punt). Inclou obligatòriament:\n"
        "**Situació del joc:** (resum de la fase analitzada)\n"
        "**Anàlisi tàctica:** (formació, pressing, transicions, moviments detectats)\n"
        "**Moments clau:** (instants destacats basats en la densitat i el context)\n"
        "**Punts forts i febles:** (del propi equip i del rival)\n"
        "**Recomanacions:** (aspectes a treballar per a l'entrenador)\n\n"
        "Escriu directament l'anàlisi sense introducció, en 200-350 paraules."
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()

    text = resp.json().get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama no ha retornat cap resposta")
    return text


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
