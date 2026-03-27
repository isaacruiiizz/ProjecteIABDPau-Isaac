import asyncio
import json
import logging
from uuid import uuid4

import httpx
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


async def get_ls_stats(ls_url: str, api_token: str, project_id: int) -> dict:
    """
    Retorna el total de tasques, quantes estan anotades (humà) i quantes
    tenen prediccions (inference worker) al projecte de Label Studio.
    """
    headers = {"Authorization": f"Token {api_token}"}
    base = ls_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            project_resp, pred_resp = await asyncio.gather(
                client.get(f"{base}/api/projects/{project_id}/", headers=headers),
                client.get(
                    f"{base}/api/predictions/",
                    headers=headers,
                    params={"project": project_id},
                ),
            )
            project_resp.raise_for_status()
            project_data = project_resp.json()
            predicted = 0
            if pred_resp.is_success:
                pred_data = pred_resp.json()
                # LS 1.14 retorna llista plana (no dict paginat)
                if isinstance(pred_data, list):
                    predicted = len(pred_data)
                else:
                    predicted = pred_data.get("count", 0)
            return {
                "total_tasks": project_data.get("task_number", 0),
                "annotated_tasks": project_data.get("num_tasks_with_annotations", 0),
                "predicted_tasks": predicted,
            }
    except Exception as exc:
        logger.warning("get_ls_stats error: %s", str(exc))
        return {"total_tasks": 0, "annotated_tasks": 0, "predicted_tasks": 0}


def _delete_all_labeling_frames_sync(s3) -> int:
    """Esborra tots els frames del bucket labeling-frames. Retorna el nombre d'objectes eliminats."""
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_LABELING_FRAMES):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=BUCKET_LABELING_FRAMES, Delete={"Objects": objects})
            deleted += len(objects)
    return deleted


async def clear_ls_tasks(ls_url: str, api_token: str, project_id: int, s3) -> dict:
    """
    Esborra totes les tasques del projecte Label Studio i tots els frames
    del bucket labeling-frames de MinIO, deixant el sistema net per a una
    nova sessió d'etiquetatge.
    """
    headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
    url = f"{ls_url.rstrip('/')}/api/dm/actions?id=delete_tasks&project={project_id}"
    body = {
        "filters": {"conjunction": "and", "items": []},
        "selectedItems": {"all": True, "excluded": []},
        "project": project_id,
    }
    deleted_tasks = 0
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            deleted_tasks = resp.json().get("processed_items", 0)
    except Exception as exc:
        logger.warning("clear_ls_tasks LS error: %s", str(exc))

    deleted_frames = await asyncio.to_thread(_delete_all_labeling_frames_sync, s3)
    logger.info("clear_ls_tasks: deleted_tasks=%d deleted_frames=%d", deleted_tasks, deleted_frames)
    return {"deleted": deleted_tasks, "deleted_frames": deleted_frames}
