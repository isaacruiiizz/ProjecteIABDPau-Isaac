import base64
import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import redis
import requests
import sentry_sdk
from minio import Minio

logger = logging.getLogger(__name__)


def _publish_frame_to_redis(
    redis_client: redis.Redis,
    queue: str,
    session_id: str,
    minio_key: str,
    frame_name: str,
) -> None:
    msg = json.dumps({
        "session_id": session_id,
        "minio_key": minio_key,
        "frame_name": frame_name,
    })
    redis_client.lpush(queue, msg)


def _trigger_label_studio_sync(
    ls_url: str,
    api_token: str,
    storage_id: int,
) -> None:
    url = f"{ls_url.rstrip('/')}/api/storages/s3/{storage_id}/sync"
    headers = {"Authorization": f"Token {api_token}"}
    try:
        resp = requests.post(url, headers=headers, timeout=60)
        resp.raise_for_status()
        logger.info('{"event":"ls_sync_triggered","storage_id":%d}', storage_id)
    except requests.RequestException as exc:
        logger.warning(
            '{"event":"ls_sync_error","storage_id":%d,"error":"%s"}',
            storage_id, str(exc)
        )


def _patch_session_task_urls(
    ls_url: str,
    api_token: str,
    session_id: str,
    minio_public_url: str,
    project_id: int = 1,
    max_wait_s: int = 120,
) -> None:
    """
    Actualitza les URLs de les tasques de Label Studio d'una sessió per apuntar
    al MinIO públic (accessible des del navegador) en lloc de l'endpoint intern.

    Label Studio genera `/tasks/N/presign/?fileuri=BASE64(s3://...)` amb l'endpoint
    intern de MinIO (sc-object-storage:9000), que el navegador no pot resoldre.
    Convertim-les a `{minio_public_url}/labeling-frames/{session_id}/{frame}`.
    """
    ls_url = ls_url.rstrip("/")
    headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
    target_prefix = f"s3://labeling-frames/{session_id}/"

    # Espera que el sync hagi creat les tasques (fins a max_wait_s)
    deadline = time.time() + max_wait_s
    tasks = []
    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{ls_url}/api/tasks/",
                headers=headers,
                params={"project": project_id, "page_size": 500},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            all_tasks = data.get("tasks", data) if isinstance(data, dict) else data
            tasks = [
                t for t in all_tasks
                if "fileuri=" in t.get("data", {}).get("image", "")
            ]
            if tasks:
                break
        except requests.RequestException:
            pass
        time.sleep(3)

    patched = 0
    for task in tasks:
        img = task["data"].get("image", "")
        if "fileuri=" not in img:
            continue
        b64 = img.split("fileuri=")[-1]
        try:
            s3_uri = base64.b64decode(b64 + "==").decode("utf-8", errors="ignore").rstrip("\x00")
            if not s3_uri.startswith(target_prefix):
                continue
            key = s3_uri.replace("s3://labeling-frames/", "")
            new_url = f"{minio_public_url.rstrip('/')}/labeling-frames/{key}"
            resp = requests.patch(
                f"{ls_url}/api/tasks/{task['id']}/",
                headers=headers,
                json={"data": {"image": new_url}},
                timeout=10,
            )
            resp.raise_for_status()
            patched += 1
        except Exception as exc:
            logger.warning('{"event":"ls_patch_task_error","task_id":%d,"error":"%s"}',
                           task["id"], str(exc))

    logger.info('{"event":"ls_task_urls_patched","session_id":"%s","patched":%d}',
                session_id, patched)


def process_labeling(
    payload: dict,
    minio_client: Minio,
    bucket_labeling_frames: str,
    redis_client: redis.Redis | None = None,
    redis_queue_labeling: str = "labeling_frames_to_infer",
    label_studio_url: str = "",
    label_studio_api_token: str = "",
    label_studio_source_storage_id: int = 0,
    minio_public_url: str = "",
) -> int:
    """
    Descarrega un vídeo de labeling-videos, extreu 1 frame cada frame_interval segons
    amb FFmpeg i puja els frames a labeling-frames.

    Per cada frame pujat, publica un missatge a labeling_frames_to_infer (Redis)
    perquè sc-inference-worker el pre-anoti amb YOLOv8n.

    Un cop tots els frames pujats, dispara la sync del Source Storage de Label Studio.

    No escriu res a MongoDB.
    Retorna el nombre de frames pujats.
    """
    session_id: str = payload["session_id"]
    minio_bucket: str = payload["minio_bucket"]
    minio_key: str = payload["minio_key"]
    frame_interval: int = int(payload["frame_interval"])

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"sc-vm-{session_id}-"))
    video_path = tmp_dir / "original.mp4"
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    try:
        # 1. Descarrega el vídeo de MinIO
        logger.info('{"event":"download_start","session_id":"%s","bucket":"%s","key":"%s"}',
                    session_id, minio_bucket, minio_key)
        minio_client.fget_object(minio_bucket, minio_key, str(video_path))
        logger.info('{"event":"download_done","session_id":"%s"}', session_id)

        # 2. Extreu frames amb FFmpeg: 1 frame cada frame_interval segons
        frame_pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps=1/{frame_interval}",
            "-q:v", "2",
            frame_pattern,
            "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

        # 3. Puja cada frame a labeling-frames
        # No es publica a Redis aquí — l'usuari selecciona el color de samarreta
        # al frontend i prem "Iniciar etiquetatge", que crida POST /api/v1/labeling/start
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        uploaded = 0
        for frame_file in frame_files:
            # frame_000001.jpg → {session_id}/frame_000001.jpg
            object_key = f"{session_id}/{frame_file.name}"
            minio_client.fput_object(
                bucket_labeling_frames,
                object_key,
                str(frame_file),
                content_type="image/jpeg",
            )
            uploaded += 1

        logger.info('{"event":"labeling_done","session_id":"%s","frames_uploaded":%d}',
                    session_id, uploaded)

        # 4. Dispara sync del Source Storage de Label Studio
        if label_studio_url and label_studio_api_token and label_studio_source_storage_id:
            _trigger_label_studio_sync(
                label_studio_url,
                label_studio_api_token,
                label_studio_source_storage_id,
            )

        # 5. Patch task URLs per a accés directe des del navegador
        if label_studio_url and label_studio_api_token and minio_public_url:
            _patch_session_task_urls(
                label_studio_url,
                label_studio_api_token,
                session_id,
                minio_public_url,
            )

        # 6. Elimina el vídeo original de labeling-videos (ja no cal un cop els frames estan pujats)
        try:
            minio_client.remove_object(minio_bucket, minio_key)
            logger.info('{"event":"source_video_deleted","session_id":"%s","bucket":"%s","key":"%s"}',
                        session_id, minio_bucket, minio_key)
        except Exception as exc:
            logger.warning('{"event":"source_video_delete_error","session_id":"%s","error":"%s"}',
                           session_id, str(exc))

        return uploaded

    except Exception as exc:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("session_id", session_id)
            scope.set_context("payload", payload)
            sentry_sdk.capture_exception(exc)
        logger.error('{"event":"labeling_error","session_id":"%s","error":"%s"}',
                     session_id, str(exc))
        raise

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
