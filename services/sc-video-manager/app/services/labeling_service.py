import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import sentry_sdk
from minio import Minio

logger = logging.getLogger(__name__)


def process_labeling(payload: dict, minio_client: Minio, bucket_labeling_frames: str) -> int:
    """
    Descarrega un vídeo de labeling-videos, extreu 1 frame cada frame_interval segons
    amb FFmpeg i puja els frames a labeling-frames.

    No escriu res a MongoDB. No publica res a Redis.
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
