import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

FRAMES_PER_SEC = 2


def process_match(payload: dict, minio_client, redis_client, settings) -> None:
    match_id   = payload["match_id"]
    minio_key  = payload["minio_key"]
    start_s    = float(payload.get("start_seconds") or 0.0)
    end_s      = float(payload["end_seconds"])
    bucket_raw = payload.get("minio_bucket", settings.MINIO_BUCKET_RAW)

    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"sc-match-{match_id[:8]}-"))
    video_path = tmp_dir / "video.mp4"
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    try:
        logger.info('{"event":"download_start","match_id":"%s","key":"%s"}', match_id, minio_key)
        minio_client.fget_object(bucket_raw, minio_key, str(video_path))
        logger.info('{"event":"download_done","match_id":"%s"}', match_id)

        frame_pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg", "-ss", str(start_s), "-to", str(end_s),
            "-i", str(video_path),
            "-vf", f"fps={FRAMES_PER_SEC}", "-q:v", "2", frame_pattern, "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-500:]}")

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            raise RuntimeError("FFmpeg generated no frames")

        total_frames = len(frame_files)
        for frame_file in frame_files:
            object_key = f"{match_id}/{frame_file.name}"
            minio_client.fput_object(
                settings.MINIO_BUCKET_PENDING, object_key, str(frame_file),
                content_type="image/jpeg",
            )

        logger.info('{"event":"frames_uploaded","match_id":"%s","count":%d}', match_id, total_frames)

        redis_client.set(f"frames:{match_id}:total", str(total_frames))

        for frame_number, frame_file in enumerate(frame_files, start=1):
            timestamp_s = start_s + (frame_number - 1) / FRAMES_PER_SEC
            task_msg = json.dumps({
                "match_id":     match_id,
                "frame_id":     f"{match_id}_f{frame_number:06d}",
                "minio_bucket": settings.MINIO_BUCKET_PENDING,
                "minio_key":    f"{match_id}/{frame_file.name}",
                "frame_number": frame_number,
                "timestamp_s":  round(timestamp_s, 3),
            })
            redis_client.rpush(settings.REDIS_QUEUE_FRAMES, task_msg)

        logger.info('{"event":"tasks_published","match_id":"%s","count":%d}', match_id, total_frames)

    except Exception:
        logger.exception('{"event":"process_match_error","match_id":"%s"}', match_id)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
