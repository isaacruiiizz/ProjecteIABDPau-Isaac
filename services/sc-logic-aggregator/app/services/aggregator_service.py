import json
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
from bson import ObjectId

logger = logging.getLogger(__name__)

FRAMES_PER_SEC = 2

_COLOR_OWN   = (0, 200,   0)   # green
_COLOR_OTHER = (0,   0, 200)   # red
_COLOR_DEF   = (200, 200,  0)  # cyan


def handle_frame_result(payload: dict, redis_client, minio_client, db, settings) -> None:
    match_id     = payload["match_id"]
    frame_number = int(payload["frame_number"])
    detections   = payload.get("detections") or []

    redis_client.hset(f"frames:{match_id}:results", str(frame_number), json.dumps(detections))
    processed = redis_client.hincrby(f"frames:{match_id}:meta", "processed", 1)
    total_str = redis_client.get(f"frames:{match_id}:total")

    if total_str is None:
        logger.warning('{"event":"no_total_key","match_id":"%s","frame":%d}', match_id, frame_number)
        return

    total = int(total_str)
    logger.info(
        '{"event":"frame_result","match_id":"%s","frame":%d,"processed":%d,"total":%d}',
        match_id, frame_number, processed, total,
    )

    if processed >= total:
        # Guard: only render once even if extra messages arrive
        locked = redis_client.set(f"frames:{match_id}:rendering", "1", nx=True, ex=3600)
        if locked:
            _render_and_finalize(match_id, redis_client, minio_client, db, settings)


def _render_and_finalize(match_id: str, redis_client, minio_client, db, settings) -> None:
    logger.info('{"event":"render_start","match_id":"%s"}', match_id)

    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"sc-render-{match_id[:8]}-"))
    output_mp4 = tmp_dir / "output.mp4"
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    try:
        all_results = redis_client.hgetall(f"frames:{match_id}:results")
        if not all_results:
            raise RuntimeError("No frame results in Redis")

        sorted_frames = sorted(all_results.items(), key=lambda kv: int(kv[0]))
        annotated_paths: list[Path] = []

        for frame_number_str, dets_json in sorted_frames:
            frame_number = int(frame_number_str)
            frame_name   = f"frame_{frame_number:06d}.jpg"
            minio_key    = f"{match_id}/{frame_name}"
            local_path   = frames_dir / frame_name

            try:
                minio_client.fget_object(settings.MINIO_BUCKET_PENDING, minio_key, str(local_path))
            except Exception as exc:
                logger.warning('{"event":"frame_dl_error","key":"%s","error":"%s"}', minio_key, str(exc))
                continue

            img = cv2.imread(str(local_path))
            if img is None:
                logger.warning('{"event":"frame_read_error","path":"%s"}', local_path)
                continue

            detections = json.loads(dets_json)
            for det in detections:
                x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
                label = det.get("class_name", "player")
                conf  = det.get("confidence", 0.0)

                if "own" in label:
                    color = _COLOR_OWN
                    text  = f"Own {conf:.2f}"
                elif "other" in label:
                    color = _COLOR_OTHER
                    text  = f"Other {conf:.2f}"
                else:
                    color = _COLOR_DEF
                    text  = f"{label} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, text, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            annotated_path = frames_dir / f"ann_{frame_number:06d}.jpg"
            cv2.imwrite(str(annotated_path), img)
            annotated_paths.append(annotated_path)

        if not annotated_paths:
            raise RuntimeError("No annotated frames produced")

        # Use FFmpeg for H.264 output so browsers can play it directly
        frame_list_path = tmp_dir / "frames.txt"
        with open(frame_list_path, "w") as fh:
            for p in annotated_paths:
                fh.write(f"file '{p}'\n")
                fh.write(f"duration {1 / FRAMES_PER_SEC}\n")

        cmd = [
            "ffmpeg",
            "-f", "concat", "-safe", "0", "-i", str(frame_list_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23",
            str(output_mp4), "-y",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg render failed: {result.stderr[-500:]}")

        output_key = f"{match_id}/output.mp4"
        minio_client.fput_object(
            settings.MINIO_BUCKET_OUTPUT, output_key, str(output_mp4),
            content_type="video/mp4",
        )
        logger.info('{"event":"video_uploaded","match_id":"%s","key":"%s"}', match_id, output_key)

        db["matches"].update_one(
            {"_id": ObjectId(match_id)},
            {"$set": {
                "status":       "done",
                "output_video": output_key,
                "updated_at":   datetime.now(timezone.utc),
            }},
        )
        logger.info('{"event":"match_done","match_id":"%s"}', match_id)

        for key in (f"frames:{match_id}:results", f"frames:{match_id}:meta",
                    f"frames:{match_id}:total", f"frames:{match_id}:rendering"):
            redis_client.delete(key)

    except Exception:
        logger.exception('{"event":"render_error","match_id":"%s"}', match_id)
        db["matches"].update_one(
            {"_id": ObjectId(match_id)},
            {"$set": {"status": "error", "updated_at": datetime.now(timezone.utc)}},
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
