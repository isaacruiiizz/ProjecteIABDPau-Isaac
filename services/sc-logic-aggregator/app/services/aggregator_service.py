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

# Max normalised center-to-center distance to consider two detections the same player
_MATCH_THRESHOLD_SQ = 0.09   # 0.3 in normalised coords


def _interpolate_detections(dets_a: list, dets_b: list, alpha: float) -> list:
    """
    Match detections from two consecutive inference keyframes by proximity and
    linearly interpolate their bounding boxes.  alpha=0 → dets_a, alpha=1 → dets_b.
    Unmatched detections pass through unchanged.
    """
    if not dets_a:
        return dets_b
    if not dets_b:
        return dets_a

    used_b: set[int] = set()
    result = []

    for det_a in dets_a:
        cx_a = (det_a["x1"] + det_a["x2"]) / 2
        cy_a = (det_a["y1"] + det_a["y2"]) / 2

        best_sq = _MATCH_THRESHOLD_SQ
        best_j  = -1
        for j, det_b in enumerate(dets_b):
            if j in used_b:
                continue
            cx_b = (det_b["x1"] + det_b["x2"]) / 2
            cy_b = (det_b["y1"] + det_b["y2"]) / 2
            sq = (cx_b - cx_a) ** 2 + (cy_b - cy_a) ** 2
            if sq < best_sq:
                best_sq = sq
                best_j  = j

        if best_j >= 0:
            b = dets_b[best_j]
            used_b.add(best_j)
            result.append({
                "x1":         det_a["x1"] + (b["x1"] - det_a["x1"]) * alpha,
                "y1":         det_a["y1"] + (b["y1"] - det_a["y1"]) * alpha,
                "x2":         det_a["x2"] + (b["x2"] - det_a["x2"]) * alpha,
                "y2":         det_a["y2"] + (b["y2"] - det_a["y2"]) * alpha,
                "class_name": b.get("class_name") if alpha >= 0.5 else det_a.get("class_name", "player"),
                "confidence": det_a["confidence"] + (b["confidence"] - det_a["confidence"]) * alpha,
            })
        else:
            result.append(det_a)

    for j, det_b in enumerate(dets_b):
        if j not in used_b:
            result.append(det_b)

    return result


def handle_frame_result(payload: dict, redis_client, minio_client, db, settings) -> None:
    match_id     = payload["match_id"]
    frame_number = int(payload["frame_number"])
    detections   = payload.get("detections") or []
    timestamp_s  = float(payload.get("timestamp_s") or 0.0)

    frame_data = json.dumps({"timestamp_s": timestamp_s, "detections": detections})
    redis_client.hset(f"frames:{match_id}:results", str(frame_number), frame_data)
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
        locked = redis_client.set(f"frames:{match_id}:rendering", "1", nx=True, ex=3600)
        if locked:
            _render_and_finalize(match_id, redis_client, minio_client, db, settings)


def _render_and_finalize(match_id: str, redis_client, minio_client, db, settings) -> None:
    logger.info('{"event":"render_start","match_id":"%s"}', match_id)

    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"sc-render-{match_id[:8]}-"))
    output_mp4 = tmp_dir / "output.mp4"

    try:
        all_results = redis_client.hgetall(f"frames:{match_id}:results")
        if not all_results:
            raise RuntimeError("No frame results in Redis")

        # Build map: frame_number -> {timestamp_s, detections}
        frame_map: dict[int, dict] = {}
        for fn_str, data_json in all_results.items():
            data = json.loads(data_json)
            fn = int(fn_str)
            if isinstance(data, list):
                frame_map[fn] = {"timestamp_s": (fn - 1) / FRAMES_PER_SEC, "detections": data}
            else:
                frame_map[fn] = data

        # Get match metadata
        match_doc = db["matches"].find_one(
            {"_id": ObjectId(match_id)},
            {"video_raw": 1, "start_seconds": 1, "end_seconds": 1},
        )
        if not match_doc or not match_doc.get("video_raw"):
            raise RuntimeError("Match has no video_raw field")

        start_s = float(match_doc.get("start_seconds") or 0.0)
        end_s   = float(match_doc.get("end_seconds") or 0.0)

        # Download original video
        orig_path = tmp_dir / "original.mp4"
        minio_client.fget_object(settings.MINIO_BUCKET_RAW, match_doc["video_raw"], str(orig_path))
        logger.info('{"event":"original_downloaded","match_id":"%s"}', match_id)

        # Trim to the selected time range (stream copy = fast, no re-encode)
        trimmed_path = tmp_dir / "trimmed.mp4"
        trim_cmd = [
            "ffmpeg", "-ss", str(start_s), "-to", str(end_s),
            "-i", str(orig_path), "-c", "copy",
            str(trimmed_path), "-y",
        ]
        r = subprocess.run(trim_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg trim failed: {r.stderr[-300:]}")

        # Open trimmed video with OpenCV
        cap = cv2.VideoCapture(str(trimmed_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open trimmed video")

        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Pipe raw BGR frames directly into FFmpeg → H.264 (no intermediate frame files)
        encode_cmd = [
            "ffmpeg",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{vid_w}x{vid_h}", "-r", str(fps_orig),
            "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23",
            str(output_mp4), "-y",
        ]
        ffmpeg_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

        frame_idx = 0
        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    break

                t_relative = frame_idx / fps_orig

                # Current inference keyframe and interpolation alpha (0→1)
                inf_fn = int(t_relative * FRAMES_PER_SEC) + 1
                inf_t0 = (inf_fn - 1) / FRAMES_PER_SEC
                alpha  = min(1.0, (t_relative - inf_t0) * FRAMES_PER_SEC)

                res_a = frame_map.get(inf_fn)
                res_b = frame_map.get(inf_fn + 1)
                dets_a = res_a["detections"] if res_a else []
                dets_b = res_b["detections"] if res_b else []
                detections = _interpolate_detections(dets_a, dets_b, alpha)

                for det in detections:
                    x1 = int(det["x1"] * vid_w)
                    y1 = int(det["y1"] * vid_h)
                    x2 = int(det["x2"] * vid_w)
                    y2 = int(det["y2"] * vid_h)
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

                ffmpeg_proc.stdin.write(img.tobytes())
                frame_idx += 1
        finally:
            cap.release()
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

        if ffmpeg_proc.returncode != 0:
            raise RuntimeError("FFmpeg encoding failed")

        logger.info('{"event":"render_done","match_id":"%s","frames":%d}', match_id, frame_idx)

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
