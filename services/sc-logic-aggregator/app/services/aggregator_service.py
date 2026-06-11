import json
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests
from bson import ObjectId

logger = logging.getLogger(__name__)

FRAMES_PER_SEC = 2

_COLOR_OWN   = (0, 200,   0)   # green
_COLOR_OTHER = (0,   0, 200)   # red
_COLOR_DEF   = (200, 200,  0)  # cyan

# Distància màxima en coords normalitzades per considerar el mateix jugador entre frames
_MATCH_DIST = 0.06


def _find_closest_det(cx: float, cy: float, dets: list[dict]) -> dict | None:
    best, best_d = None, _MATCH_DIST ** 2
    for d in dets:
        dx = (d["x1"] + d["x2"]) / 2 - cx
        dy = (d["y1"] + d["y2"]) / 2 - cy
        dist = dx * dx + dy * dy
        if dist < best_d:
            best_d, best = dist, d
    return best


def _smooth_frame_map(frame_map: dict[int, dict]) -> dict[int, dict]:
    """
    Suavització temporal: per a cada detecció, vota amb els frames ±1.
    Majoria de 3 vots determina l'etiqueta final (current, prev, next).
    Estabilitza les caixes que salten entre OWN/OTHER cada frame.
    """
    keys = sorted(frame_map.keys())
    smoothed: dict[int, dict] = {}

    for i, fn in enumerate(keys):
        prev_dets = frame_map[keys[i - 1]]["detections"] if i > 0 else []
        next_dets = frame_map[keys[i + 1]]["detections"] if i < len(keys) - 1 else []
        new_dets  = []

        for det in frame_map[fn]["detections"]:
            cx = (det["x1"] + det["x2"]) / 2
            cy = (det["y1"] + det["y2"]) / 2
            cur_label = det.get("class_name", "person")

            votes = [cur_label]
            for adj in (prev_dets, next_dets):
                match = _find_closest_det(cx, cy, adj)
                if match:
                    votes.append(match.get("class_name", "person"))

            own_v   = sum(1 for v in votes if "own"   in v)
            other_v = sum(1 for v in votes if "other" in v)

            new_det = dict(det)
            if own_v > other_v:
                new_det["class_name"] = "player_own"
            elif other_v > own_v:
                new_det["class_name"] = "player_other"
            # empat → manté l'etiqueta actual

            new_dets.append(new_det)

        smoothed[fn] = {**frame_map[fn], "detections": new_dets}

    return smoothed


def _compute_stats(frame_map: dict[int, dict], start_s: float, end_s: float) -> dict:
    total_frames = len(frame_map)
    all_confs: list[float] = []
    counts_total: list[int] = []
    counts_own:   list[int] = []
    counts_other: list[int] = []
    max_count, max_frame, max_time = 0, 1, 0.0

    for fn, data in frame_map.items():
        dets  = data.get("detections") or []
        ts    = data.get("timestamp_s", 0.0)
        n     = len(dets)
        n_own   = sum(1 for d in dets if "own"   in (d.get("class_name") or ""))
        n_other = sum(1 for d in dets if "other" in (d.get("class_name") or ""))
        counts_total.append(n)
        counts_own.append(n_own)
        counts_other.append(n_other)
        all_confs.extend(d.get("confidence", 0.0) for d in dets)
        if n > max_count:
            max_count, max_frame, max_time = n, fn, ts

    def mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    avg_total = mean(counts_total)
    avg_own   = mean(counts_own)
    avg_other = mean(counts_other)
    avg_conf  = mean(all_confs)
    pct_own   = (sum(counts_own) / max(sum(counts_total), 1)) * 100

    return {
        "total_frames":         total_frames,
        "avg_players_per_frame": round(avg_total, 2),
        "avg_own_per_frame":     round(avg_own, 2),
        "avg_other_per_frame":   round(avg_other, 2),
        "pct_own":               round(pct_own, 1),
        "avg_confidence":        round(avg_conf, 3),
        "max_density_frame":     max_frame,
        "max_density_time_s":    round(max_time, 2),
        "max_density_count":     max_count,
        "duration_s":            round(end_s - start_s, 1),
    }


def _generate_ai_report(stats: dict, ollama_base_url: str) -> str | None:
    prompt = (
        "Ets un analista de FUTBOL SALA. Un sistema de visió per computador ha analitzat "
        "un fragment de partit i ha obtingut les estadístiques següents:\n\n"
        f"- Fragment analitzat: {stats['duration_s']:.0f} s ({stats['total_frames']} frames)\n"
        f"- Jugadors detectats de mitjana per frame: {stats['avg_players_per_frame']:.1f} "
        f"({stats['avg_own_per_frame']:.1f} propis, {stats['avg_other_per_frame']:.1f} rivals)\n"
        f"- % de presència de l'equip propi: {stats['pct_own']:.0f}%\n"
        f"- Fiabilitat de les deteccions: {stats['avg_confidence']:.0%}\n"
        f"- Pic d'intensitat: {stats['max_density_count']} jugadors al segon {stats['max_density_time_s']:.1f}\n\n"
        "Escriu un resum inicial en 3-4 frases en català. "
        "Interpreta les dades en clau de futbol sala (5 jugadors per equip a la pista), "
        "comenta la distribució entre equips, la qualitat de les deteccions i el moment de màxima intensitat. "
        "Escriu directament sense introducció."
    )
    try:
        resp = requests.post(
            f"{ollama_base_url}/api/generate",
            json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception as exc:
        logger.warning('{"event":"ollama_unavailable","error":"%s"}', exc)
        return None


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
                # Legacy format: just a list of detections (no timestamp stored)
                frame_map[fn] = {"timestamp_s": (fn - 1) / FRAMES_PER_SEC, "detections": data}
            else:
                frame_map[fn] = data

        # Temporal smoothing: reduces per-frame label flickering
        frame_map = _smooth_frame_map(frame_map)

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

        # Visual constants scaled to video resolution
        line_thick  = max(2, vid_w // 960)
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_scale  = max(0.5, vid_w / 2000)
        text_thick  = max(1, vid_w // 1920)
        pad         = max(2, vid_w // 1920)

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

                # Map video frame time → nearest inference result
                t_relative    = frame_idx / fps_orig
                inf_frame_num = int(t_relative * FRAMES_PER_SEC) + 1
                result = frame_map.get(inf_frame_num) or frame_map.get(max(1, inf_frame_num - 1))

                if result:
                    for det in result["detections"]:
                        x1 = int(det["x1"] * vid_w)
                        y1 = int(det["y1"] * vid_h)
                        x2 = int(det["x2"] * vid_w)
                        y2 = int(det["y2"] * vid_h)
                        label = det.get("class_name", "person")
                        conf  = det.get("confidence", 0.0)

                        if "own" in label:
                            color = _COLOR_OWN
                            text  = f"Own {conf:.2f}"
                        elif "other" in label:
                            color = _COLOR_OTHER
                            text  = f"Other {conf:.2f}"
                        else:
                            color = _COLOR_DEF
                            text  = f"? {conf:.2f}"

                        # Bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thick)

                        # Label: filled background + white text
                        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thick)
                        lbl_y1 = max(0, y1 - th - 2 * pad)
                        lbl_y2 = y1
                        cv2.rectangle(img, (x1, lbl_y1), (x1 + tw + 2 * pad, lbl_y2), color, -1)
                        cv2.putText(img, text, (x1 + pad, lbl_y2 - pad),
                                    font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

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

        ai_stats  = _compute_stats(frame_map, start_s, end_s)
        ai_report = _generate_ai_report(ai_stats, settings.OLLAMA_BASE_URL)
        logger.info('{"event":"ai_report_done","match_id":"%s","has_report":%s}',
                    match_id, str(ai_report is not None).lower())

        db["matches"].update_one(
            {"_id": ObjectId(match_id)},
            {"$set": {
                "status":       "done",
                "output_video": output_key,
                "ai_stats":     ai_stats,
                "ai_report":    ai_report,
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
