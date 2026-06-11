import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

BUCKET_RAW     = "raw-videos"
BUCKET_PENDING = "pending-frames"
FRAMES_PER_SEC = 2


def _roi_bounding_box(roi_polygon: list[dict]) -> tuple[int, int, int, int]:
    """Retorna (x, y, w, h) del bounding box rectangular del polígon ROI."""
    xs = [int(p["x"]) for p in roi_polygon]
    ys = [int(p["y"]) for p in roi_polygon]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    # FFmpeg necessita amplada i alçada parells
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1
    return x, y, w, h


def extract_frames(match_doc: dict, s3) -> int:
    """
    Descarrega el vídeo de MinIO, extreu frames aplicant ROI i rang de temps
    amb FFmpeg, i puja els frames a pending-frames/{match_id}/.

    Retorna el nombre de frames pujats.
    """
    match_id      = str(match_doc["_id"])
    minio_key     = match_doc["video_raw"]
    roi_polygon   = match_doc["roi_polygon"]
    start_seconds = float(match_doc["start_seconds"])
    end_seconds   = float(match_doc["end_seconds"])

    x, y, w, h = _roi_bounding_box(roi_polygon)
    logger.info("match=%s ROI bounding box: x=%d y=%d w=%d h=%d", match_id, x, y, w, h)

    tmp_dir    = Path(tempfile.mkdtemp(prefix=f"sc-frames-{match_id[:8]}-"))
    video_path = tmp_dir / "original.mp4"
    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()

    try:
        # 1. Descarrega el vídeo de MinIO
        logger.info("Descarregant vídeo: bucket=%s key=%s", BUCKET_RAW, minio_key)
        s3.download_file(BUCKET_RAW, minio_key, str(video_path))
        logger.info("Vídeo descarregat a %s", video_path)

        # 2. Extreu frames amb FFmpeg
        #    -ss i -to  → rang temporal
        #    fps=N      → N frames per segon
        #    crop=w:h:x:y → retall rectangular del ROI
        frame_pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-ss", str(start_seconds),
            "-to", str(end_seconds),
            "-i", str(video_path),
            "-vf", f"fps={FRAMES_PER_SEC},crop={w}:{h}:{x}:{y}",
            "-q:v", "2",
            frame_pattern,
            "-y",
        ]
        logger.info("Executant FFmpeg: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg ha fallat:\n{result.stderr[-500:]}")

        # 3. Puja cada frame a pending-frames
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_files:
            raise RuntimeError(
                "FFmpeg no ha generat cap frame. Comprova el rang de temps i el ROI."
            )

        uploaded = 0
        for frame_file in frame_files:
            object_key = f"{match_id}/{frame_file.name}"
            s3.upload_file(str(frame_file), BUCKET_PENDING, object_key)
            uploaded += 1

        logger.info("match=%s frames pujats a pending-frames: %d", match_id, uploaded)
        return uploaded

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
