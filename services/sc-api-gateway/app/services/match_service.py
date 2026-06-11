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
        "match_id":          str(doc["_id"]),
        "title":             doc["title"],
        "status":            doc["status"],
        "created_at":        doc["created_at"],
        "start_seconds":     doc.get("start_seconds"),
        "end_seconds":       doc.get("end_seconds"),
        "download_url":      download_url,
        "ai_stats":          doc.get("ai_stats"),
        "ai_report":         doc.get("ai_report"),
        "ai_report_refined": doc.get("ai_report_refined"),
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

    conf_pct  = f"{float(stats['avg_confidence']) * 100:.0f}%" if stats.get('avg_confidence') else "—"
    max_time  = _fmt('max_density_time_s')
    max_count = _fmt('max_density_count')

    prompt = (
        "Eres un analista táctico experto en FÚTBOL SALA (futsal). "
        "Tienes acceso a datos objetivos de un fragmento de partido obtenidos por visión por computador "
        "y al contexto real aportado por el entrenador/a. "
        "Tu misión es generar un informe táctico profesional, concreto y accionable en castellano.\n\n"

        "NOTAS SOBRE LOS DATOS:\n"
        "- El sistema detecta jugadores por color de camiseta y los clasifica como 'propios' o 'rivales'\n"
        "- En fútbol sala hay máximo 5 jugadores por equipo en pista (portero incluido)\n"
        "- Una alta densidad en un instante concreto indica presión alta, jugada de estrategia o disputa en el centro\n"
        "- Usa el contexto del entrenador/a para interpretar las anomalías numéricas\n\n"

        "DATOS OBJETIVOS (IA):\n"
        f"• Fragmento analizado: {_fmt('duration_s')} s  |  {_fmt('total_frames')} frames\n"
        f"• Jugadores detectados (media/frame): {_fmt('avg_players_per_frame')} totales "
        f"→ {_fmt('avg_own_per_frame')} propios ({_fmt('pct_own')}%) / {_fmt('avg_other_per_frame')} rivales\n"
        f"• Fiabilidad de las detecciones: {conf_pct}\n"
        f"• PICO DE INTENSIDAD: {max_count} jugadores en el segundo {max_time} "
        "(máxima aglomeración — probable presión, estrategia o disputa decisiva)\n\n"

        "PRIMERA LECTURA AUTOMÁTICA (sin contexto):\n"
        f"{initial}\n\n"

        "CONTEXTO DEL ENTRENADOR/A:\n"
        f"{user_context}\n\n"

        "INSTRUCCIONES PARA EL INFORME:\n"
        "- Usa la terminología específica del fútbol sala: portero, cierre, ala derecha/izquierda, pivot, "
        "sistemas tácticos (2-2, 1-2-1 en diamante, 3-1, 1-3), pressing 4+1, portero-jugador, "
        "rotación defensiva, basculación, transición ataque-defensa, doble pivot, trampa de pressing, "
        "salida de portero, bloque defensivo bajo, pase interior, juego de esquina.\n"
        "- Interpreta los datos numéricos a la luz del contexto del entrenador/a.\n"
        "- Cita el segundo exacto del pico de intensidad y explícalo tácticamente.\n"
        "- Las recomendaciones deben ser ejercicios o situaciones concretas de entrenamiento.\n"
        "- Escribe directamente el informe, sin preámbulo ni conclusión genérica.\n\n"

        "ESTRUCTURA DEL INFORME (síguela exactamente, usa **Título:** y guiones -):\n\n"

        "**Lectura del fragmento:**\n"
        "- Describe la fase de juego, el ritmo y la distribución sobre la pista\n"
        "- Relaciona los datos numéricos con lo que realmente ocurrió\n\n"

        "**Sistema y comportamiento táctico:**\n"
        "- Sistema defensivo y ofensivo detectado (propio y rival)\n"
        "- Rol del pivot, el cierre y los alas en la fase analizada\n"
        "- Pressing, coberturas y transiciones observadas\n\n"

        f"**Momento clave (s. {max_time}):**\n"
        f"- ¿Por qué hay {max_count} jugadores concentrados en ese instante?\n"
        "- ¿Cómo se relaciona con el contexto descrito? ¿Qué jugada o situación podría ser?\n\n"

        "**Diagnóstico del equipo:**\n"
        "- Puntos fuertes a mantener y potenciar\n"
        "- Vulnerabilidades prioritarias (máximo 3, ordenadas por impacto)\n\n"

        "**Plan de acción para el entrenamiento:**\n"
        "- Propón 3-4 ejercicios o situaciones específicas de fútbol sala para corregir los puntos débiles\n"
        "- Cada punto debe ser directamente aplicable a la próxima sesión\n\n"

        "Extensión total: 300-400 palabras. Tono directo, profesional y orientado a la mejora."
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

    await db["matches"].update_one(
        {"_id": ObjectId(match_id)},
        {"$set": {"ai_report_refined": text, "updated_at": datetime.now(timezone.utc)}},
    )
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
