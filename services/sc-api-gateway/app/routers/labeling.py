import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.dependencies import get_redis, get_s3, require_roles
from app.schemas.labeling import (
    LabelingFrameResponse,
    LabelingStartRequest,
    LabelingStartResponse,
    LabelingUploadResponse,
)
from app.services import labeling_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/labeling", tags=["labeling"])


@router.post("/upload", response_model=LabelingUploadResponse, status_code=202)
async def upload_labeling_video(
    video: UploadFile = File(..., description="Fitxer .mp4 per a etiquetatge"),
    frame_interval: int = Query(default=2, ge=1, le=60),
    _user=Depends(require_roles("admin")),
    redis=Depends(get_redis),
    s3=Depends(get_s3),
):
    file_bytes = await video.read()
    result = await labeling_service.upload_labeling_video(
        file_bytes=file_bytes,
        frame_interval=frame_interval,
        s3=s3,
        redis=redis,
    )
    return LabelingUploadResponse(**result)


@router.get("/frame", response_model=LabelingFrameResponse)
async def get_labeling_frame(
    video_key: str = Query(..., description="session_id del vídeo d'etiquetatge"),
    frame_number: int = Query(default=1, ge=1, description="Índex 1-based del frame"),
    _user=Depends(require_roles("admin")),
    s3=Depends(get_s3),
):
    """
    Retorna la URL pre-signada d'un frame concret i el total de frames extrets.

    El frontend usa total_frames per calcular els índexos representatius:
    10%, 25%, 50%, 75%, 90% del total.
    """
    result = await labeling_service.get_labeling_frame(
        video_key=video_key,
        frame_number=frame_number,
        s3=s3,
    )
    return LabelingFrameResponse(**result)


@router.post("/start", response_model=LabelingStartResponse, status_code=202)
async def start_labeling(
    body: LabelingStartRequest,
    _user=Depends(require_roles("admin")),
    redis=Depends(get_redis),
    s3=Depends(get_s3),
):
    """
    Publica tots els frames d'una sessió a labeling_frames_to_infer perquè
    sc-inference-worker els pre-anoti amb YOLOv8n usant el color de samarreta indicat.

    jersey_own_color_hsv és opcional. Si és null, sc-inference-worker fa fallback
    a la variable d'entorn JERSEY_OWN_COLOR_HSV.
    """
    result = await labeling_service.start_labeling(
        video_key=body.video_key,
        jersey_own_color_hsv=body.jersey_own_color_hsv,
        jersey_color_threshold=body.jersey_color_threshold,
        s3=s3,
        redis=redis,
    )
    return LabelingStartResponse(**result)
