import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.dependencies import get_redis, get_s3, require_roles
from app.schemas.labeling import LabelingUploadResponse
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