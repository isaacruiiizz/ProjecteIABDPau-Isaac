import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_app_db, get_current_user, get_s3
from app.schemas.auth import TokenPayload
from app.schemas.matches import MatchConfigRequest, MatchConfigResponse, MatchCreateResponse
from app.services import match_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/matches", tags=["matches"])


@router.post("", response_model=MatchCreateResponse, status_code=201)
async def create_match(
    video: UploadFile = File(..., description="Fitxer .mp4 del partit"),
    title: str = Form(..., min_length=1, max_length=200),
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
    s3=Depends(get_s3),
):
    if not video.content_type or "video" not in video.content_type:
        raise HTTPException(status_code=422, detail="El fitxer ha de ser un vídeo MP4")

    file_bytes = await video.read()
    try:
        result = await match_service.upload_match(
            file_bytes=file_bytes,
            title=title,
            user_id=current_user.sub,
            s3=s3,
            db=db,
        )
    except Exception:
        logger.exception("Error creant el partit")
        raise HTTPException(status_code=500, detail="Error pujant el vídeo")

    return MatchCreateResponse(**result)


@router.patch("/{match_id}/config", response_model=MatchConfigResponse)
async def update_match_config(
    match_id: str,
    body: MatchConfigRequest,
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
):
    try:
        result = await match_service.update_config(
            match_id=match_id,
            roi_polygon=body.roi_polygon,
            start_seconds=body.start_seconds,
            end_seconds=body.end_seconds,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Error actualitzant config del partit")
        raise HTTPException(status_code=500, detail="Error actualitzant la configuració")
    return MatchConfigResponse(**result)
