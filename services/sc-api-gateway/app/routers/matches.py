import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_app_db, get_current_user, get_s3
from app.schemas.auth import TokenPayload
from app.schemas.matches import MatchCreateResponse
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
