import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_app_db, get_current_user, get_redis, get_s3
from app.schemas.auth import TokenPayload
from app.schemas.matches import MatchConfigRequest, MatchConfigResponse, MatchCreateResponse, MatchDetail, MatchListItem, ProcessMatchResponse
from app.services import match_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/matches", tags=["matches"])


@router.get("", response_model=list[MatchListItem])
async def list_matches(
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
):
    results = await match_service.list_matches(current_user.sub, db)
    return [MatchListItem(**r) for r in results]


@router.delete("/{match_id}", status_code=204)
async def delete_match(
    match_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
    s3=Depends(get_s3),
):
    try:
        await match_service.delete_match(match_id, current_user.sub, db, s3)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Error eliminant el partit")
        raise HTTPException(status_code=500, detail="Error eliminant el partit")


@router.get("/{match_id}", response_model=MatchDetail)
async def get_match(
    match_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
):
    try:
        result = await match_service.get_match_detail(match_id, current_user.sub, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Error obtenint detall del partit")
        raise HTTPException(status_code=500, detail="Error obtenint el partit")
    return MatchDetail(**result)


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


@router.post("/{match_id}/process", response_model=ProcessMatchResponse, status_code=202)
async def process_match(
    match_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    db=Depends(get_app_db),
    redis=Depends(get_redis),
):
    try:
        result = await match_service.process_match(
            match_id=match_id,
            redis=redis,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception:
        logger.exception("Error iniciant el processament del partit")
        raise HTTPException(status_code=500, detail="Error iniciant el processament")
    return ProcessMatchResponse(**result)


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
