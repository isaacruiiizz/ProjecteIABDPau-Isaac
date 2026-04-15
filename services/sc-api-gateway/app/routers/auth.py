import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import settings
from app.dependencies import get_app_db, get_auth_db
from app.schemas.auth import LoginRequest, TokenResponse
from app.services import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])

_COOKIE_NAME = "refresh_token"
_COOKIE_MAX_AGE = 7 * 24 * 3600


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite="lax",
        path="/auth",
        max_age=_COOKIE_MAX_AGE,
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    auth_db: AsyncIOMotorDatabase = Depends(get_auth_db),
    app_db: AsyncIOMotorDatabase = Depends(get_app_db),
) -> JSONResponse:
    try:
        access_token, refresh_raw = await auth_service.login(
            body.email, body.password, auth_db, app_db
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Credencials incorrectes")

    response = JSONResponse(
        content={"access_token": access_token, "token_type": "bearer"}
    )
    _set_refresh_cookie(response, refresh_raw)
    return response


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str | None = Cookie(default=None, alias="refresh_token"),
    auth_db: AsyncIOMotorDatabase = Depends(get_auth_db),
    app_db: AsyncIOMotorDatabase = Depends(get_app_db),
) -> JSONResponse:
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Token invàlid o expirat")

    try:
        access_token, new_refresh_raw = await auth_service.refresh(
            refresh_token, auth_db, app_db
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Token invàlid o expirat")

    response = JSONResponse(
        content={"access_token": access_token, "token_type": "bearer"}
    )
    _set_refresh_cookie(response, new_refresh_raw)
    return response
