import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from jose import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from passlib.hash import bcrypt

from app.config import settings
from app.repositories import auth_repository, user_profile_repository

logger = logging.getLogger(__name__)


def _create_access_token(user_id: str, role: str, team_ids: list[str]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": user_id, "role": role, "team_ids": team_ids, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def _create_refresh_token_pair() -> tuple[str, str]:
    raw = secrets.token_urlsafe(64)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, token_hash


def _verify_password(password: str, stored_hash: str, force_reset: bool) -> bool:
    if force_reset:
        return password == stored_hash
    return bcrypt.verify(password, stored_hash)


async def login(
    email: str,
    password: str,
    auth_db: AsyncIOMotorDatabase,
    app_db: AsyncIOMotorDatabase,
) -> tuple[str, str]:
    user = await auth_repository.get_user_by_email(auth_db, email)
    if not user or not user.get("active", False):
        raise ValueError("Credencials incorrectes")

    force_reset = user.get("force_reset", False)
    if not _verify_password(password, user["password_hash"], force_reset):
        raise ValueError("Credencials incorrectes")

    user_id = str(user["_id"])

    if force_reset:
        new_hash = bcrypt.hash(password)
        await auth_repository.update_user_password(auth_db, user_id, new_hash)
        logger.info("force_reset completat per user_id=%s", user_id)

    profile = await user_profile_repository.get_profile_by_user_id(app_db, user_id)
    team_ids = [str(t) for t in (profile or {}).get("team_ids", [])]

    access_token = _create_access_token(user_id, user["role"], team_ids)
    refresh_raw, refresh_hash = _create_refresh_token_pair()
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    await auth_repository.create_refresh_token(auth_db, user_id, refresh_hash, expires_at)

    return access_token, refresh_raw


async def refresh(
    refresh_token_raw: str,
    auth_db: AsyncIOMotorDatabase,
    app_db: AsyncIOMotorDatabase,
) -> tuple[str, str]:
    token_hash = hashlib.sha256(refresh_token_raw.encode()).hexdigest()
    token_doc = await auth_repository.get_refresh_token_by_hash(auth_db, token_hash)

    if token_doc is None:
        raise ValueError("Token invàlid o expirat")

    if token_doc["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        await auth_repository.delete_refresh_token(auth_db, token_hash)
        raise ValueError("Token invàlid o expirat")

    user_id = str(token_doc["user_id"])
    await auth_repository.delete_refresh_token(auth_db, token_hash)

    user = await auth_repository.get_user_by_id(auth_db, user_id)
    if not user or not user.get("active", False):
        raise ValueError("Token invàlid o expirat")

    profile = await user_profile_repository.get_profile_by_user_id(app_db, user_id)
    team_ids = [str(t) for t in (profile or {}).get("team_ids", [])]

    access_token = _create_access_token(user_id, user["role"], team_ids)
    new_raw, new_hash = _create_refresh_token_pair()
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    await auth_repository.create_refresh_token(auth_db, user_id, new_hash, expires_at)

    return access_token, new_raw
