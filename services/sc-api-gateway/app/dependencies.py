from bson import ObjectId
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import settings
from app.schemas.auth import TokenPayload

# Inicialitzats al lifespan de main.py (startup) i tancats en shutdown.
# Cap altre mòdul ha d'instanciar AsyncIOMotorClient directament.
auth_client: AsyncIOMotorClient | None = None
app_client: AsyncIOMotorClient | None = None


def get_auth_db() -> AsyncIOMotorDatabase:
    """Retorna la BD sc-auth-db (credencials, sessions). Usat via Depends()."""
    return auth_client.get_default_database()


def get_app_db() -> AsyncIOMotorDatabase:
    """Retorna la BD sc-app-db (partits, jugadors, equips). Usat via Depends()."""
    return app_client.get_default_database()


# ---------------------------------------------------------------------------
# Autenticació i autorització
# ---------------------------------------------------------------------------

_bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> TokenPayload:
    """Valida el JWT localment (sense consultar BD) i retorna el payload."""
    try:
        payload = jwt.decode(
            credentials.credentials, settings.JWT_SECRET, algorithms=["HS256"]
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invàlid o expirat",
        )
    return TokenPayload(**payload)


def require_roles(*roles: str):
    """Factory: retorna un Depends() que exigeix que l'usuari tingui un dels rols indicats."""
    async def _check(
        current_user: TokenPayload = Depends(get_current_user),
    ) -> TokenPayload:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tens permisos per a aquesta acció",
            )
        return current_user
    return _check


def get_team_filter(current_user: TokenPayload) -> dict:
    """Retorna el filtre MongoDB per team_id segons el rol.

    - admin  → {} (sense filtre, veu tot)
    - resta  → {"team_id": {"$in": [ObjectId, ...]}}

    Funció pura: no és un Depends(). Es crida des de services/ passant-li
    el current_user ja resolt.
    """
    if current_user.role == "admin":
        return {}
    return {"team_id": {"$in": [ObjectId(tid) for tid in current_user.team_ids]}}


async def verify_internal_api_key(
    x_internal_api_key: str = Header(..., alias="X-Internal-API-Key"),
) -> None:
    """Valida la capçalera X-Internal-API-Key per a peticions entre microserveis."""
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clau interna invàlida",
        )
