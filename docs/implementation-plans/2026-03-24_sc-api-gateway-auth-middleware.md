# Pla d'implementació: Middleware d'autorització per rols

**Ticket:** PJM-23
**Data:** 2026-03-24
**Estat:** Completat ✓ (2026-03-24)

---

## Objectiu

Afegir a `sc-api-gateway` les dependències FastAPI que gestionen:
1. Validació del JWT i extracció del payload (`get_current_user`)
2. Control d'accés per rol (`require_roles`)
3. Filtre MongoDB per `team_id` (`get_team_filter`)
4. Validació de la capçalera interna (`verify_internal_api_key`)

---

## Context (de les specs)

**JWT payload emès pel login** (`auth_service._create_access_token`):
```python
{"sub": user_id, "role": role, "team_ids": [str, ...], "exp": datetime}
```

**Regles d'accés per rol** (spec 04-seguretat-bd.md §3):
- `admin` → sense filtre `team_id`, veu tot
- `coach` / `assistant` / `player` → `{ "team_id": { "$in": token.team_ids } }`

**Internal API Key** (spec 04-seguretat-bd.md §1):
- Capçalera `X-Internal-API-Key` amb valor de la variable `INTERNAL_API_KEY`
- Valida peticions entre microserveis interns

**Validació del JWT:**
- Es fa **localment** sense consultar cap base de dades
- `team_ids` i `role` s'extreuen directament del payload del token

---

## Fitxers afectats

| Fitxer | Acció | Motiu |
|--------|-------|-------|
| `app/schemas/auth.py` | Ampliar | Afegir `TokenPayload` (model Pydantic del payload del JWT) |
| `app/dependencies.py` | Ampliar | Afegir les 4 dependències noves sense tocar les existents |

Cap fitxer nou. Cap router modificat en aquest ticket (els routers s'injectaran quan s'implementin els endpoints de `matches` i `players`).

---

## Fase 1 — `schemas/auth.py`: afegir `TokenPayload`

Afegir al final del fitxer existent (sense tocar `LoginRequest` ni `TokenResponse`):

```python
class TokenPayload(BaseModel):
    sub: str          # user_id
    role: str         # admin | coach | assistant | player
    team_ids: list[str]
    exp: int
```

---

## Fase 2 — `dependencies.py`: afegir middleware

Afegir al final del fitxer existent (sense tocar `get_auth_db` ni `get_app_db`):

### 2a. Imports nous necessaris

```python
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from bson import ObjectId

from app.config import settings
from app.schemas.auth import TokenPayload
```

### 2b. `get_current_user` — valida JWT localment

```python
_bearer = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> TokenPayload:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invàlid o expirat",
        )
    return TokenPayload(**payload)
```

### 2c. `require_roles` — factory de control d'accés per rol

```python
def require_roles(*roles: str):
    async def _check(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No tens permisos per a aquesta acció",
            )
        return current_user
    return _check
```

**Ús als routers:**
```python
# Només admin i coach poden crear partits
@router.post("/matches")
async def create_match(user: TokenPayload = Depends(require_roles("admin", "coach"))):
    ...
```

### 2d. `get_team_filter` — filtre MongoDB per `team_id`

```python
def get_team_filter(current_user: TokenPayload) -> dict:
    if current_user.role == "admin":
        return {}
    return {"team_id": {"$in": [ObjectId(tid) for tid in current_user.team_ids]}}
```

**Nota:** No és una dependència FastAPI (no usa `Depends`). S'utilitza com a funció pura als `services/` passant-li el `current_user` ja resolt.

### 2e. `verify_internal_api_key` — capçalera entre microserveis

```python
async def verify_internal_api_key(
    x_internal_api_key: str = Header(..., alias="X-Internal-API-Key"),
) -> None:
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clau interna invàlida",
        )
```

**Ús als routers interns:**
```python
@router.post("/internal/some-endpoint")
async def internal_endpoint(_: None = Depends(verify_internal_api_key)):
    ...
```

---

## Resum de dependències exportades

| Funció | Tipus | Ús |
|--------|-------|----|
| `get_current_user` | `Depends()` | Tots els endpoints autenticats |
| `require_roles("admin", "coach")` | `Depends()` factory | Endpoints amb restricció de rol |
| `get_team_filter(user)` | Funció pura | Capa `services/` per filtrar queries |
| `verify_internal_api_key` | `Depends()` | Endpoints cridats per microserveis interns |

---

## Verificació post-implementació

- [ ] `GET /health` segueix sense autenticació (no usa `Depends(get_current_user)`)
- [ ] `POST /auth/login` i `POST /auth/refresh` segueixen sense autenticació
- [ ] Token expirat → `401 Unauthorized`
- [ ] Token vàlid però rol incorrecte → `403 Forbidden`
- [ ] `X-Internal-API-Key` incorrecta → `403 Forbidden`
- [ ] Coach consulta → filtre `team_id` aplicat
- [ ] Admin consulta → sense filtre

---

## Dependències Python

`python-jose` ja és a `requirements.txt` (usada per `auth_service.py`). `bson` ve amb `motor`. Cap dependència nova.
