# Pla d'Implementació — PJM-19
# [backend] Implementar POST /auth/login i POST /auth/refresh amb JWT

**Estat:** Completat ✓ (2026-03-24)
**Ticket:** PJM-19
**Data:** 2026-03-22
**Etiqueta commit:** `feat(backend): implementar POST /auth/login i POST /auth/refresh [PJM-19]`

---

## Context

Implementar el sistema d'autenticació complet de `sc-api-gateway`:
- `POST /auth/login` — verifica credencials, gestiona `force_reset`, emet Access Token (JWT) + Refresh Token (cookie HttpOnly)
- `POST /auth/refresh` — Refresh Token Rotation amb detecció de reutilització i invalidació de sessió

Els endpoints **no porten prefix `/api/v1/`** (spec punt 05-config.md).

---

## Flux d'autenticació (spec 04-seguretat-bd.md)

### POST /auth/login
```
Client envia email + password
        │
        ▼
auth_repository: get_user_by_email(auth_db, email)
        │
        ├── No trobat → 401
        ├── active=false → 401
        │
        ▼
Comprova force_reset
        │
        ├── force_reset=true:
        │     Compara password rebut == password_hash (clar, provisional)
        │     Si correcte → bcrypt hash → update_user_password(auth_db, user_id, hash)
        │     (elimina force_reset del document)
        │
        └── force_reset=false/absent:
              bcrypt.verify(password, password_hash)
              Si incorrecte → 401
        │
        ▼
user_profile_repository: get_profile_by_user_id(app_db, user_id)
        │  → obté team_ids per incloure al JWT
        │
        ▼
Genera Access Token JWT HS256:
  payload = {sub: str(user_id), role: role, team_ids: [str(id)...], exp: +15min}
        │
        ▼
Genera Refresh Token: secrets.token_urlsafe(64) (opac, 64 bytes)
Calcula token_hash = SHA-256(refresh_token_raw)
auth_repository: create_refresh_token(auth_db, user_id, token_hash, expires_at)
        │
        ▼
Resposta:
  Body: {"access_token": "...", "token_type": "bearer"}
  Cookie: refresh_token=<raw>, HttpOnly, Secure, SameSite=Strict,
          path=/auth, max_age=7*24*3600
```

### POST /auth/refresh
```
Client envia cookie refresh_token
        │
        ▼
Llegeix raw token de la cookie → token_hash = SHA-256(raw)
        │
        ▼
auth_repository: get_refresh_token_by_hash(auth_db, token_hash)
        │
        ├── No trobat (ja rotat o inexistent):
        │     REUSE DETECTED → auth_repository: delete_all_user_refresh_tokens(auth_db, user_id)
        │     → 401 "Session revoked"
        │     (però no sabem user_id... → retornem 401 sense més context)
        │
        ├── Trobat però expirat → delete + 401
        │
        └── Trobat i vàlid:
              Obté user_id del document
              auth_repository: delete_refresh_token(auth_db, token_hash)  ← invalida l'antic
                      │
                      ▼
              get_user_by_email / get_user_by_id → comprova active=true
                      │
                      ▼
              user_profile_repository: get_profile_by_user_id(app_db, user_id)
                      │
                      ▼
              Genera nou Access Token JWT
              Genera nou Refresh Token (raw + hash)
              auth_repository: create_refresh_token(auth_db, user_id, new_hash, expires_at)
                      │
                      ▼
              Resposta:
                Body: {"access_token": "...", "token_type": "bearer"}
                Cookie: refresh_token=<new_raw>, HttpOnly, ...
```

---

## Fitxers a crear / modificar

### 1. `app/schemas/auth.py` — NOU
```python
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

### 2. `app/repositories/auth_repository.py` — NOU (sc-auth-db)
Funcions pures d'accés a dades. Cap lògica de negoci:
- `get_user_by_email(auth_db, email)` → dict | None
- `get_user_by_id(auth_db, user_id)` → dict | None
- `update_user_password(auth_db, user_id, password_hash)` → None (unset force_reset)
- `create_refresh_token(auth_db, user_id, token_hash, expires_at)` → None
- `get_refresh_token_by_hash(auth_db, token_hash)` → dict | None
- `delete_refresh_token(auth_db, token_hash)` → None
- `delete_all_user_refresh_tokens(auth_db, user_id)` → None (reuse: invalida sessió)

### 3. `app/repositories/user_profile_repository.py` — NOU (sc-app-db)
- `get_profile_by_user_id(app_db, user_id)` → dict | None (retorna team_ids)

### 4. `app/services/auth_service.py` — NOU
Tota la lògica de negoci. Orquestra els dos repositoris:
- `login(email, password, auth_db, app_db)` → tuple[str, str] (access_token, refresh_token_raw)
- `refresh(refresh_token_raw, auth_db, app_db)` → tuple[str, str] (access_token, refresh_token_raw)

Funcions internes (privades, prefix `_`):
- `_create_access_token(user_id, role, team_ids)` → str (JWT HS256)
- `_create_refresh_token_pair()` → tuple[str, str] (raw, hash SHA-256)
- `_verify_password(password, stored_hash, force_reset)` → bool

### 5. `app/routers/auth.py` — NOU
```
POST /auth/login    → crida auth_service.login(), set cookie, retorna TokenResponse
POST /auth/refresh  → llegeix cookie, crida auth_service.refresh(), set cookie, retorna TokenResponse
```
Gestió d'errors: HTTPException 401 per a qualsevol cas d'error d'autenticació.
**Mai exposa detalls interns** (quin camp és incorrecte, si l'usuari existeix, etc.).

### 6. `app/main.py` — MODIFICAR
- `app.include_router(auth.router)` sense cap prefix

---

## Consideracions tècniques

### JWT (Access Token)
- Algorisme: HS256 — `python-jose`
- Payload: `{"sub": str(user_id), "role": "coach", "team_ids": ["..."], "exp": datetime}`
- `sub` és `str(ObjectId)` — ObjectId no és serialitzable per JWT directament
- Clau: `settings.JWT_SECRET` — mai hardcoded

### Refresh Token
- Generat amb `secrets.token_urlsafe(64)` — 64 bytes d'entropia
- Emmagatzemat: SHA-256 del raw token (mai el raw a la BD)
- Cookie: `HttpOnly=True, Secure=True, SameSite="strict", path="/auth"`
  - `path="/auth"` → el navegador ONLY envia la cookie als endpoints `/auth/*`
- Expiració: `settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600` segons

### force_reset
- Si `force_reset=True`: `user['password_hash']` conté el password EN CLAR (provisional, del seed)
- Comparació directa (string == string), sense bcrypt
- Si correcte: `passlib.hash.bcrypt.hash(password)` → update_user_password → elimina `force_reset`
- A partir d'aquí, funciona com qualsevol altre usuari

### Detecció de reutilització (Refresh Token Rotation)
- Si `get_refresh_token_by_hash()` retorna `None` → el token no existeix a la BD
- Pot ser perquè ja va ser rotat → POSSIBLE ATAC → 401 "sessió revocada"
- NOTA: no podem invalidar tota la sessió fàcilment perquè no sabem el `user_id`
  sense el token. Opció: 401 genèric sense invalidació addicional. La invalidació
  completa (amb user_id) requereix emmagatzemar user_id a la cookie o al JWT —
  però el Refresh Token és opac. Solució: si token existent però expirat → delete + 401.
  Si token no trobat → 401 genèric (el sessió ja estava invalidada o és un atac).

### Imports absoluts
`from app.services.auth_service import login` — mai relatius amb `..`

### Errors HTTP
- 401 `{"detail": "Credencials incorrectes"}` — per a login fallat
- 401 `{"detail": "Token invàlid o expirat"}` — per a refresh fallat
- **Mai** indicar si l'email existeix o no (prevenir user enumeration)

---

## Fases d'implementació

1. `app/schemas/auth.py`
2. `app/repositories/auth_repository.py`
3. `app/repositories/user_profile_repository.py`
4. `app/services/auth_service.py`
5. `app/routers/auth.py`
6. `app/main.py` (include router)
7. Actualitzar `CLAUDE.md`
