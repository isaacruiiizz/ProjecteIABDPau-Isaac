# Endpoints — SmartChrono IP

Registre de tots els endpoints implementats. Actualitzar cada cop que s'afegeix o modifica un endpoint.

**Format:** `MÈTODE /path` — descripció breu + autenticació + servei

---

## Sistema

| Mètode | Path | Auth | Servei | Descripció |
|---|---|---|---|---|
| `GET` | `/health` | Cap | `sc-api-gateway` | Healthcheck Docker. Retorna `{"status": "ok"}` |

---

## Autenticació

Els endpoints d'autenticació **no porten prefix `/api/v1/`**.

| Mètode | Path | Auth | Servei | Descripció |
|---|---|---|---|---|
| `POST` | `/auth/login` | Cap | `sc-api-gateway` | Login amb email + password. Retorna `access_token` (body) + `refresh_token` (cookie HttpOnly) |
| `POST` | `/auth/refresh` | Cookie `refresh_token` (HttpOnly) | `sc-api-gateway` | Refresh Token Rotation. Invalida el token anterior, emet nous tokens |

### `POST /auth/login`

**Request body:**
```json
{ "email": "usuari@club.cat", "password": "..." }
```

**Response 200:**
```json
{ "access_token": "eyJ...", "token_type": "bearer" }
```
+ `Set-Cookie: refresh_token=<raw>; HttpOnly; Secure; SameSite=Strict; Path=/auth; Max-Age=604800`

**Errors:**
- `401` `{"detail": "Credencials incorrectes"}` — email o password incorrectes, usuari inactiu

---

### `POST /auth/refresh`

**Cookie requerida:** `refresh_token` (enviada automàticament pel navegador amb `withCredentials: true`)

**Response 200:**
```json
{ "access_token": "eyJ...", "token_type": "bearer" }
```
+ nova cookie `refresh_token` (token anterior invalidat)

**Errors:**
- `401` `{"detail": "Token invàlid o expirat"}` — token no trobat, expirat o reutilitzat (possible atac → sessió revocada)

---

## API de Negoci

*Pendent d'implementació — sprints posteriors.*

Tots els endpoints de negoci seguiran el prefix `/api/v1/{recurs}` en plural i minúscules.
Requeriran `Authorization: Bearer <access_token>` a la capçalera.
