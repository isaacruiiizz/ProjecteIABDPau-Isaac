# Pla d'Implementació — PJM-18
# [backend] Implementar GET /health i estructura base de FastAPI

**Estat:** Completat ✓ (2026-03-22)
**Ticket:** PJM-18
**Data:** 2026-03-22
**Etiqueta commit:** `feat(backend): implementar GET /health i estructura base de sc-api-gateway [PJM-18]`

---

## Context

`sc-api-gateway` és el punt d'entrada REST únic del sistema. Aquest ticket estableix
l'esquelet obligatori de 3 capes, la configuració base i el primer endpoint funcional
(`GET /health`) que els healthchecks de Docker necessiten.

---

## Estat actual

| Fitxer | Estat |
|---|---|
| `services/sc-api-gateway/Dockerfile` | Ja existeix (PJM-55) |
| `services/sc-api-gateway/.env.example` | Ja existeix (PJM-15) |
| `services/sc-api-gateway/requirements.txt` | Placeholder buit |
| `services/sc-api-gateway/app/__init__.py` | Buit |
| Resta de fitxers (`main.py`, `config.py`, etc.) | No existeixen |

---

## Fitxers a crear / modificar

### 1. `requirements.txt` — MODIFICAR (ara és placeholder)

```
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
motor>=3.7.0
pydantic-settings>=2.8.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
redis>=5.2.0
boto3>=1.37.0
sentry-sdk>=2.22.0
python-multipart>=0.0.20
```

Justificació de cada dependència:
- `fastapi` + `uvicorn[standard]` — framework i servidor ASGI
- `motor` — driver MongoDB async (dos clients: auth_db + app_db)
- `pydantic-settings` — lectura de variables d'entorn via `BaseSettings`
- `python-jose[cryptography]` — JWT (signin HS256)
- `passlib[bcrypt]` — hash de contrasenyes (cost factor 12)
- `redis` — cues de treball (`video_to_process`, etc.)
- `boto3` — client S3 compatible MinIO
- `sentry-sdk` — reporting d'errors (ERROR/CRITICAL)
- `python-multipart` — upload de fitxers (vídeos)

### 2. `app/config.py` — NOU

Pydantic `BaseSettings` llegint totes les variables de `.env`. Instància singleton
`settings` importable des de qualsevol mòdul.

Variables llegides (alineades amb `.env.example`):
- `API_HOST`, `API_PORT`, `API_ENV`
- `JWT_SECRET`, `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`, `JWT_REFRESH_TOKEN_EXPIRE_DAYS`
- `INTERNAL_API_KEY`
- `MONGO_AUTH_URI`, `MONGO_APP_URI`
- `REDIS_HOST`, `REDIS_PORT`
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_USE_SSL`
- `SENTRY_DSN`

### 3. `app/dependencies.py` — NOU

Dos clients Motor independents exposats com a variables de mòdul, inicialitzats
al `lifespan` de FastAPI i tancats en cleanup:

```python
auth_client: AsyncIOMotorClient | None = None  # → sc-auth-db
app_client:  AsyncIOMotorClient | None = None  # → sc-app-db

def get_auth_db() -> AsyncIOMotorDatabase: ...
def get_app_db()  -> AsyncIOMotorDatabase: ...
```

Els routers usen `Depends(get_auth_db)` / `Depends(get_app_db)` — mai accedeixen
als clients directament.

### 4. `app/main.py` — NOU

Entrypoint. Ordre d'inicialització obligatori:
1. `setup_logging("sc-api-gateway", settings.SENTRY_DSN)` — primer de tot
2. Lifespan: inicialitza `auth_client` + `app_client` Motor (startup) i els tanca (shutdown)
3. `FastAPI(lifespan=lifespan)`
4. Middleware CORS — **només** si `API_ENV == "development"` (origen `http://localhost:3000`)
5. `app.include_router(health.router)` — sense prefix

Implementació de `setup_logging()` extreta literalment del punt 1 de `docs/specs/03-infraestructura.md`.

### 5. `app/routers/health.py` — NOU

```python
@router.get("/health")
async def health_check():
    return {"status": "ok"}
```

Sense prefix `/api/v1/`. Retorna 200 amb `{"status": "ok"}`. Usat pel `healthcheck`
del docker-compose: `curl -f http://localhost:8000/health`.

### 6. `__init__.py` de totes les capes — NOU (buits)

- `app/routers/__init__.py`
- `app/schemas/__init__.py`
- `app/repositories/__init__.py`
- `app/services/__init__.py`

Marquen cada directori com a paquet Python. Buits per ara — cada ticket posterior
els omplirà.

---

## Fases d'implementació

### Fase 1 — `requirements.txt`
Substituir el placeholder amb les dependències reals.

### Fase 2 — `app/config.py`
Pydantic Settings amb totes les variables del `.env`.

### Fase 3 — `app/dependencies.py`
Dos clients Motor + funcions `get_auth_db()` / `get_app_db()`.

### Fase 4 — `app/main.py`
Entrypoint complet: `setup_logging`, lifespan, CORS condicional, router health.

### Fase 5 — `app/routers/health.py` + `__init__.py` de les 4 capes
Endpoint `GET /health` i estructura de directoris.

### Fase 6 — Actualitzar CLAUDE.md
Reflectir l'estructura completa de `sc-api-gateway` a l'arbre de directoris.

---

## Consideracions tècniques

- **Motor lifespan:** els clients MongoDB s'inicialitzen una sola vegada a l'arrencada i es
  tanquen en shutdown. Evita connexions repetides i permet que FastAPI gestioni el cicle de vida.
- **`get_default_database()`:** Motor parseja la URI (`mongodb://host:port/nom-db`) i retorna
  la base de dades especificada al path. `MONGO_AUTH_URI` acaba en `/sc-auth-db` →
  `get_default_database()` retorna `sc-auth-db` sense hardcoding.
- **CORS:** el middleware s'afegeix condicionalment. En producció (`API_ENV=production`)
  frontend i API son al mateix origen, no cal CORS.
- **`setup_logging` primer:** Sentry ha d'estar inicialitzat abans de qualsevol altre codi
  per capturar errors d'arrencada. Per això és la primera crida a `main.py`.
- **Imports absoluts:** `from app.config import settings` — mai relatius amb `..`.
- **No hi ha lògica de negoci** en aquest ticket — els repositories, services i schemas
  queden com a `__init__.py` buits per als tickets posteriors.
