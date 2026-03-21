from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

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
