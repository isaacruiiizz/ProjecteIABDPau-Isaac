from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servidor
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_ENV: str = "development"

    # JWT
    JWT_SECRET: str
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Internal API Key
    INTERNAL_API_KEY: str

    # MongoDB
    MONGO_AUTH_URI: str
    MONGO_APP_URI: str

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False
    # URL pública de MinIO accessible des del navegador (ex: http://localhost:9000).
    # Si és buit, s'usa MINIO_ENDPOINT (vàlid per a entorns sense accés extern).
    MINIO_PUBLIC_URL: str = ""

    # Label Studio
    LABEL_STUDIO_URL: str = "http://sc-label-studio:8081"
    LABEL_STUDIO_API_TOKEN: str = ""
    LABEL_STUDIO_PROJECT_ID: int = 1

    # CORS — llista separada per comes dels orígens permesos
    # Ex: "http://localhost:3000,http://smartchrono.asixsapa.cat:3000"
    CORS_ORIGINS: str = "http://localhost:3000"

    # Cookie del refresh token: True requereix HTTPS. Posar False si HTTP.
    COOKIE_SECURE: bool = False

    # Ollama
    OLLAMA_BASE_URL: str = "http://sc-ollama:11434"

    # Sentry
    SENTRY_DSN: str = ""


settings = Settings()
