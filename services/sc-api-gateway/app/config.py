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

    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False

    # Sentry
    SENTRY_DSN: str = ""


settings = Settings()
