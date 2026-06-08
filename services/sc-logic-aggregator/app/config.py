from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Redis
    REDIS_HOST: str = "sc-redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_QUEUE_RESULTS: str = "detected_frames_results"

    # MinIO
    MINIO_ENDPOINT: str = "sc-object-storage:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET_RAW: str = "raw-videos"
    MINIO_BUCKET_PENDING: str = "pending-frames"
    MINIO_BUCKET_OUTPUT: str = "processed-videos"

    # MongoDB
    MONGO_APP_URI: str

    # Sentry
    SENTRY_DSN: str = ""


settings = Settings()
