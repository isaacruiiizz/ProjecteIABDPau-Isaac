from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    REDIS_HOST: str = "sc-redis"
    REDIS_PORT: int = 6379
    REDIS_QUEUE_VIDEO: str = "video_to_process"
    REDIS_QUEUE_FRAMES: str = "task_frames"

    # MinIO
    MINIO_ENDPOINT: str = "sc-object-storage:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET_RAW: str = "raw-videos"
    MINIO_BUCKET_PENDING: str = "pending-frames"
    MINIO_BUCKET_PROCESSED_FRAMES: str = "processed-frames"
    MINIO_BUCKET_OUTPUT: str = "processed-videos"
    MINIO_BUCKET_LABELING_VIDEOS: str = "labeling-videos"
    MINIO_BUCKET_LABELING_FRAMES: str = "labeling-frames"

    # FFmpeg
    VIDEO_FPS_DEFAULT: int = 25

    # Sentry
    SENTRY_DSN: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
