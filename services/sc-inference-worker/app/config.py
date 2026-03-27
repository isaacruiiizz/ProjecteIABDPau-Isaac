from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Redis
    REDIS_HOST: str = "sc-redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_QUEUE_FRAMES: str = "task_frames"
    REDIS_QUEUE_RESULTS: str = "detected_frames_results"
    REDIS_QUEUE_MODEL_PROMOTED: str = "model_promoted"
    REDIS_QUEUE_LABELING: str = "labeling_frames_to_infer"

    # MinIO
    MINIO_ENDPOINT: str = "sc-object-storage:9000"
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_USE_SSL: bool = False
    MINIO_BUCKET_PENDING: str = "pending-frames"
    MINIO_BUCKET_FEEDBACK: str = "feedback-data"
    MINIO_BUCKET_MODELS: str = "models"
    MINIO_BUCKET_LABELING_FRAMES: str = "labeling-frames"
    MINIO_MODEL_KEY: str = "yolo/base/yolov8n.pt"  # clau del model base dins el bucket models

    # IA — producció
    YOLO_MODEL_PATH: str = "yolo/weights/v1.pt"
    CNN_MODEL_PATH: str = "cnn/weights/v1.keras"
    INFERENCE_CONFIDENCE_THRESHOLD: float = 0.6
    PREFETCH_BUFFER_SIZE: int = 8

    # IA — pre-anotació etiquetatge
    INFERENCE_LABELING_CONFIDENCE: float = 0.3
    JERSEY_OWN_COLOR_HSV: str = ""   # ex: "120,80,150". Buit → fallback player_own
    JERSEY_COLOR_THRESHOLD: int = 30

    # Label Studio
    LABEL_STUDIO_URL: str = "http://sc-label-studio:8081"
    LABEL_STUDIO_API_TOKEN: str = ""
    LABEL_STUDIO_PROJECT_ID: int = 1

    # Sentry
    SENTRY_DSN: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
