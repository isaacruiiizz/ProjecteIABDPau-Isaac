from pydantic import BaseModel, Field


class LabelingUploadResponse(BaseModel):
    session_id: str
    minio_key: str
    status: str = "queued"


class LabelingFrameResponse(BaseModel):
    frame_url: str
    total_frames: int


class LabelingStartRequest(BaseModel):
    video_key: str
    jersey_own_color_hsv: str | None = Field(
        default=None,
        description="Color HSV dominant de la samarreta pròpia en format 'H,S,V' (ex: '120,80,150'). "
                    "Si és null, sc-inference-worker usa la variable d'entorn JERSEY_OWN_COLOR_HSV.",
    )
    jersey_color_threshold: int = Field(
        default=30,
        ge=1,
        le=180,
        description="Distància màxima HSV per classificar un jugador com a 'player_own'. Rang recomanat: 10–60.",
    )


class LabelingStartResponse(BaseModel):
    status: str
    frames_queued: int


class LsStatsResponse(BaseModel):
    total_tasks: int
    annotated_tasks: int
    predicted_tasks: int = 0


class LsTasksClearResponse(BaseModel):
    deleted: int
    deleted_frames: int = 0
