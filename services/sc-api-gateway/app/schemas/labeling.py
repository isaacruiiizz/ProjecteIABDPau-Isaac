from pydantic import BaseModel


class LabelingUploadResponse(BaseModel):
    session_id: str
    minio_key: str
    status: str = "queued"
