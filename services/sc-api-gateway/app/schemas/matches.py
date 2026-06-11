from datetime import datetime
from pydantic import BaseModel, field_validator, model_validator


class MatchCreateResponse(BaseModel):
    match_id: str
    status: str


class RoiPoint(BaseModel):
    x: float
    y: float


class MatchConfigRequest(BaseModel):
    roi_polygon: list[RoiPoint]
    start_seconds: float
    end_seconds: float

    @field_validator('roi_polygon')
    @classmethod
    def min_3_points(cls, v):
        if len(v) < 3:
            raise ValueError('roi_polygon mínim 3 punts')
        return v

    @model_validator(mode='after')
    def end_after_start(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError('end_seconds ha de ser major que start_seconds')
        return self


class MatchConfigResponse(BaseModel):
    match_id: str
    roi_polygon: list[RoiPoint]
    start_seconds: float
    end_seconds: float


class MatchListItem(BaseModel):
    match_id: str
    title: str
    status: str
    created_at: datetime
    start_seconds: float | None = None
    end_seconds: float | None = None
    has_roi: bool


class ProcessMatchResponse(BaseModel):
    match_id: str
    status: str


class MatchDetail(BaseModel):
    match_id: str
    title: str
    status: str
    created_at: datetime
    start_seconds: float | None = None
    end_seconds: float | None = None
    download_url: str | None = None
    ai_stats: dict | None = None
    ai_report: str | None = None
