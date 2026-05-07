from pydantic import BaseModel


class MatchCreateResponse(BaseModel):
    match_id: str
    status: str
