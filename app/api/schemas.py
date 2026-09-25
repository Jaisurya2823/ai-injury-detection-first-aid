from __future__ import annotations
from pydantic import BaseModel, Field

class ContextRequest(BaseModel):
    mechanism: str | None=None
    time_since_injury: str | None=None
    bleeding_status: str | None=None
    pain_level: str | None=None
    movement_limitation: str | None=None
    red_flags: list[str]=Field(default_factory=list)
