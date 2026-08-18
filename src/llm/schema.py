import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator, confloat


class TriageInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text", mode="before")
    def strip_and_not_blank(cls, v):
        if v is None:
            raise ValueError("text is required")
        if isinstance(v, str):
            s = v.strip()
            if len(s) == 0:
                raise ValueError("text must not be empty or whitespace")
            return s
        return v


class TriageOutput(BaseModel):
    category: Literal["billing", "bug", "feature", "other"]
    urgency: Literal["low", "normal", "high"]
    confidence: confloat(ge=0.0, le=1.0)
    reason: str

    model_config = {"extra": "forbid"}
