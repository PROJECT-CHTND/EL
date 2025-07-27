from __future__ import annotations

from pydantic import BaseModel, Field


class Question(BaseModel):
    """Representation of a generated question aimed at filling a slot."""

    slot_name: str = Field(..., description="Target slot name")
    text: str = Field(..., description="Question text")
    specificity: float | None = Field(None, ge=0.0, le=1.0)
    tacit_power: float | None = Field(None, ge=0.0, le=1.0) 