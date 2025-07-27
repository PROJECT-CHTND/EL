from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Knowledge graph entity node."""

    id: str = Field(..., description="Unique identifier (temporary or permanent)")
    label: str = Field(..., description="Human-readable label")
    type: Optional[str] = Field(None, description="Entity type/category")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Extraction confidence 0–1")


class Relation(BaseModel):
    """Knowledge graph edge between two entities."""

    source: str = Field(..., description="Source entity ID")
    target: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relation type")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class KGPayload(BaseModel):
    """Container for extracted KG fragment."""

    entities: List[Entity]
    relations: List[Relation]

    def with_confidence(self, logprobs: dict) -> "KGPayload":
        """Placeholder: attach confidence using logprobs (not implemented)."""
        # TODO: implement real confidence calculation
        return self