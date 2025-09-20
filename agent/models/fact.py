from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


ImpactLevel = Literal["normal", "critical"]
Decision = Literal["approve", "reject", "pending"]


class FactIn(BaseModel):
    """Input schema for submitting a fact for approval workflow."""

    id: Optional[str] = Field(None, description="Client-provided identifier; UUID will be assigned if missing")
    text: str = Field(..., description="Fact statement text")
    belief: float = Field(..., ge=0.0, le=1.0, description="Belief/confidence score 0–1")
    impact: ImpactLevel = Field(..., description='Impact level, one of ["normal", "critical"]')


class Fact(BaseModel):
    """Persisted Fact with status."""

    id: str
    text: str
    belief: float
    impact: ImpactLevel
    status: Literal["pending", "confirmed", "rejected"]


class Approval(BaseModel):
    id: str
    fact_id: str
    approver: Optional[str] = None
    ts: str  # ISO8601
    decision: Decision


class ApproveRequest(BaseModel):
    fact_id: str
    decision: Literal["approve", "reject"]


