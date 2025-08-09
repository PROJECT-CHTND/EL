from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


def _clamp01(value: float) -> float:
    if value is None:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


class Hypothesis(BaseModel):
    id: str
    text: str
    slots: List[str] = Field(default_factory=list)
    belief: float = 0.0
    belief_ci: Tuple[float, float] = (0.0, 0.0)
    novelty: float = 0.0
    contradictions: List[str] = Field(default_factory=list)
    last_update: int = 0
    provenance: List[Dict[str, Any]] = Field(default_factory=list)
    action_cost: Dict[str, float] = Field(default_factory=lambda: {"ask": 0.0, "search": 0.0})
    status: Literal["open", "confirmed", "rejected", "stale"] = "open"

    @field_validator("belief", "novelty")
    @classmethod
    def _clamp_float_fields(cls, v: float) -> float:
        return _clamp01(v)

    @field_validator("belief_ci")
    @classmethod
    def _validate_ci(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = (_clamp01(v[0]), _clamp01(v[1]))
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    @field_validator("action_cost")
    @classmethod
    def _clamp_action_cost(cls, v: Dict[str, float]) -> Dict[str, float]:
        ask = _clamp01(v.get("ask", 0.0))
        search = _clamp01(v.get("search", 0.0))
        return {"ask": ask, "search": search}

    @field_validator("status")
    @classmethod
    def _status_allowed(cls, v: str) -> str:
        allowed = {"open", "confirmed", "rejected", "stale"}
        if v not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return v


class Evidence(BaseModel):
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relations: List[Dict[str, Any]] = Field(default_factory=list)
    supports: List[Dict[str, Any]] = Field(default_factory=list)
    feature_vector: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("feature_vector")
    @classmethod
    def _clamp_feature_vector(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("cosine", "source_trust", "recency"):
            if key in v and isinstance(v[key], (int, float)):
                v[key] = _clamp01(float(v[key]))
        return v


class StrategistAction(BaseModel):
    target_hypothesis: str
    action: Literal["ask", "search", "none"]
    question: Optional[str] = None
    expected_gain: float = 0.0
    estimated_cost: float = 0.0
    stop_rule_hit: bool = False

    @field_validator("expected_gain", "estimated_cost")
    @classmethod
    def _clamp_scores(cls, v: float) -> float:
        return _clamp01(v)


