"""Schemas for EL Evaluation Framework."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalDomain(str, Enum):
    """Evaluation domain types matching test data structure."""

    POSTMORTEM = "postmortem"
    MANUAL = "manual"  # SOP
    RECIPE = "recepi"  # Note: typo in data directory name
    DAILY_WORK = "daily_work"


class TestCase(BaseModel):
    """A single test case loaded from evaluation data."""

    case_id: str = Field(..., description="Unique identifier for the test case")
    domain: EvalDomain = Field(..., description="Domain category")
    initial_note: str = Field(..., description="User's initial input (from user_initial_note.txt)")
    gold_slots: dict[str, Any] = Field(..., description="Expected output slots (from gold_slots.json)")
    
    # Optional metadata
    system_prompt: str | None = Field(default=None, description="System prompt if available")
    reference_dialogue: str | None = Field(default=None, description="Reference dialogue if available")

    @property
    def slot_count(self) -> int:
        """Count the number of top-level slots in gold_slots."""
        return len(self.gold_slots)

    def flatten_slots(self) -> dict[str, str]:
        """Flatten nested gold_slots into dot-notation keys with string values.
        
        Example:
            {"impact": {"users": "100人"}} -> {"impact.users": "100人"}
        """
        result: dict[str, str] = {}
        
        def _flatten(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    _flatten(value, new_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_key = f"{prefix}[{i}]"
                    _flatten(item, new_key)
            else:
                result[prefix] = str(obj)
        
        _flatten(self.gold_slots)
        return result


class ConversationLog(BaseModel):
    """Log of a single conversation turn during evaluation."""

    turn_number: int
    user_message: str
    assistant_response: str
    insights_saved: list[dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SlotMatch(BaseModel):
    """A match between an extracted insight and a gold slot."""

    slot_key: str = Field(..., description="The gold slot key (e.g., 'impact.users')")
    slot_value: str = Field(..., description="The expected value from gold_slots")
    insight_content: str = Field(..., description="The extracted insight content")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity")
    is_match: bool = Field(..., description="Whether similarity exceeds threshold")


class QuestionQuality(BaseModel):
    """Quality assessment of a single question from the agent."""

    question: str = Field(..., description="The question text")
    turn_number: int = Field(..., description="Which turn this question appeared in")
    
    # Quality dimensions (0.0 - 1.0)
    empathy_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How empathetic and understanding the question is"
    )
    insight_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How insightful and thought-provoking the question is"
    )
    specificity_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How specific and concrete the question is"
    )
    flow_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How well the question flows from previous context"
    )
    
    # Overall score
    overall_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted average of all quality dimensions"
    )
    
    # Reasoning from the judge
    reasoning: str = Field(default="", description="LLM judge's reasoning")


class QuestionQualitySummary(BaseModel):
    """Summary of question quality across a conversation."""

    avg_empathy: float = Field(..., ge=0.0, le=1.0)
    avg_insight: float = Field(..., ge=0.0, le=1.0)
    avg_specificity: float = Field(..., ge=0.0, le=1.0)
    avg_flow: float = Field(..., ge=0.0, le=1.0)
    avg_overall: float = Field(..., ge=0.0, le=1.0)
    
    question_count: int = Field(..., ge=0)
    questions: list[QuestionQuality] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Result of evaluating a single test case."""

    case_id: str
    domain: EvalDomain
    
    # Core metrics
    slot_coverage: float = Field(..., ge=0.0, le=1.0, description="Proportion of gold slots covered")
    turn_count: int = Field(..., ge=0, description="Number of conversation turns")
    turn_efficiency: float = Field(..., ge=0.0, le=1.0, description="Efficiency score based on turns")
    domain_accuracy: bool = Field(..., description="Whether domain was correctly detected")
    
    # Question quality metrics (Phase 2)
    question_quality: QuestionQualitySummary | None = Field(
        default=None, description="Question quality assessment from LLM-as-Judge"
    )
    
    # Detailed data
    matched_slots: list[SlotMatch] = Field(default_factory=list)
    unmatched_slots: list[str] = Field(default_factory=list)
    conversation_log: list[ConversationLog] = Field(default_factory=list)
    insights_saved: list[dict[str, Any]] = Field(default_factory=list)
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    @property
    def total_slots(self) -> int:
        """Total number of gold slots."""
        return len(self.matched_slots) + len(self.unmatched_slots)


class EvalSummary(BaseModel):
    """Summary of evaluation across multiple test cases."""

    total_cases: int
    completed_cases: int
    failed_cases: int
    
    # Aggregate metrics
    avg_slot_coverage: float
    avg_turn_count: float
    avg_turn_efficiency: float
    domain_accuracy_rate: float
    
    # Question quality metrics (Phase 2)
    avg_question_quality: float | None = Field(
        default=None, description="Average overall question quality score"
    )
    avg_empathy: float | None = Field(default=None)
    avg_insight: float | None = Field(default=None)
    avg_specificity: float | None = Field(default=None)
    
    # Per-domain breakdown
    by_domain: dict[str, dict[str, float]] = Field(default_factory=dict)
    
    # Individual results
    results: list[EvalResult] = Field(default_factory=list)
    
    # Timing
    total_duration_seconds: float
    started_at: datetime
    completed_at: datetime
