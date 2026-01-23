"""Pydantic schemas for EL Core."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Conversation domain types."""

    DAILY_WORK = "daily_work"
    RECIPE = "recipe"
    POSTMORTEM = "postmortem"
    CREATIVE = "creative"
    GENERAL = "general"


class Role(str, Enum):
    """Message role types."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class Insight(BaseModel):
    """An insight extracted from the conversation and saved to knowledge graph."""

    subject: str = Field(..., description="Subject entity of the insight")
    predicate: str = Field(..., description="Relationship type (e.g., has_insight, learned_that)")
    object: str = Field(..., description="Object/content of the insight")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    domain: Domain = Field(default=Domain.GENERAL, description="Detected domain")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class FactStatus(str, Enum):
    """Status of a fact in the knowledge graph."""

    ACTIVE = "active"          # Currently valid fact
    SUPERSEDED = "superseded"  # Replaced by newer version
    DISPUTED = "disputed"      # Contradiction detected, needs resolution


class KnowledgeItem(BaseModel):
    """A knowledge item retrieved from the knowledge graph."""

    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    domain: Domain
    created_at: datetime
    status: FactStatus = FactStatus.ACTIVE

    model_config = {"frozen": True}


class FactVersion(BaseModel):
    """A version of a fact for history tracking."""

    id: str = Field(..., description="Version ID")
    fact_id: str = Field(..., description="Parent fact ID")
    value: str = Field(..., description="The value at this version")
    source: str = Field(default="会話", description="Source of this version")
    session_id: str | None = Field(default=None, description="Session where this was recorded")
    valid_from: datetime = Field(default_factory=datetime.now)
    valid_until: datetime | None = Field(default=None, description="None if current version")
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class FactWithHistory(BaseModel):
    """A fact with its version history."""

    id: str
    subject: str
    predicate: str
    current_value: str
    status: FactStatus
    domain: Domain
    confidence: float
    versions: list[FactVersion] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"ser_json_always": True}


class ConversationTurn(BaseModel):
    """A single turn in the conversation (user input + assistant response)."""

    user_message: str
    assistant_response: str
    insights_saved: list[Insight] = Field(default_factory=list)
    knowledge_used: list[KnowledgeItem] = Field(default_factory=list)
    detected_domain: Domain = Domain.GENERAL
    timestamp: datetime = Field(default_factory=datetime.now)


class Session(BaseModel):
    """A conversation session."""

    id: str
    user_id: str
    topic: str
    language: str = Field(default="Japanese")
    domain: Domain = Field(default=Domain.GENERAL)
    turns: list[ConversationTurn] = Field(default_factory=list)
    prior_context: str = Field(default="", description="Pre-fetched context from knowledge graph")
    prior_knowledge: list[KnowledgeItem] = Field(default_factory=list, description="Related knowledge items")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn to the session."""
        self.turns.append(turn)
        self.updated_at = datetime.now()
        # Update domain if detected differently
        if turn.detected_domain != Domain.GENERAL:
            self.domain = turn.detected_domain

    @property
    def message_history(self) -> list[dict[str, str]]:
        """Get message history in OpenAI format."""
        messages: list[dict[str, str]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_response})
        return messages

    @property
    def insights_count(self) -> int:
        """Get total number of insights saved in this session."""
        return sum(len(turn.insights_saved) for turn in self.turns)


class SessionMetadata(BaseModel):
    """Persisted session metadata for Neo4j storage."""

    id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    topic: str = Field(..., description="Conversation topic")
    domain: Domain = Field(default=Domain.GENERAL, description="Detected domain")
    turn_count: int = Field(default=0, description="Number of conversation turns")
    insights_count: int = Field(default=0, description="Number of insights saved")
    status: str = Field(default="active", description="Session status: active, ended, archived")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class SessionSummary(BaseModel):
    """Summary of a conversation session for long-term storage."""

    id: str = Field(..., description="Summary ID")
    session_id: str = Field(..., description="Associated session ID")
    
    # Natural language summary
    content: str = Field(..., description="2-3 sentence summary of the conversation")
    
    # Structured key points
    key_points: list[str] = Field(default_factory=list, description="Important facts extracted")
    
    # Topics and entities
    topics: list[str] = Field(default_factory=list, description="Main topics discussed")
    entities_mentioned: list[str] = Field(default_factory=list, description="People, projects, etc.")
    
    # Metadata
    turn_range: tuple[int, int] = Field(default=(0, 0), description="Range of turns summarized")
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"ser_json_always": True}


# Tool-related schemas for OpenAI Function Calling


class SearchKnowledgeGraphParams(BaseModel):
    """Parameters for search_knowledge_graph tool."""

    query: str = Field(..., description="Search query (keywords or concepts)")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of results")


class SaveInsightParams(BaseModel):
    """Parameters for save_insight tool."""

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship type")
    object: str = Field(..., description="Object/content of the insight")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    domain: Domain = Field(default=Domain.GENERAL, description="Detected domain")


class ToolCall(BaseModel):
    """A tool call made by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ConsistencyIssueKind(str, Enum):
    """Type of consistency issue."""

    CONTRADICTION = "contradiction"  # Direct conflict with past information
    CHANGE = "change"  # Information has changed/updated


class ConsistencyIssue(BaseModel):
    """A detected consistency issue between current and past information."""

    kind: ConsistencyIssueKind = Field(..., description="Type of issue")
    title: str = Field(..., description="Brief title of the issue")
    fact_id: str | None = Field(default=None, description="ID of the related fact in KG")
    previous_text: str = Field(..., description="What was said before")
    previous_source: str = Field(default="", description="Source/context of previous info")
    current_text: str = Field(..., description="What is being said now")
    current_source: str = Field(default="現在の会話", description="Source of current info")
    suggested_question: str = Field(default="", description="Question to clarify")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in detection")

    model_config = {"ser_json_always": True}


class AgentResponse(BaseModel):
    """Response from the EL Agent."""

    message: str = Field(..., description="The assistant's response message")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    insights_saved: list[Insight] = Field(default_factory=list)
    knowledge_used: list[KnowledgeItem] = Field(default_factory=list)
    detected_domain: Domain = Field(default=Domain.GENERAL)
    consistency_issues: list[ConsistencyIssue] = Field(
        default_factory=list,
        description="Detected consistency issues with past knowledge"
    )

    model_config = {"ser_json_always": True}
