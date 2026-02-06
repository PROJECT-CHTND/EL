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


class DateType(str, Enum):
    """Type of date information."""

    EXACT = "exact"  # Exact date (e.g., 2024-05-01)
    APPROXIMATE = "approximate"  # Approximate date (e.g., around May)
    RANGE = "range"  # Date range (e.g., May 1 - May 15)
    UNKNOWN = "unknown"  # No date information


class Message(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {"frozen": True}


class Insight(BaseModel):
    """An insight extracted from the conversation and saved to knowledge graph."""

    id: str | None = Field(default=None, description="Unique identifier (set after saving)")
    subject: str = Field(..., description="Subject entity of the insight")
    predicate: str = Field(..., description="Relationship type (e.g., has_insight, learned_that)")
    object: str = Field(..., description="Object/content of the insight")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    domain: Domain = Field(default=Domain.GENERAL, description="Detected domain")
    timestamp: datetime = Field(default_factory=datetime.now)
    # Temporal information
    event_date: datetime | None = Field(default=None, description="Date when the event occurred")
    event_date_end: datetime | None = Field(default=None, description="End date for date ranges")
    date_type: DateType = Field(default=DateType.UNKNOWN, description="Type of date information")

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
    # Temporal information
    event_date: datetime | None = None
    event_date_end: datetime | None = None
    date_type: DateType = DateType.UNKNOWN

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
    questions_asked: list[str] = Field(default_factory=list, description="IDs of questions presented in this turn")
    questions_answered: list[str] = Field(default_factory=list, description="IDs of questions answered in this turn")
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
    referenced_dates: list[datetime] = Field(default_factory=list, description="Dates referenced in this session")
    pending_questions: list[PendingQuestion] = Field(default_factory=list, description="Pending questions for the user")
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


class QuestionKind(str, Enum):
    """Type of question to ask the user."""

    CONTRADICTION = "contradiction"  # 矛盾の確認
    CHANGE = "change"               # 変更の確認
    MISSING = "missing"             # 不足情報の補完
    CLARIFICATION = "clarification" # 曖昧な情報の明確化


class QuestionStatus(str, Enum):
    """Status of a pending question."""

    PENDING = "pending"      # 未回答
    ANSWERED = "answered"    # 回答済み
    SKIPPED = "skipped"      # スキップ
    EXPIRED = "expired"      # 期限切れ


class PendingQuestion(BaseModel):
    """A question pending user response, tracked across turns."""

    id: str = Field(..., description="Unique identifier for the question")
    kind: QuestionKind = Field(..., description="Type of question")
    question: str = Field(..., description="The question text to ask the user")
    context: str = Field(default="", description="Background context for the question")
    related_fact_id: str | None = Field(default=None, description="ID of the related fact in KG")
    related_entity: str | None = Field(default=None, description="Related entity name")
    priority: int = Field(default=0, description="Priority (higher = more important)")
    status: QuestionStatus = Field(default=QuestionStatus.PENDING, description="Current status")
    answer: str | None = Field(default=None, description="User's answer text")
    session_id: str | None = Field(default=None, description="Session where this was created")
    asked_at: datetime | None = Field(default=None, description="When the question was presented")
    answered_at: datetime | None = Field(default=None, description="When the user answered")
    created_at: datetime = Field(default_factory=datetime.now, description="When the question was created")

    model_config = {"ser_json_always": True}


class ConsistencyIssueKind(str, Enum):
    """Type of consistency issue."""

    CONTRADICTION = "contradiction"  # Direct conflict with past information
    CHANGE = "change"  # Information has changed/updated


class ConsistencyIssueStatus(str, Enum):
    """Status of a consistency issue."""

    UNRESOLVED = "unresolved"  # Not yet resolved
    RESOLVED = "resolved"  # Resolved by user action
    IGNORED = "ignored"  # User chose to ignore


class ConsistencyIssue(BaseModel):
    """A detected consistency issue between current and past information."""

    id: str | None = Field(default=None, description="Unique identifier for the issue")
    kind: ConsistencyIssueKind = Field(..., description="Type of issue")
    status: ConsistencyIssueStatus = Field(
        default=ConsistencyIssueStatus.UNRESOLVED,
        description="Current resolution status"
    )
    title: str = Field(..., description="Brief title of the issue")
    fact_id: str | None = Field(default=None, description="ID of the related fact in KG")
    previous_text: str = Field(..., description="What was said before")
    previous_source: str = Field(default="", description="Source/context of previous info")
    current_text: str = Field(..., description="What is being said now")
    current_source: str = Field(default="現在の会話", description="Source of current info")
    suggested_question: str = Field(default="", description="Question to clarify")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in detection")
    # Resolution tracking
    resolution: str | None = Field(default=None, description="Resolution action: accept_current, keep_previous, ignore")
    resolved_at: datetime | None = Field(default=None, description="When the issue was resolved")
    session_id: str | None = Field(default=None, description="Session where this was detected")
    created_at: datetime = Field(default_factory=datetime.now, description="When the issue was detected")

    model_config = {"ser_json_always": True}


class AgentResponse(BaseModel):
    """Response from Eager Learner (EL)."""

    message: str = Field(..., description="The assistant's response message")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    insights_saved: list[Insight] = Field(default_factory=list)
    knowledge_used: list[KnowledgeItem] = Field(default_factory=list)
    detected_domain: Domain = Field(default=Domain.GENERAL)
    consistency_issues: list[ConsistencyIssue] = Field(
        default_factory=list,
        description="Detected consistency issues with past knowledge"
    )
    pending_questions: list[PendingQuestion] = Field(
        default_factory=list,
        description="Pending questions for the user (includes missing info, contradictions, etc.)"
    )
    questions_answered: list[str] = Field(
        default_factory=list,
        description="IDs of questions answered in this turn"
    )

    model_config = {"ser_json_always": True}


# Document and Knowledge Aggregation schemas


class DocumentStatus(str, Enum):
    """Status of document processing."""

    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    """An uploaded document with extracted knowledge."""

    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    size_bytes: int = Field(..., description="File size in bytes")
    
    # Extracted content
    extracted_summary: str = Field(default="", description="LLM-generated summary")
    extracted_facts_count: int = Field(default=0, description="Number of facts extracted")
    raw_content_preview: str = Field(default="", description="First 500 chars of raw content")
    
    # Categorization
    topics: list[str] = Field(default_factory=list, description="Detected topics")
    entities: list[str] = Field(default_factory=list, description="Mentioned entities")
    domain: Domain = Field(default=Domain.GENERAL, description="Detected domain")
    
    # Metadata
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADING)
    error_message: str | None = Field(default=None, description="Error if failed")
    page_count: int = Field(default=1, description="Number of pages/sheets")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    processed_at: datetime | None = Field(default=None)

    model_config = {"ser_json_always": True}


class DocumentChunk(BaseModel):
    """A chunk of a document, typically split by date or section.
    
    Stores the original content for accurate reference during search.
    """

    id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Original content of the chunk (preserved exactly)")
    chunk_index: int = Field(..., description="Position of chunk within document (0-based)")
    
    # Date information for the chunk
    chunk_date: datetime | None = Field(default=None, description="Date associated with this chunk")
    chunk_date_end: datetime | None = Field(default=None, description="End date if chunk spans a period")
    
    # Metadata
    heading: str = Field(default="", description="Section heading if available")
    char_count: int = Field(default=0, description="Number of characters in content")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"ser_json_always": True}


class ExtractedFact(BaseModel):
    """A fact extracted from a document by LLM."""

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship type")
    object: str = Field(..., description="Object/value")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_context: str = Field(default="", description="Context from document")
    # Temporal information
    event_date: datetime | None = Field(default=None, description="Date when the event occurred")
    event_date_end: datetime | None = Field(default=None, description="End date for date ranges")
    date_type: DateType = Field(default=DateType.UNKNOWN, description="Type of date information")


class DocumentExtractionResult(BaseModel):
    """Result of LLM extraction from a document."""

    summary: str = Field(..., description="2-3 sentence summary")
    facts: list[ExtractedFact] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    domain: Domain = Field(default=Domain.GENERAL)


class TopicStats(BaseModel):
    """Statistics for a topic/entity."""

    name: str = Field(..., description="Topic or entity name")
    fact_count: int = Field(default=0, description="Number of related facts")
    document_count: int = Field(default=0, description="Number of related documents")
    last_updated: datetime | None = Field(default=None)
    
    model_config = {"ser_json_always": True}


class TopicSummary(BaseModel):
    """LLM-generated summary for a topic."""

    topic: str = Field(..., description="Topic name")
    summary: str = Field(..., description="Comprehensive summary")
    key_points: list[str] = Field(default_factory=list, description="Main points")
    related_entities: list[str] = Field(default_factory=list)
    fact_count: int = Field(default=0)
    document_count: int = Field(default=0)
    time_range: str = Field(default="", description="Time range of facts")
    generated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"ser_json_always": True}


class KnowledgeStats(BaseModel):
    """Overall knowledge base statistics."""

    total_facts: int = Field(default=0)
    total_documents: int = Field(default=0)
    total_sessions: int = Field(default=0)
    topics: list[TopicStats] = Field(default_factory=list)
    entities: list[TopicStats] = Field(default_factory=list)
    
    model_config = {"ser_json_always": True}


# ==================== Tag System Schemas ====================


class Tag(BaseModel):
    """A tag for categorizing insights and documents."""

    id: str = Field(..., description="Tag ID")
    name: str = Field(..., description="Normalized tag name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names merged into this tag")
    color: str | None = Field(default=None, description="UI display color (hex code)")
    description: str = Field(default="", description="Optional tag description")
    usage_count: int = Field(default=0, description="Number of items tagged")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"ser_json_always": True}


class TagStats(BaseModel):
    """Statistics for a tag."""

    id: str = Field(..., description="Tag ID")
    name: str = Field(..., description="Tag name")
    aliases: list[str] = Field(default_factory=list)
    color: str | None = Field(default=None)
    insight_count: int = Field(default=0, description="Number of insights with this tag")
    document_count: int = Field(default=0, description="Number of documents with this tag")
    total_count: int = Field(default=0, description="Total items tagged")
    avg_relevance: float = Field(default=0.0, description="Average relevance score")
    last_used: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"ser_json_always": True}


class TaggedItemType(str, Enum):
    """Type of tagged item."""

    INSIGHT = "insight"
    DOCUMENT = "document"
    CHUNK = "chunk"


class TaggedItem(BaseModel):
    """An item tagged with a specific tag, including relevance score."""

    item_id: str = Field(..., description="ID of the tagged item")
    item_type: TaggedItemType = Field(..., description="Type of item")
    relevance: float = Field(default=0.8, ge=0.0, le=1.0, description="Relevance score 0.0-1.0")
    
    # Item details (populated when fetched)
    title: str = Field(default="", description="Display title of the item")
    summary: str = Field(default="", description="Brief summary or preview")
    created_at: datetime | None = Field(default=None)
    
    # For insights
    subject: str | None = Field(default=None)
    predicate: str | None = Field(default=None)
    object: str | None = Field(default=None)
    
    # For documents
    filename: str | None = Field(default=None)

    model_config = {"ser_json_always": True}


class TagSuggestion(BaseModel):
    """A tag suggestion from LLM analysis."""

    name: str = Field(..., description="Suggested tag name")
    relevance: float = Field(default=0.8, ge=0.0, le=1.0, description="Relevance to the content")
    reason: str = Field(default="", description="Why this tag was suggested")
    existing_tag_id: str | None = Field(default=None, description="ID if matches existing tag")
    is_new: bool = Field(default=True, description="Whether this is a new tag")

    model_config = {"ser_json_always": True}


class TagSuggestionResult(BaseModel):
    """Result of tag suggestion analysis."""

    suggestions: list[TagSuggestion] = Field(default_factory=list)
    content_summary: str = Field(default="", description="Brief summary of analyzed content")
    existing_tags_matched: int = Field(default=0)
    new_tags_suggested: int = Field(default=0)

    model_config = {"ser_json_always": True}


class TagMergeRequest(BaseModel):
    """Request to merge multiple tags into one."""

    source_tag_ids: list[str] = Field(..., description="Tags to merge (will be deleted)")
    target_tag_id: str = Field(..., description="Tag to merge into (will be kept)")
    add_as_aliases: bool = Field(default=True, description="Add source names as aliases")

    model_config = {"frozen": True}


class TagMergeResult(BaseModel):
    """Result of tag merge operation."""

    target_tag: Tag = Field(..., description="The merged tag")
    merged_count: int = Field(default=0, description="Number of tags merged")
    items_updated: int = Field(default=0, description="Number of item relationships updated")
    aliases_added: list[str] = Field(default_factory=list)

    model_config = {"ser_json_always": True}


# ==================== Knowledge Graph Visualization ====================


class KnowledgeGraphNode(BaseModel):
    """A node in the knowledge graph visualization."""

    id: str = Field(..., description="Node ID (tag ID)")
    name: str = Field(..., description="Tag name")
    weight: float = Field(default=1.0, ge=0.0, description="Node importance/size weight")
    color: str | None = Field(default=None, description="Node color (hex)")
    usage_count: int = Field(default=0, description="Number of items tagged")
    insight_count: int = Field(default=0, description="Number of insights")
    document_count: int = Field(default=0, description="Number of documents")

    model_config = {"ser_json_always": True}


class KnowledgeGraphEdge(BaseModel):
    """An edge (relationship) in the knowledge graph visualization."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    weight: float = Field(default=1.0, ge=0.0, description="Edge strength/weight")
    co_occurrence_count: int = Field(default=0, description="Number of shared items")

    model_config = {"ser_json_always": True}


class KnowledgeGraphData(BaseModel):
    """Complete data for knowledge graph visualization."""

    nodes: list[KnowledgeGraphNode] = Field(default_factory=list)
    edges: list[KnowledgeGraphEdge] = Field(default_factory=list)
    total_tags: int = Field(default=0, description="Total number of tags")
    total_connections: int = Field(default=0, description="Total number of connections")

    model_config = {"ser_json_always": True}
