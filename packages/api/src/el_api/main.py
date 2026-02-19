"""EL API - FastAPI application for the EL interview agent."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from el_core import ELAgent
from el_core.document_parser import DocumentParser, truncate_content, chunk_document
from el_core.schemas import (
    ConsistencyIssue,
    ConsistencyIssueKind,
    ConsistencyIssueStatus,
    Document,
    DocumentChunk,
    DocumentStatus,
    ExtractedFact,
    KnowledgeStats,
    PendingQuestion,
    QuestionKind,
    QuestionStatus,
    Tag,
    TaggedItem,
    TaggedItemType,
    TagMergeRequest,
    TagMergeResult,
    TagStats,
    TagSuggestion,
    TagSuggestionResult,
    TopicSummary,
)
from el_core.llm.client import LLMClient
from el_core.stores.kg_store import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Global agent instance
agent: ELAgent | None = None
kg_store: KnowledgeGraphStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global agent, kg_store

    # Initialize LLM client
    llm_client = LLMClient()

    # Try to connect to Neo4j (optional)
    neo4j_uri = os.getenv("NEO4J_URI")
    if neo4j_uri:
        try:
            kg_store = KnowledgeGraphStore()
            await kg_store.connect()
            await kg_store.setup_indexes()
            logger.info("Connected to Neo4j knowledge graph")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j (will run without KG): {e}")
            kg_store = None

    # Initialize agent
    agent = ELAgent(llm_client=llm_client, kg_store=kg_store)
    logger.info("EL Agent initialized")

    yield

    # Cleanup
    if kg_store:
        await kg_store.close()


app = FastAPI(
    title="EL API",
    description="Curiosity-driven Interview Agent API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class StartSessionRequest(BaseModel):
    topic: str
    user_id: str = "default"
    language: str | None = None


class InsightDetail(BaseModel):
    """Detail of a saved insight."""
    subject: str
    predicate: str
    object: str
    domain: str
    confidence: float


class KnowledgeDetail(BaseModel):
    """Detail of used knowledge from past conversations."""
    subject: str
    predicate: str
    object: str
    domain: str
    created_at: str


class StartSessionResponse(BaseModel):
    session_id: str
    opening_message: str
    prior_knowledge_count: int = 0
    
    model_config = {"ser_json_always": True}


class MessageRequest(BaseModel):
    message: str


class ConsistencyIssueDetail(BaseModel):
    """Consistency issue detail for API response."""
    id: str | None = None  # Issue ID for resolution
    kind: str  # "contradiction" or "change"
    status: str = "unresolved"  # "unresolved", "resolved", "ignored"
    title: str
    fact_id: str | None = None  # ID of the related fact for resolution
    previous_text: str
    previous_source: str
    current_text: str
    current_source: str
    suggested_question: str
    explanation: str = ""
    confidence: float
    resolution: str | None = None
    resolved_at: str | None = None
    session_id: str | None = None
    created_at: str | None = None
    
    model_config = {"ser_json_always": True}


class AggregationSuggestion(BaseModel):
    """Suggestion to aggregate knowledge when threshold is exceeded."""
    topic: str
    fact_count: int
    threshold: int
    message: str


class PendingQuestionDetail(BaseModel):
    """Pending question detail for API response."""
    id: str
    kind: str  # "contradiction", "change", "missing", "clarification"
    question: str
    context: str = ""
    related_fact_id: str | None = None
    related_entity: str | None = None
    priority: int = 0
    status: str = "pending"
    answer: str | None = None
    session_id: str | None = None
    asked_at: str | None = None
    answered_at: str | None = None
    created_at: str | None = None

    model_config = {"ser_json_always": True}


class MessageResponse(BaseModel):
    response: str
    domain: str
    insights_saved: int
    knowledge_used: int
    # Extended details for UI display
    insights_detail: list[InsightDetail] = []
    knowledge_detail: list[KnowledgeDetail] = []
    # Consistency issues detected
    consistency_issues: list[ConsistencyIssueDetail] = []
    # Pending questions for the user
    pending_questions: list[PendingQuestionDetail] = []
    # IDs of questions answered in this turn
    questions_answered: list[str] = []
    # Aggregation suggestion (when threshold exceeded)
    aggregation_suggestion: AggregationSuggestion | None = None
    
    model_config = {"ser_json_always": True}


class SessionSummary(BaseModel):
    session_id: str
    topic: str
    language: str
    domain: str
    turn_count: int
    insights_saved: int
    knowledge_used: int


class SessionListItem(BaseModel):
    """Session item for list display."""
    id: str
    topic: str
    domain: str
    turn_count: int
    insights_count: int
    created_at: str
    updated_at: str


class SessionSummaryDetail(BaseModel):
    """Detailed session summary returned when ending a session."""
    id: str
    session_id: str
    content: str
    key_points: list[str] = []
    topics: list[str] = []
    entities_mentioned: list[str] = []
    turn_range: tuple[int, int] = (0, 0)
    created_at: str
    
    model_config = {"ser_json_always": True}


class SessionEndResponse(BaseModel):
    """Response when ending a session."""
    status: str
    session_id: str
    summary: SessionSummaryDetail | None = None
    
    model_config = {"ser_json_always": True}
    
    model_config = {"ser_json_always": True}


class ResumeSessionRequest(BaseModel):
    """Request to resume a session."""
    user_id: str = "default"


class ConversationMessage(BaseModel):
    """A message in conversation history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str | None = None


class ResumeSessionResponse(BaseModel):
    """Response when resuming a session."""
    session_id: str
    topic: str
    domain: str
    resume_message: str
    prior_insights: list[InsightDetail] = []
    conversation_history: list[ConversationMessage] = []
    turn_count: int = 0
    insights_count: int = 0
    
    model_config = {"ser_json_always": True}


# REST API endpoints
@app.get("/health")
async def health_check() -> dict[str, bool]:
    """Health check endpoint."""
    return {"ok": True, "agent_ready": agent is not None}


@app.post("/api/sessions", response_model=StartSessionResponse)
async def create_session(request: StartSessionRequest) -> StartSessionResponse:
    """Start a new conversation session."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session_id, opening_message = await agent.start_session(
        user_id=request.user_id,
        topic=request.topic,
        language=request.language,
    )

    # Get prior knowledge count from session
    session = agent.get_session(session_id)
    prior_knowledge_count = len(session.prior_knowledge) if session else 0

    return StartSessionResponse(
        session_id=session_id,
        opening_message=opening_message,
        prior_knowledge_count=prior_knowledge_count,
    )


@app.post("/api/sessions/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: str, request: MessageRequest) -> MessageResponse:
    """Send a message and get a response."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session = agent.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    response = await agent.respond(session_id, request.message)

    # Build detailed insight list
    insights_detail = [
        InsightDetail(
            subject=insight.subject,
            predicate=insight.predicate,
            object=insight.object,
            domain=insight.domain.value,
            confidence=insight.confidence,
        )
        for insight in response.insights_saved
    ]

    # Build detailed knowledge list
    knowledge_detail = [
        KnowledgeDetail(
            subject=item.subject,
            predicate=item.predicate,
            object=item.object,
            domain=item.domain.value,
            created_at=item.created_at.isoformat(),
        )
        for item in response.knowledge_used
    ]

    # Build consistency issues list with full details including IDs
    consistency_issues_detail = [
        ConsistencyIssueDetail(
            id=issue.id,
            kind=issue.kind.value,
            status=issue.status.value,
            title=issue.title,
            fact_id=issue.fact_id,
            previous_text=issue.previous_text,
            previous_source=issue.previous_source,
            current_text=issue.current_text,
            current_source=issue.current_source,
            suggested_question=issue.suggested_question,
            explanation=issue.explanation,
            confidence=issue.confidence,
            resolution=issue.resolution,
            resolved_at=issue.resolved_at.isoformat() if issue.resolved_at else None,
            session_id=issue.session_id,
            created_at=issue.created_at.isoformat() if issue.created_at else None,
        )
        for issue in response.consistency_issues
    ]

    # Check for aggregation suggestion
    aggregation_suggestion = None
    AGGREGATION_THRESHOLD = 15  # Suggest aggregation when a topic has this many facts
    
    if kg_store and response.detected_domain.value != "general":
        try:
            topics = await kg_store.get_all_topics(limit=5)
            for topic in topics:
                if topic.fact_count >= AGGREGATION_THRESHOLD:
                    aggregation_suggestion = AggregationSuggestion(
                        topic=topic.name,
                        fact_count=topic.fact_count,
                        threshold=AGGREGATION_THRESHOLD,
                        message=f"「{topic.name}」に{topic.fact_count}件の事実が蓄積されました。サマリー生成をお勧めします。",
                    )
                    break
        except Exception as e:
            logger.warning(f"Failed to check aggregation threshold: {e}")

    # Build pending questions list
    pending_questions_detail = [
        PendingQuestionDetail(
            id=q.id,
            kind=q.kind.value,
            question=q.question,
            context=q.context,
            related_fact_id=q.related_fact_id,
            related_entity=q.related_entity,
            priority=q.priority,
            status=q.status.value,
            answer=q.answer,
            session_id=q.session_id,
            asked_at=q.asked_at.isoformat() if q.asked_at else None,
            answered_at=q.answered_at.isoformat() if q.answered_at else None,
            created_at=q.created_at.isoformat() if q.created_at else None,
        )
        for q in response.pending_questions
    ]

    return MessageResponse(
        response=response.message,
        domain=response.detected_domain.value,
        insights_saved=len(response.insights_saved),
        knowledge_used=len(response.knowledge_used),
        insights_detail=insights_detail,
        knowledge_detail=knowledge_detail,
        consistency_issues=consistency_issues_detail,
        pending_questions=pending_questions_detail,
        questions_answered=response.questions_answered,
        aggregation_suggestion=aggregation_suggestion,
    )


@app.post("/api/sessions/{session_id}/messages/stream")
async def send_message_stream(session_id: str, request: MessageRequest) -> StreamingResponse:
    """Send a message and stream the response via SSE."""
    import json as _json

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session = agent.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        try:
            async for event in agent.respond_stream(session_id, request.message):
                yield f"data: {_json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    """Get session information."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    summary = await agent.get_session_summary(session_id)

    return SessionSummary(**summary)


@app.delete("/api/sessions/{session_id}", response_model=SessionEndResponse)
async def end_session(session_id: str) -> SessionEndResponse:
    """End a session and generate summary."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session, summary = await agent.end_session(session_id, generate_summary=True)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    summary_detail = None
    if summary:
        summary_detail = SessionSummaryDetail(
            id=summary.id,
            session_id=summary.session_id,
            content=summary.content,
            key_points=summary.key_points,
            topics=summary.topics,
            entities_mentioned=summary.entities_mentioned,
            turn_range=summary.turn_range,
            created_at=summary.created_at.isoformat(),
        )

    return SessionEndResponse(
        status="ended",
        session_id=session_id,
        summary=summary_detail,
    )


@app.post("/api/sessions/{session_id}/end", response_model=SessionEndResponse)
async def end_session_post(session_id: str) -> SessionEndResponse:
    """End a session and generate summary (POST method for easier UI integration)."""
    return await end_session(session_id)


@app.get("/api/users/{user_id}/sessions", response_model=list[SessionListItem])
async def list_user_sessions(user_id: str, limit: int = 10) -> list[SessionListItem]:
    """List recent sessions for a user."""
    if kg_store is None:
        # Return empty list if KG not available
        return []

    try:
        sessions = await kg_store.get_user_sessions(user_id, limit=limit)
        return [
            SessionListItem(
                id=s.id,
                topic=s.topic,
                domain=s.domain.value,
                turn_count=s.turn_count,
                insights_count=s.insights_count,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"Failed to list user sessions: {e}")
        return []


@app.post("/api/sessions/{session_id}/resume", response_model=ResumeSessionResponse)
async def resume_session(session_id: str, request: ResumeSessionRequest) -> ResumeSessionResponse:
    """Resume an existing session."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        resumed_id, resume_message, prior_insights = await agent.resume_session(
            session_id=session_id,
            user_id=request.user_id,
        )

        # Get session metadata for additional info
        session = agent.get_session(resumed_id)
        
        # Build insight details
        insights_detail = [
            InsightDetail(
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                domain=item.domain.value,
                confidence=item.confidence,
            )
            for item in prior_insights
        ]
        
        # Build conversation history
        conversation_history = []
        if session:
            for turn in session.turns:
                conversation_history.append(ConversationMessage(
                    role="user",
                    content=turn.user_message,
                    timestamp=turn.timestamp.isoformat() if hasattr(turn, 'timestamp') else None,
                ))
                conversation_history.append(ConversationMessage(
                    role="assistant",
                    content=turn.assistant_response,
                    timestamp=turn.timestamp.isoformat() if hasattr(turn, 'timestamp') else None,
                ))

        return ResumeSessionResponse(
            session_id=resumed_id,
            topic=session.topic if session else "",
            domain=session.domain.value if session else "general",
            resume_message=resume_message,
            prior_insights=insights_detail,
            conversation_history=conversation_history,
            turn_count=len(session.turns) if session else 0,
            insights_count=len(prior_insights),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume session: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume session")


# ==================== Question Management Endpoints ====================


class QuestionsListResponse(BaseModel):
    """Response for listing session questions."""
    session_id: str
    questions: list[PendingQuestionDetail] = []
    total: int = 0
    pending_count: int = 0
    answered_count: int = 0

    model_config = {"ser_json_always": True}


class SkipQuestionResponse(BaseModel):
    """Response for skipping a question."""
    question_id: str
    status: str = "skipped"
    message: str = ""

    model_config = {"ser_json_always": True}


@app.get("/api/sessions/{session_id}/questions", response_model=QuestionsListResponse)
async def get_session_questions(
    session_id: str,
    status: str | None = None,
) -> QuestionsListResponse:
    """Get all questions for a session.
    
    Args:
        session_id: Session ID.
        status: Optional filter by status (pending, answered, skipped, expired).
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session = agent.get_session(session_id)

    questions: list[PendingQuestion] = []

    if session is not None:
        # Get from in-memory session
        if status:
            questions = [q for q in session.pending_questions if q.status.value == status]
        else:
            questions = session.pending_questions
    elif kg_store is not None:
        # Fall back to KG store
        try:
            questions = await kg_store.get_pending_questions(session_id, status=status)
        except Exception as e:
            logger.warning(f"Failed to get questions from KG: {e}")
    else:
        raise HTTPException(status_code=404, detail="Session not found")

    questions_detail = [
        PendingQuestionDetail(
            id=q.id,
            kind=q.kind.value,
            question=q.question,
            context=q.context,
            related_fact_id=q.related_fact_id,
            related_entity=q.related_entity,
            priority=q.priority,
            status=q.status.value,
            answer=q.answer,
            session_id=q.session_id,
            asked_at=q.asked_at.isoformat() if q.asked_at else None,
            answered_at=q.answered_at.isoformat() if q.answered_at else None,
            created_at=q.created_at.isoformat() if q.created_at else None,
        )
        for q in questions
    ]

    pending_count = sum(1 for q in questions if q.status == QuestionStatus.PENDING)
    answered_count = sum(1 for q in questions if q.status == QuestionStatus.ANSWERED)

    return QuestionsListResponse(
        session_id=session_id,
        questions=questions_detail,
        total=len(questions),
        pending_count=pending_count,
        answered_count=answered_count,
    )


@app.post("/api/sessions/{session_id}/questions/{question_id}/skip", response_model=SkipQuestionResponse)
async def skip_question(session_id: str, question_id: str) -> SkipQuestionResponse:
    """Skip a pending question.
    
    Args:
        session_id: Session ID.
        question_id: Question ID to skip.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session = agent.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the question
    question = next((q for q in session.pending_questions if q.id == question_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    if question.status != QuestionStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Question is already {question.status.value}"
        )

    from datetime import datetime
    # Update in-memory status
    question.status = QuestionStatus.SKIPPED
    question.answered_at = datetime.now()

    # Update in KG
    if kg_store:
        try:
            await kg_store.update_question_status(
                question_id=question_id,
                status="skipped",
            )
        except Exception as e:
            logger.warning(f"Failed to update question status in KG: {e}")

    return SkipQuestionResponse(
        question_id=question_id,
        status="skipped",
        message="質問をスキップしました。" if session.language != "English" else "Question skipped.",
    )


# WebSocket for real-time chat
@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    if agent is None:
        await websocket.close(code=1013, reason="Agent not initialized")
        return

    session = agent.get_session(session_id)
    if session is None:
        await websocket.close(code=1008, reason="Session not found")
        return

    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            # Get response from agent
            response = await agent.respond(session_id, user_message)

            # Build detailed lists
            insights_detail = [
                {
                    "subject": insight.subject,
                    "predicate": insight.predicate,
                    "object": insight.object,
                    "domain": insight.domain.value,
                    "confidence": insight.confidence,
                }
                for insight in response.insights_saved
            ]

            knowledge_detail = [
                {
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "object": item.object,
                    "domain": item.domain.value,
                    "created_at": item.created_at.isoformat(),
                }
                for item in response.knowledge_used
            ]

            # Send response back
            await websocket.send_json({
                "type": "response",
                "message": response.message,
                "domain": response.detected_domain.value,
                "insights_saved": len(response.insights_saved),
                "knowledge_used": len(response.knowledge_used),
                "insights_detail": insights_detail,
                "knowledge_detail": knowledge_detail,
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")


# Knowledge/Fact Management API

class FactVersionDetail(BaseModel):
    """Fact version for API response."""
    id: str
    value: str
    source: str
    session_id: str | None
    valid_from: str
    valid_until: str | None
    created_at: str


class FactWithHistoryResponse(BaseModel):
    """Fact with history for API response."""
    id: str
    subject: str
    predicate: str
    current_value: str
    status: str
    domain: str
    confidence: float
    versions: list[FactVersionDetail]
    created_at: str
    updated_at: str


class ResolveConsistencyRequest(BaseModel):
    """Request to resolve a consistency issue."""
    resolution: str  # "accept_current" | "keep_previous" | "manual"
    new_value: str | None = None
    session_id: str | None = None


class ResolveConsistencyResponse(BaseModel):
    """Response from consistency resolution."""
    fact_id: str
    resolution: str
    previous_value: str
    new_value: str | None
    version_id: str | None


@app.get("/api/knowledge/facts/{fact_id}", response_model=FactWithHistoryResponse)
async def get_fact(fact_id: str) -> FactWithHistoryResponse:
    """Get a fact with its version history."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        fact = await kg_store.get_fact_with_history(fact_id)
        if fact is None:
            raise HTTPException(status_code=404, detail="Fact not found")

        return FactWithHistoryResponse(
            id=fact.id,
            subject=fact.subject,
            predicate=fact.predicate,
            current_value=fact.current_value,
            status=fact.status.value,
            domain=fact.domain.value,
            confidence=fact.confidence,
            versions=[
                FactVersionDetail(
                    id=v.id,
                    value=v.value,
                    source=v.source,
                    session_id=v.session_id,
                    valid_from=v.valid_from.isoformat(),
                    valid_until=v.valid_until.isoformat() if v.valid_until else None,
                    created_at=v.created_at.isoformat(),
                )
                for v in fact.versions
            ],
            created_at=fact.created_at.isoformat(),
            updated_at=fact.updated_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fact: {e}")
        raise HTTPException(status_code=500, detail="Failed to get fact")


@app.post("/api/consistency/resolve/{fact_id}", response_model=ResolveConsistencyResponse)
async def resolve_consistency(fact_id: str, request: ResolveConsistencyRequest) -> ResolveConsistencyResponse:
    """Resolve a consistency issue for a fact."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        result = await kg_store.resolve_consistency_issue(
            fact_id=fact_id,
            resolution=request.resolution,
            new_value=request.new_value,
            session_id=request.session_id,
        )

        return ResolveConsistencyResponse(
            fact_id=result["fact_id"],
            resolution=result["resolution"],
            previous_value=result["previous_value"],
            new_value=result["new_value"],
            version_id=result["version_id"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resolve consistency: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve consistency issue")


# ==================== Document Upload API ====================


class DocumentResponse(BaseModel):
    """Response model for document."""
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: str
    extracted_summary: str = ""
    extracted_facts_count: int = 0
    topics: list[str] = []
    entities: list[str] = []
    domain: str = "general"
    error_message: str | None = None
    created_at: str
    processed_at: str | None = None
    review_session_id: str | None = None


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: list[DocumentResponse]
    total: int


async def process_document_background(
    document_id: str,
    content: bytes,
    filename: str,
):
    """Background task to process uploaded document.
    
    Phase 2: Documents are chunked by date (or paragraph if no dates found),
    and each chunk is processed separately. Original content is preserved
    for accurate reference.
    """
    import uuid
    from datetime import datetime
    from el_core.schemas import Insight, Domain
    
    if kg_store is None or agent is None:
        logger.error("KG store or agent not available for document processing")
        return

    try:
        # Update status to processing
        await kg_store.update_document_status(document_id, DocumentStatus.PROCESSING)

        # Parse document
        parsed = DocumentParser.parse(content, filename)
        original_length = len(parsed.content)
        logger.info(f"Document {filename}: {original_length:,} chars, starting chunk processing")

        # Chunk the document by date (Phase 2)
        chunks = chunk_document(parsed, document_id)
        logger.info(f"Document {filename}: split into {len(chunks)} chunks")
        
        # Save chunks to knowledge graph
        await kg_store.save_chunks(chunks)
        
        # Process each chunk and extract facts
        all_facts: list[dict] = []
        all_topics: set[str] = set()
        all_entities: set[str] = set()
        detected_domain = Domain.GENERAL
        summaries: list[str] = []
        
        for chunk in chunks:
            # For small chunks, use content directly
            # For large chunks, truncate (should rarely happen with date-based chunking)
            chunk_content = truncate_content(chunk.content, max_chars=50000)
            
            try:
                # Extract information from this chunk
                extraction = await agent.extract_from_document(
                    chunk_content, 
                    f"{filename} (chunk {chunk.chunk_index + 1}/{len(chunks)})"
                )
                
                # Save extracted facts and link to chunk
                for fact in extraction.facts:
                    # Use chunk's date if fact has no specific date
                    event_date = fact.event_date or chunk.chunk_date
                    
                    insight = Insight(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        confidence=fact.confidence,
                        domain=extraction.domain,
                        event_date=event_date,
                        event_date_end=fact.event_date_end or chunk.chunk_date_end,
                        date_type=fact.date_type,
                    )
                    fact_id = await kg_store.save_insight(insight)
                    await kg_store.link_fact_to_document(fact_id, document_id)
                    # Link insight to its source chunk for accurate reference
                    await kg_store.link_insight_to_chunk(fact_id, chunk.id)
                    all_facts.append({
                        "id": fact_id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                    })
                
                # Aggregate topics and entities
                all_topics.update(extraction.topics)
                all_entities.update(extraction.entities)
                
                # Keep track of domain (use most common non-general domain)
                if extraction.domain != Domain.GENERAL:
                    detected_domain = extraction.domain
                
                # Only add meaningful summaries (skip placeholder defaults)
                if extraction.summary and extraction.summary not in ("ドキュメントの要約", ""):
                    summaries.append(extraction.summary)

                logger.info(
                    f"Chunk {chunk.chunk_index + 1}/{len(chunks)} of {filename}: "
                    f"{len(extraction.facts)} facts extracted, summary='{extraction.summary[:50]}...'"
                )
                    
            except Exception as chunk_error:
                logger.warning(f"Failed to process chunk {chunk.chunk_index} of {filename}: {chunk_error}", exc_info=True)
                continue
        
        # Generate overall summary
        if summaries:
            # If multiple chunks, combine summaries (or could use LLM to synthesize)
            if len(summaries) == 1:
                overall_summary = summaries[0]
            else:
                overall_summary = " ".join(summaries[:5])  # Limit to first 5 summaries
        else:
            overall_summary = f"ドキュメント '{filename}' から {len(all_facts)} 件のファクトを抽出しました。"

        # Update document with extraction results
        await kg_store.update_document_status(
            document_id,
            DocumentStatus.COMPLETED,
            extraction_result={
                "summary": overall_summary,
                "facts_count": len(all_facts),
                "topics": list(all_topics),
                "entities": list(all_entities),
                "domain": detected_domain.value,
            },
        )

        # Auto-tag the document based on its content
        try:
            # Build tagging content from summary, topics, entities and sample facts
            tag_content_parts = [overall_summary]
            if all_topics:
                tag_content_parts.append(f"トピック: {', '.join(list(all_topics)[:10])}")
            if all_entities:
                tag_content_parts.append(f"エンティティ: {', '.join(list(all_entities)[:10])}")
            if all_facts:
                sample_facts = [f"{f['subject']} {f['predicate']} {f['object']}" for f in all_facts[:5]]
                tag_content_parts.append(f"サンプルファクト: {'; '.join(sample_facts)}")
            
            tag_content = "\n".join(tag_content_parts)
            
            # Auto-tag the document
            tags_applied = await agent.auto_tag_document(
                document_id=document_id,
                content=tag_content,
                max_tags=5,
            )
            
            if tags_applied:
                tag_names = [tag.name for tag, _ in tags_applied]
                logger.info(f"Auto-tagged document {document_id} with: {tag_names}")
        except Exception as tag_error:
            logger.warning(f"Failed to auto-tag document {document_id}: {tag_error}")

        # Auto-tag extracted facts using LLM (batch processing)
        if all_facts:
            try:
                # Get existing tags for consistency
                all_tags = await kg_store.get_all_tag_stats()
                existing_tag_names = [t.name for t in sorted(all_tags, key=lambda x: -x.total_count)[:50]]
                
                # Process in batches of 10 facts
                batch_size = 10
                total_fact_tags = 0
                for i in range(0, len(all_facts), batch_size):
                    batch = all_facts[i:i + batch_size]
                    tagged = await agent.auto_tag_insights_batch(
                        insights=batch,
                        max_tags_per_insight=2,
                        existing_tags=existing_tag_names,
                    )
                    total_fact_tags += sum(len(tags) for tags in tagged.values())
                    # Update existing tags list
                    for tag_list in tagged.values():
                        for tag, _ in tag_list:
                            if tag.name not in existing_tag_names:
                                existing_tag_names.append(tag.name)
                
                logger.info(f"Auto-tagged {len(all_facts)} facts with {total_fact_tags} total tags")
            except Exception as fact_tag_error:
                logger.warning(f"Failed to auto-tag facts for document {document_id}: {fact_tag_error}")

        logger.info(f"Document {document_id} processed: {len(chunks)} chunks, {len(all_facts)} facts extracted")

    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")
        await kg_store.update_document_status(
            document_id,
            DocumentStatus.FAILED,
            error_message=str(e),
        )


@app.post("/api/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and process a document."""
    import uuid
    from datetime import datetime

    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    # Validate file type
    try:
        doc_type = DocumentParser.detect_type(file.filename or "unknown", file.content_type)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}"
        )

    # Read file content
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Create document record
    document_id = str(uuid.uuid4())
    document = Document(
        id=document_id,
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(content),
        status=DocumentStatus.UPLOADING,
        raw_content_preview=content[:500].decode("utf-8", errors="replace") if content else "",
    )

    # Save document metadata
    await kg_store.save_document(document)

    # Start background processing
    background_tasks.add_task(
        process_document_background,
        document_id,
        content,
        file.filename or "unknown",
    )

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status.value,
        created_at=document.created_at.isoformat(),
    )


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        doc_results = await kg_store.get_documents()
        return DocumentListResponse(
            documents=[
                DocumentResponse(
                    id=d.id,
                    filename=d.filename,
                    content_type=d.content_type,
                    size_bytes=d.size_bytes,
                    status=d.status.value,
                    extracted_summary=d.extracted_summary,
                    extracted_facts_count=d.extracted_facts_count,
                    topics=d.topics,
                    entities=d.entities,
                    domain=d.domain.value,
                    error_message=d.error_message,
                    created_at=d.created_at.isoformat(),
                    processed_at=d.processed_at.isoformat() if d.processed_at else None,
                    review_session_id=review_sid,
                )
                for d, review_sid in doc_results
            ],
            total=len(doc_results),
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")


@app.get("/api/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get document details."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        document = await kg_store.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            status=document.status.value,
            extracted_summary=document.extracted_summary,
            extracted_facts_count=document.extracted_facts_count,
            topics=document.topics,
            entities=document.entities,
            domain=document.domain.value,
            error_message=document.error_message,
            created_at=document.created_at.isoformat(),
            processed_at=document.processed_at.isoformat() if document.processed_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document")


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its extracted facts."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        deleted = await kg_store.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@app.post("/api/documents/{document_id}/review", response_model=StartSessionResponse)
async def start_document_review_session(
    document_id: str,
    request: StartSessionRequest,
):
    """Start a review session for an uploaded document.
    
    ドキュメントアップロード後に、抽出された事実と過去の知識を照合し、
    矛盾点や不足情報について確認するセッションを開始します。
    """
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        # Get document
        document = await kg_store.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if document.status != DocumentStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Document processing not completed. Status: {document.status.value}"
            )
        
        # Get extracted facts from document
        insights = await kg_store.get_insights_for_document(document_id)
        
        # Convert Insight to ExtractedFact for consistency check
        extracted_facts: list[ExtractedFact] = []
        for insight in insights:
            extracted_facts.append(ExtractedFact(
                subject=insight.subject,
                predicate=insight.predicate,
                object=insight.object,
                confidence=insight.confidence,
                source_context="",
                event_date=insight.event_date,
                event_date_end=insight.event_date_end,
                date_type=insight.date_type,
            ))
        
        # Create review session
        session_id, opening_message, consistency_issues = await agent.create_document_review_session(
            user_id=request.user_id,
            document_id=document_id,
            document_filename=document.filename,
            extracted_facts=extracted_facts,
            language=request.language,
        )
        
        # Get session for prior knowledge count
        session = agent.get_session(session_id)
        prior_knowledge_count = len(session.prior_knowledge) if session else 0
        
        return StartSessionResponse(
            session_id=session_id,
            opening_message=opening_message,
            prior_knowledge_count=prior_knowledge_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start document review session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start review session: {str(e)}")


# ==================== Document Chunk API (Phase 2) ====================


class ChunkResponse(BaseModel):
    """Response model for a document chunk."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    chunk_date: str | None = None
    chunk_date_end: str | None = None
    heading: str = ""
    char_count: int = 0
    created_at: str


class ChunkListResponse(BaseModel):
    """Response model for chunk list."""
    chunks: list[ChunkResponse]
    total: int


class ChunkSearchRequest(BaseModel):
    """Request model for chunk search."""
    query: str | None = None
    date: str | None = None  # Specific date (YYYY-MM-DD)
    start_date: str | None = None  # Range start
    end_date: str | None = None  # Range end
    limit: int = 10


@app.get("/api/documents/{document_id}/chunks", response_model=ChunkListResponse)
async def get_document_chunks(document_id: str):
    """Get all chunks for a document."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        chunks = await kg_store.get_chunks_by_document(document_id)
        return ChunkListResponse(
            chunks=[
                ChunkResponse(
                    id=c.id,
                    document_id=c.document_id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    chunk_date=c.chunk_date.isoformat() if c.chunk_date else None,
                    chunk_date_end=c.chunk_date_end.isoformat() if c.chunk_date_end else None,
                    heading=c.heading,
                    char_count=c.char_count,
                    created_at=c.created_at.isoformat(),
                )
                for c in chunks
            ],
            total=len(chunks),
        )
    except Exception as e:
        logger.error(f"Failed to get document chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document chunks")


@app.post("/api/chunks/search", response_model=ChunkListResponse)
async def search_chunks(request: ChunkSearchRequest):
    """Search document chunks by date or content.
    
    Phase 2: Returns original document content for accurate reference.
    Supports:
    - date: Exact date match (YYYY-MM-DD)
    - start_date + end_date: Date range search
    - query: Keyword search in content
    """
    from datetime import datetime
    
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        chunks: list[DocumentChunk] = []
        
        # Date-based search
        if request.date:
            try:
                target_date = datetime.fromisoformat(request.date)
                chunks = await kg_store.get_chunks_by_date(target_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
        elif request.start_date and request.end_date:
            try:
                start = datetime.fromisoformat(request.start_date)
                end = datetime.fromisoformat(request.end_date)
                chunks = await kg_store.get_chunks_by_date_range(start, end)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
        # Keyword search
        elif request.query:
            chunks = await kg_store.search_chunks(request.query, limit=request.limit)
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Provide either 'date', 'start_date+end_date', or 'query' for search."
            )
        
        return ChunkListResponse(
            chunks=[
                ChunkResponse(
                    id=c.id,
                    document_id=c.document_id,
                    content=c.content,
                    chunk_index=c.chunk_index,
                    chunk_date=c.chunk_date.isoformat() if c.chunk_date else None,
                    chunk_date_end=c.chunk_date_end.isoformat() if c.chunk_date_end else None,
                    heading=c.heading,
                    char_count=c.char_count,
                    created_at=c.created_at.isoformat(),
                )
                for c in chunks[:request.limit]
            ],
            total=len(chunks),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to search chunks")


@app.get("/api/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str):
    """Get a specific chunk by ID."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        chunk = await kg_store.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        return ChunkResponse(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            chunk_date=chunk.chunk_date.isoformat() if chunk.chunk_date else None,
            chunk_date_end=chunk.chunk_date_end.isoformat() if chunk.chunk_date_end else None,
            heading=chunk.heading,
            char_count=chunk.char_count,
            created_at=chunk.created_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunk: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chunk")


# ==================== Knowledge Aggregation API ====================


class TopicStatsResponse(BaseModel):
    """Response model for topic statistics."""
    name: str
    fact_count: int
    document_count: int
    last_updated: str | None = None


class TopicsListResponse(BaseModel):
    """Response model for topics list."""
    topics: list[TopicStatsResponse]
    total: int


class EntitiesListResponse(BaseModel):
    """Response model for entities list."""
    entities: list[TopicStatsResponse]
    total: int


class TopicSummaryResponse(BaseModel):
    """Response model for topic summary."""
    topic: str
    summary: str
    key_points: list[str]
    related_entities: list[str]
    fact_count: int
    document_count: int = 0
    time_range: str = ""
    generated_at: str


class KnowledgeStatsResponse(BaseModel):
    """Response model for overall knowledge stats."""
    total_facts: int
    total_documents: int
    total_sessions: int
    topics: list[TopicStatsResponse]
    entities: list[TopicStatsResponse]


class KnowledgeSearchItem(BaseModel):
    """A knowledge item in search results."""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    domain: str
    created_at: str
    event_date: str | None = None
    event_date_end: str | None = None
    date_type: str = "unknown"
    status: str = "active"


class KnowledgeSearchResponse(BaseModel):
    """Response model for knowledge search."""
    items: list[KnowledgeSearchItem]
    total: int
    query: str | None = None
    start_date: str | None = None
    end_date: str | None = None


@app.get("/api/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    query: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    domain: str | None = None,
    limit: int = 20,
):
    """Search knowledge base with optional date filtering.
    
    Args:
        query: Optional keyword search term.
        start_date: Start date filter (YYYY-MM-DD format).
        end_date: End date filter (YYYY-MM-DD format).
        domain: Optional domain filter.
        limit: Maximum number of results (default: 20).
    
    Returns:
        List of matching knowledge items.
    """
    from datetime import datetime
    from el_core.schemas import Domain as DomainEnum
    
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    
    try:
        # Parse dates
        parsed_start = None
        parsed_end = None
        
        if start_date:
            try:
                parsed_start = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
        
        if end_date:
            try:
                parsed_end = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")
        
        # Parse domain
        parsed_domain = None
        if domain:
            try:
                parsed_domain = DomainEnum(domain)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid domain: {domain}")
        
        # Search with date filters
        if parsed_start or parsed_end:
            # Use date range search
            items = await kg_store.search_by_date_range(
                start_date=parsed_start,
                end_date=parsed_end,
                query=query,
                domain=parsed_domain,
                limit=limit,
            )
        elif query:
            # Use keyword search
            items = await kg_store.search(
                query=query,
                limit=limit,
                domain=parsed_domain,
                start_date=parsed_start,
                end_date=parsed_end,
            )
        else:
            # No query and no date filter - return recent facts
            items = await kg_store.search_by_date_range(
                start_date=None,
                end_date=None,
                query=None,
                domain=parsed_domain,
                limit=limit,
            )
        
        return KnowledgeSearchResponse(
            items=[
                KnowledgeSearchItem(
                    id=item.id,
                    subject=item.subject,
                    predicate=item.predicate,
                    object=item.object,
                    confidence=item.confidence,
                    domain=item.domain.value,
                    created_at=item.created_at.isoformat(),
                    event_date=item.event_date.isoformat() if item.event_date else None,
                    event_date_end=item.event_date_end.isoformat() if item.event_date_end else None,
                    date_type=item.date_type.value,
                    status=item.status.value,
                )
                for item in items
            ],
            total=len(items),
            query=query,
            start_date=start_date,
            end_date=end_date,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search knowledge: {e}")
        raise HTTPException(status_code=500, detail="Failed to search knowledge")


@app.get("/api/knowledge/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats():
    """Get overall knowledge base statistics."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        stats = await kg_store.get_knowledge_stats()
        return KnowledgeStatsResponse(
            total_facts=stats.total_facts,
            total_documents=stats.total_documents,
            total_sessions=stats.total_sessions,
            topics=[
                TopicStatsResponse(
                    name=t.name,
                    fact_count=t.fact_count,
                    document_count=t.document_count,
                    last_updated=t.last_updated.isoformat() if t.last_updated else None,
                )
                for t in stats.topics
            ],
            entities=[
                TopicStatsResponse(
                    name=e.name,
                    fact_count=e.fact_count,
                    document_count=e.document_count,
                    last_updated=e.last_updated.isoformat() if e.last_updated else None,
                )
                for e in stats.entities
            ],
        )
    except Exception as e:
        logger.error(f"Failed to get knowledge stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get knowledge stats")


@app.get("/api/knowledge/topics", response_model=TopicsListResponse)
async def list_topics():
    """List all topics with statistics."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        topics = await kg_store.get_all_topics()
        return TopicsListResponse(
            topics=[
                TopicStatsResponse(
                    name=t.name,
                    fact_count=t.fact_count,
                    document_count=t.document_count,
                    last_updated=t.last_updated.isoformat() if t.last_updated else None,
                )
                for t in topics
            ],
            total=len(topics),
        )
    except Exception as e:
        logger.error(f"Failed to list topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to list topics")


@app.get("/api/knowledge/entities", response_model=EntitiesListResponse)
async def list_entities():
    """List all entities with statistics."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        entities = await kg_store.get_all_entities()
        return EntitiesListResponse(
            entities=[
                TopicStatsResponse(
                    name=e.name,
                    fact_count=e.fact_count,
                    document_count=e.document_count,
                    last_updated=e.last_updated.isoformat() if e.last_updated else None,
                )
                for e in entities
            ],
            total=len(entities),
        )
    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        raise HTTPException(status_code=500, detail="Failed to list entities")


@app.post("/api/knowledge/topics/{topic}/summarize", response_model=TopicSummaryResponse)
async def summarize_topic(topic: str):
    """Generate a comprehensive summary for a topic."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        # Get all facts for this topic
        facts = await kg_store.get_facts_by_topic(topic)
        
        # If no facts by domain, try by entity
        if not facts:
            facts = await kg_store.get_facts_by_entity(topic)

        # Generate summary using agent
        summary = await agent.generate_topic_summary(topic, facts)

        return TopicSummaryResponse(
            topic=summary.topic,
            summary=summary.summary,
            key_points=summary.key_points,
            related_entities=summary.related_entities,
            fact_count=summary.fact_count,
            time_range=summary.time_range,
            generated_at=summary.generated_at.isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to summarize topic: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate topic summary")


@app.post("/api/knowledge/entities/{entity}/summarize", response_model=TopicSummaryResponse)
async def summarize_entity(entity: str):
    """Generate a comprehensive summary for an entity."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        # Get all facts for this entity
        facts = await kg_store.get_facts_by_entity(entity)

        # Generate summary using agent
        summary = await agent.generate_topic_summary(entity, facts)

        return TopicSummaryResponse(
            topic=summary.topic,
            summary=summary.summary,
            key_points=summary.key_points,
            related_entities=summary.related_entities,
            fact_count=summary.fact_count,
            time_range=summary.time_range,
            generated_at=summary.generated_at.isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to summarize entity: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate entity summary")


# ==================== Conflict API Endpoints ====================


class ResolveConflictRequest(BaseModel):
    """Request to resolve a conflict."""
    resolution: str  # "accept_current", "keep_previous", "ignore"
    new_value: str | None = None  # For manual resolution


class ConflictListResponse(BaseModel):
    """Response for conflict list."""
    conflicts: list[ConsistencyIssueDetail]
    total: int
    unresolved: int


def _issue_to_detail(issue: ConsistencyIssue) -> ConsistencyIssueDetail:
    """Convert ConsistencyIssue to ConsistencyIssueDetail."""
    return ConsistencyIssueDetail(
        id=issue.id,
        kind=issue.kind.value,
        status=issue.status.value,
        title=issue.title,
        fact_id=issue.fact_id,
        previous_text=issue.previous_text,
        previous_source=issue.previous_source,
        current_text=issue.current_text,
        current_source=issue.current_source,
        suggested_question=issue.suggested_question,
        confidence=issue.confidence,
        resolution=issue.resolution,
        resolved_at=issue.resolved_at.isoformat() if issue.resolved_at else None,
        session_id=issue.session_id,
        created_at=issue.created_at.isoformat() if issue.created_at else None,
    )


@app.get("/api/conflicts", response_model=ConflictListResponse)
async def list_conflicts(include_resolved: bool = False, limit: int = 100):
    """List all consistency issues/conflicts.
    
    Args:
        include_resolved: Whether to include resolved conflicts.
        limit: Maximum number of conflicts to return.
    """
    if not kg_store:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    
    try:
        all_conflicts = await kg_store.get_all_conflicts(include_resolved=include_resolved, limit=limit)
        unresolved = await kg_store.get_unresolved_conflicts(limit=limit)
        
        return ConflictListResponse(
            conflicts=[_issue_to_detail(c) for c in all_conflicts],
            total=len(all_conflicts),
            unresolved=len(unresolved),
        )
    except Exception as e:
        logger.error(f"Failed to list conflicts: {e}")
        raise HTTPException(status_code=500, detail="Failed to list conflicts")


@app.get("/api/conflicts/{conflict_id}", response_model=ConsistencyIssueDetail)
async def get_conflict(conflict_id: str):
    """Get a specific conflict by ID."""
    if not kg_store:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    
    try:
        conflict = await kg_store.get_consistency_issue(conflict_id)
        if not conflict:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        return _issue_to_detail(conflict)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conflict: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conflict")


@app.post("/api/conflicts/{conflict_id}/resolve")
async def resolve_conflict(conflict_id: str, request: ResolveConflictRequest):
    """Resolve a consistency issue/conflict.
    
    Args:
        conflict_id: The conflict ID to resolve.
        request: Resolution details (resolution type and optional new value).
    """
    if not kg_store:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    
    # Validate resolution type
    valid_resolutions = ["accept_current", "keep_previous", "ignore"]
    if request.resolution not in valid_resolutions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid resolution. Must be one of: {valid_resolutions}"
        )
    
    try:
        success = await kg_store.mark_conflict_resolved(
            conflict_id=conflict_id,
            resolution=request.resolution,
            new_value=request.new_value,
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        # Get the updated conflict
        updated = await kg_store.get_consistency_issue(conflict_id)
        if updated:
            return _issue_to_detail(updated)
        
        return {"status": "resolved", "conflict_id": conflict_id, "resolution": request.resolution}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve conflict")


@app.get("/api/facts/{fact_id}/conflicts", response_model=list[ConsistencyIssueDetail])
async def get_fact_conflicts(fact_id: str):
    """Get all conflicts associated with a specific fact."""
    if not kg_store:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    
    try:
        conflicts = await kg_store.get_conflicts_for_fact(fact_id)
        return [_issue_to_detail(c) for c in conflicts]
    except Exception as e:
        logger.error(f"Failed to get conflicts for fact: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conflicts for fact")


# ==================== Tag API Endpoints ====================


class CreateTagRequest(BaseModel):
    """Request to create a new tag."""
    name: str
    color: str | None = None
    description: str = ""


class UpdateTagRequest(BaseModel):
    """Request to update a tag."""
    name: str | None = None
    color: str | None = None
    description: str | None = None
    aliases: list[str] | None = None


class TagItemRequest(BaseModel):
    """Request to tag an item."""
    tag_id: str
    relevance: float = 0.8


class TagSuggestRequest(BaseModel):
    """Request for tag suggestions."""
    content: str
    max_tags: int = 5


class MergeTagsRequest(BaseModel):
    """Request to merge tags."""
    source_tag_ids: list[str]
    target_tag_id: str
    add_as_aliases: bool = True


class TagResponse(BaseModel):
    """Response for a single tag."""
    id: str
    name: str
    aliases: list[str]
    color: str | None
    description: str
    usage_count: int
    created_at: str
    updated_at: str


class TagStatsResponse(BaseModel):
    """Response for tag statistics."""
    id: str
    name: str
    aliases: list[str]
    color: str | None
    insight_count: int
    document_count: int
    total_count: int
    avg_relevance: float
    last_used: str | None
    created_at: str


class TaggedItemResponse(BaseModel):
    """Response for a tagged item."""
    item_id: str
    item_type: str
    relevance: float
    title: str
    summary: str
    created_at: str | None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    filename: str | None = None


class TagSuggestionResponse(BaseModel):
    """Response for a tag suggestion."""
    name: str
    relevance: float
    reason: str
    existing_tag_id: str | None
    is_new: bool


class TagSuggestionsResponse(BaseModel):
    """Response for tag suggestions."""
    suggestions: list[TagSuggestionResponse]
    content_summary: str
    existing_tags_matched: int
    new_tags_suggested: int


class TagMergeResultResponse(BaseModel):
    """Response for tag merge operation."""
    target_tag: TagResponse
    merged_count: int
    items_updated: int
    aliases_added: list[str]


def _tag_to_response(tag: Tag) -> TagResponse:
    """Convert Tag to TagResponse."""
    return TagResponse(
        id=tag.id,
        name=tag.name,
        aliases=tag.aliases,
        color=tag.color,
        description=tag.description,
        usage_count=tag.usage_count,
        created_at=tag.created_at.isoformat(),
        updated_at=tag.updated_at.isoformat(),
    )


def _tag_stats_to_response(stats: TagStats) -> TagStatsResponse:
    """Convert TagStats to TagStatsResponse."""
    return TagStatsResponse(
        id=stats.id,
        name=stats.name,
        aliases=stats.aliases,
        color=stats.color,
        insight_count=stats.insight_count,
        document_count=stats.document_count,
        total_count=stats.total_count,
        avg_relevance=stats.avg_relevance,
        last_used=stats.last_used.isoformat() if stats.last_used else None,
        created_at=stats.created_at.isoformat(),
    )


def _tagged_item_to_response(item: TaggedItem) -> TaggedItemResponse:
    """Convert TaggedItem to TaggedItemResponse."""
    return TaggedItemResponse(
        item_id=item.item_id,
        item_type=item.item_type.value,
        relevance=item.relevance,
        title=item.title,
        summary=item.summary,
        created_at=item.created_at.isoformat() if item.created_at else None,
        subject=item.subject,
        predicate=item.predicate,
        object=item.object,
        filename=item.filename,
    )


# ==================== Knowledge Graph Visualization ====================


class KnowledgeGraphNodeResponse(BaseModel):
    """Response for a knowledge graph node (tag)."""
    id: str
    name: str
    weight: float
    color: str | None
    usage_count: int
    insight_count: int
    document_count: int


class KnowledgeGraphEdgeResponse(BaseModel):
    """Response for a knowledge graph edge (relationship)."""
    source: str
    target: str
    weight: float
    co_occurrence_count: int


class KnowledgeGraphDataResponse(BaseModel):
    """Response for knowledge graph visualization data."""
    nodes: list[KnowledgeGraphNodeResponse]
    edges: list[KnowledgeGraphEdgeResponse]
    total_tags: int
    total_connections: int


@app.get("/api/knowledge-graph", tags=["knowledge-graph"])
async def get_knowledge_graph(
    min_usage: int = 0,
    limit: int = 500,
) -> KnowledgeGraphDataResponse:
    """Get knowledge graph data for visualization.

    Returns nodes (tags) and edges (co-occurrence relationships).
    Tags that appear on the same Insight/Document are considered connected.
    """
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        data = await kg_store.get_knowledge_graph_data(
            min_usage_count=min_usage,
            limit=limit,
        )

        return KnowledgeGraphDataResponse(
            nodes=[
                KnowledgeGraphNodeResponse(
                    id=node.id,
                    name=node.name,
                    weight=node.weight,
                    color=node.color,
                    usage_count=node.usage_count,
                    insight_count=node.insight_count,
                    document_count=node.document_count,
                )
                for node in data.nodes
            ],
            edges=[
                KnowledgeGraphEdgeResponse(
                    source=edge.source,
                    target=edge.target,
                    weight=edge.weight,
                    co_occurrence_count=edge.co_occurrence_count,
                )
                for edge in data.edges
            ],
            total_tags=data.total_tags,
            total_connections=data.total_connections,
        )
    except Exception as e:
        logger.error(f"Failed to get knowledge graph data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get knowledge graph data")


@app.get("/api/tags", tags=["tags"])
async def list_tags(limit: int = 100) -> list[TagStatsResponse]:
    """Get all tags with statistics, ordered by usage count."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        stats = await kg_store.get_all_tag_stats(limit=limit)
        return [_tag_stats_to_response(s) for s in stats]
    except Exception as e:
        logger.error(f"Failed to list tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to list tags")


@app.post("/api/tags", tags=["tags"])
async def create_tag(request: CreateTagRequest) -> TagResponse:
    """Create a new tag."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        # Check if tag already exists
        existing = await kg_store.get_tag_by_name(request.name)
        if existing:
            raise HTTPException(status_code=409, detail=f"Tag '{request.name}' already exists")

        tag = await kg_store.create_tag(
            name=request.name,
            color=request.color,
            description=request.description,
        )
        return _tag_to_response(tag)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create tag: {e}")
        raise HTTPException(status_code=500, detail="Failed to create tag")


# NOTE: These endpoints must be defined BEFORE /api/tags/{tag_id} to avoid routing conflicts
@app.get("/api/tags/search", tags=["tags"])
async def search_tags(query: str, limit: int = 10) -> list[TagResponse]:
    """Search tags by name or alias."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        tags = await kg_store.search_tags(query, limit=limit)
        return [_tag_to_response(tag) for tag in tags]
    except Exception as e:
        logger.error(f"Failed to search tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to search tags")


@app.post("/api/tags/suggest", tags=["tags"])
async def suggest_tags_endpoint(request: TagSuggestRequest) -> TagSuggestionsResponse:
    """Suggest tags for given content using LLM."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        result = await agent.suggest_tags(
            content=request.content,
            max_tags=request.max_tags,
        )

        return TagSuggestionsResponse(
            suggestions=[
                TagSuggestionResponse(
                    name=s.name,
                    relevance=s.relevance,
                    reason=s.reason,
                    existing_tag_id=s.existing_tag_id,
                    is_new=s.is_new,
                )
                for s in result.suggestions
            ],
            content_summary=result.content_summary,
            existing_tags_matched=result.existing_tags_matched,
            new_tags_suggested=result.new_tags_suggested,
        )
    except Exception as e:
        logger.error(f"Failed to suggest tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to suggest tags")


@app.post("/api/tags/merge", tags=["tags"])
async def merge_tags_endpoint(request: MergeTagsRequest) -> TagMergeResultResponse:
    """Merge multiple tags into one."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        tag, items_updated = await kg_store.merge_tags(
            source_tag_ids=request.source_tag_ids,
            target_tag_id=request.target_tag_id,
            add_as_aliases=request.add_as_aliases,
        )

        if tag is None:
            raise HTTPException(status_code=404, detail="Target tag not found")

        return TagMergeResultResponse(
            target_tag=_tag_to_response(tag),
            merged_count=len(request.source_tag_ids),
            items_updated=items_updated,
            aliases_added=[],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to merge tags")


@app.get("/api/tags/{tag_id}", tags=["tags"])
async def get_tag(tag_id: str) -> TagStatsResponse:
    """Get a tag by ID with statistics."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        stats = await kg_store.get_tag_stats(tag_id)
        if stats is None:
            raise HTTPException(status_code=404, detail="Tag not found")
        return _tag_stats_to_response(stats)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tag: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tag")


@app.put("/api/tags/{tag_id}", tags=["tags"])
async def update_tag(tag_id: str, request: UpdateTagRequest) -> TagResponse:
    """Update a tag."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        tag = await kg_store.update_tag(
            tag_id=tag_id,
            name=request.name,
            color=request.color,
            description=request.description,
            aliases=request.aliases,
        )
        if tag is None:
            raise HTTPException(status_code=404, detail="Tag not found")
        return _tag_to_response(tag)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update tag: {e}")
        raise HTTPException(status_code=500, detail="Failed to update tag")


@app.delete("/api/tags/{tag_id}", tags=["tags"])
async def delete_tag(tag_id: str) -> dict[str, bool]:
    """Delete a tag."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        deleted = await kg_store.delete_tag(tag_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Tag not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tag: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete tag")


@app.get("/api/tags/{tag_id}/items", tags=["tags"])
async def get_tagged_items(
    tag_id: str,
    item_type: str | None = None,
    limit: int = 50,
) -> list[TaggedItemResponse]:
    """Get items tagged with a specific tag."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        # Parse item type
        parsed_type: TaggedItemType | None = None
        if item_type:
            try:
                parsed_type = TaggedItemType(item_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid item_type. Must be one of: {[t.value for t in TaggedItemType]}"
                )

        items = await kg_store.get_items_by_tag(tag_id, item_type=parsed_type, limit=limit)
        return [_tagged_item_to_response(item) for item in items]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tagged items: {e}")
        raise HTTPException(status_code=500, detail="Failed to get tagged items")


@app.post("/api/insights/{insight_id}/tags", tags=["tags"])
async def tag_insight(insight_id: str, request: TagItemRequest) -> dict[str, str]:
    """Add a tag to an insight."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        await kg_store.tag_insight(
            insight_id=insight_id,
            tag_id=request.tag_id,
            relevance=request.relevance,
        )
        return {"status": "tagged", "insight_id": insight_id, "tag_id": request.tag_id}
    except Exception as e:
        logger.error(f"Failed to tag insight: {e}")
        raise HTTPException(status_code=500, detail="Failed to tag insight")


@app.delete("/api/insights/{insight_id}/tags/{tag_id}", tags=["tags"])
async def untag_insight(insight_id: str, tag_id: str) -> dict[str, bool]:
    """Remove a tag from an insight."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        removed = await kg_store.untag_insight(insight_id, tag_id)
        return {"removed": removed}
    except Exception as e:
        logger.error(f"Failed to untag insight: {e}")
        raise HTTPException(status_code=500, detail="Failed to untag insight")


@app.get("/api/insights/{insight_id}/tags", tags=["tags"])
async def get_insight_tags(insight_id: str) -> list[dict[str, Any]]:
    """Get all tags for an insight."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        tags_with_relevance = await kg_store.get_tags_for_insight(insight_id)
        return [
            {
                "tag": _tag_to_response(tag).model_dump(),
                "relevance": relevance,
            }
            for tag, relevance in tags_with_relevance
        ]
    except Exception as e:
        logger.error(f"Failed to get insight tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to get insight tags")


@app.post("/api/documents/{document_id}/tags", tags=["tags"])
async def tag_document(document_id: str, request: TagItemRequest) -> dict[str, str]:
    """Add a tag to a document."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        await kg_store.tag_document(
            document_id=document_id,
            tag_id=request.tag_id,
            relevance=request.relevance,
        )
        return {"status": "tagged", "document_id": document_id, "tag_id": request.tag_id}
    except Exception as e:
        logger.error(f"Failed to tag document: {e}")
        raise HTTPException(status_code=500, detail="Failed to tag document")


@app.delete("/api/documents/{document_id}/tags/{tag_id}", tags=["tags"])
async def untag_document(document_id: str, tag_id: str) -> dict[str, bool]:
    """Remove a tag from a document."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        removed = await kg_store.untag_document(document_id, tag_id)
        return {"removed": removed}
    except Exception as e:
        logger.error(f"Failed to untag document: {e}")
        raise HTTPException(status_code=500, detail="Failed to untag document")


@app.get("/api/documents/{document_id}/tags", tags=["tags"])
async def get_document_tags(document_id: str) -> list[dict[str, Any]]:
    """Get all tags for a document."""
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        tags_with_relevance = await kg_store.get_tags_for_document(document_id)
        return [
            {
                "tag": _tag_to_response(tag).model_dump(),
                "relevance": relevance,
            }
            for tag, relevance in tags_with_relevance
        ]
    except Exception as e:
        logger.error(f"Failed to get document tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to get document tags")


@app.post("/api/documents/{document_id}/tags/refresh", tags=["tags"])
async def refresh_document_tags(
    document_id: str, 
    clear_existing: bool = True
) -> dict[str, Any]:
    """Re-generate and assign tags for a document using LLM.
    
    Args:
        document_id: Document ID
        clear_existing: If True, remove existing tags before re-tagging (default: True)
    """
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        # Get document info
        doc = await kg_store.get_document(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document content from chunks or summary
        content = ""
        chunks = await kg_store.get_chunks_by_document(document_id)
        if chunks:
            # Use chunk contents
            content = "\n\n".join([c.content for c in chunks[:10]])  # Limit to first 10 chunks
        elif doc.extracted_summary:
            # Fall back to extracted summary
            content = doc.extracted_summary
        
        if not content:
            raise HTTPException(status_code=400, detail="No content available for tagging")
        
        # Clear existing tags if requested
        if clear_existing:
            existing_tags = await kg_store.get_tags_for_document(document_id)
            for tag, _ in existing_tags:
                await kg_store.untag_document(document_id, tag.id)
        
        # Re-generate tags using LLM
        assigned_tags = await agent.auto_tag_document(document_id, content)
        
        logger.info(f"Refreshed tags for document {document_id}: {len(assigned_tags)} tags assigned")
        
        return {
            "status": "refreshed",
            "document_id": document_id,
            "tags_assigned": len(assigned_tags),
            "tags": [{"id": tag.id, "name": tag.name, "relevance": relevance} for tag, relevance in assigned_tags]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh document tags: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh document tags: {str(e)}")


@app.post("/api/documents/{document_id}/facts/extract", tags=["documents"])
async def extract_document_facts(
    document_id: str,
) -> dict[str, Any]:
    """Extract (or re-extract) facts from a document using LLM.
    
    This reads the document's stored chunks and runs LLM extraction on each chunk.
    If facts already exist for this document, they are deleted and re-extracted.
    After extraction, facts are automatically tagged.
    
    Args:
        document_id: Document ID
    """
    import uuid
    from datetime import datetime
    from el_core.schemas import Insight, Domain

    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        # Verify document exists
        document = await kg_store.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get document chunks
        chunks = await kg_store.get_chunks_by_document(document_id)
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="ドキュメントのチャンクが見つかりません。ドキュメントを再アップロードしてください。"
            )

        # Delete existing facts for this document (re-extraction)
        deleted_count = await kg_store.delete_document_insights(document_id)
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing facts for document {document_id} before re-extraction")

        # Process each chunk and extract facts
        all_facts: list[dict] = []
        all_topics: set[str] = set()
        all_entities: set[str] = set()
        detected_domain = Domain.GENERAL
        summaries: list[str] = []
        extraction_errors: list[str] = []

        for chunk in chunks:
            chunk_content = truncate_content(chunk.content, max_chars=50000)

            try:
                extraction = await agent.extract_from_document(
                    chunk_content,
                    f"{document.filename} (chunk {chunk.chunk_index + 1}/{len(chunks)})"
                )

                logger.info(
                    f"Chunk {chunk.chunk_index + 1}/{len(chunks)} of {document.filename}: "
                    f"extracted {len(extraction.facts)} facts, summary='{extraction.summary[:50]}...'"
                )

                for fact in extraction.facts:
                    event_date = fact.event_date or chunk.chunk_date

                    insight = Insight(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        confidence=fact.confidence,
                        domain=extraction.domain,
                        event_date=event_date,
                        event_date_end=fact.event_date_end or chunk.chunk_date_end,
                        date_type=fact.date_type,
                    )
                    fact_id = await kg_store.save_insight(insight)
                    await kg_store.link_fact_to_document(fact_id, document_id)
                    await kg_store.link_insight_to_chunk(fact_id, chunk.id)
                    all_facts.append({
                        "id": fact_id,
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                    })

                all_topics.update(extraction.topics)
                all_entities.update(extraction.entities)

                if extraction.domain != Domain.GENERAL:
                    detected_domain = extraction.domain

                if extraction.summary and extraction.summary != "ドキュメントの要約":
                    summaries.append(extraction.summary)

            except Exception as chunk_error:
                error_msg = f"Chunk {chunk.chunk_index + 1}: {chunk_error}"
                logger.warning(f"Failed to process chunk of {document.filename}: {chunk_error}")
                extraction_errors.append(error_msg)
                continue

        # Update document summary
        if summaries:
            overall_summary = " ".join(summaries[:5]) if len(summaries) > 1 else summaries[0]
        else:
            overall_summary = f"ドキュメント '{document.filename}' から {len(all_facts)} 件のファクトを抽出しました。"

        await kg_store.update_document_status(
            document_id,
            DocumentStatus.COMPLETED,
            extraction_result={
                "summary": overall_summary,
                "facts_count": len(all_facts),
                "topics": list(all_topics),
                "entities": list(all_entities),
                "domain": detected_domain.value,
            },
        )

        # Auto-tag extracted facts
        total_fact_tags = 0
        if all_facts:
            try:
                all_tags = await kg_store.get_all_tag_stats()
                existing_tag_names = [t.name for t in sorted(all_tags, key=lambda x: -x.total_count)[:50]]

                batch_size = 10
                for i in range(0, len(all_facts), batch_size):
                    batch = all_facts[i:i + batch_size]
                    try:
                        tagged = await agent.auto_tag_insights_batch(
                            insights=batch,
                            max_tags_per_insight=2,
                            existing_tags=existing_tag_names,
                        )
                        total_fact_tags += sum(len(tags) for tags in tagged.values())
                        for tag_list in tagged.values():
                            for tag, _ in tag_list:
                                if tag.name not in existing_tag_names:
                                    existing_tag_names.append(tag.name)
                    except Exception as batch_error:
                        logger.warning(f"Batch tagging failed: {batch_error}")
            except Exception as tag_error:
                logger.warning(f"Failed to auto-tag facts: {tag_error}")

        logger.info(
            f"Document {document_id} fact extraction complete: "
            f"{len(chunks)} chunks, {len(all_facts)} facts, {total_fact_tags} tags"
        )

        return {
            "status": "extracted",
            "document_id": document_id,
            "chunks_processed": len(chunks),
            "facts_extracted": len(all_facts),
            "facts_tagged": total_fact_tags,
            "topics": list(all_topics),
            "entities": list(all_entities),
            "summary": overall_summary,
            "errors": extraction_errors,
            "previous_facts_deleted": deleted_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract facts from document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ファクト抽出に失敗しました: {str(e)}")


@app.get("/api/facts/recent", tags=["facts"])
async def get_recent_facts(
    limit: int = 50,
) -> dict[str, Any]:
    """Get recent facts from all sources (documents and sessions).
    
    Args:
        limit: Maximum number of facts to return (default: 50)
    """
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")

    try:
        facts = await kg_store.get_all_recent_insights(limit=limit)
        return {
            "facts": facts,
            "total": len(facts),
        }
    except Exception as e:
        logger.error(f"Failed to get recent facts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recent facts")


@app.post("/api/documents/{document_id}/facts/tags/refresh", tags=["tags"])
async def refresh_document_facts_tags(
    document_id: str,
    clear_existing: bool = True
) -> dict[str, Any]:
    """Re-generate and assign tags for all facts in a document using LLM.
    
    Args:
        document_id: Document ID
        clear_existing: If True, remove existing tags before re-tagging (default: True)
    """
    if kg_store is None:
        raise HTTPException(status_code=503, detail="Knowledge graph not available")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        # Get document's linked facts
        facts = await kg_store.get_insights_for_document(document_id)
        if not facts:
            return {
                "status": "no_facts",
                "document_id": document_id,
                "facts_tagged": 0,
                "total_tags": 0,
            }
        
        # Clear existing tags if requested
        if clear_existing:
            for fact in facts:
                existing_tags = await kg_store.get_tags_for_insight(fact.id)
                for tag, _ in existing_tags:
                    await kg_store.untag_insight(fact.id, tag.id)
        
        # Prepare facts for batch tagging
        facts_data = [
            {
                "id": fact.id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
            }
            for fact in facts
        ]
        
        # Get existing tags for consistency
        all_tags = await kg_store.get_all_tag_stats()
        existing_tag_names = [t.name for t in sorted(all_tags, key=lambda x: -x.total_count)[:50]]
        
        # Batch tag facts
        batch_size = 10
        all_tagged: dict[str, Any] = {}
        for i in range(0, len(facts_data), batch_size):
            batch = facts_data[i:i + batch_size]
            try:
                tagged = await agent.auto_tag_insights_batch(
                    insights=batch,
                    max_tags_per_insight=2,
                    existing_tags=existing_tag_names,
                )
                all_tagged.update(tagged)
                # Update existing tags list with newly created tags
                for tag_list in tagged.values():
                    for tag, _ in tag_list:
                        if tag.name not in existing_tag_names:
                            existing_tag_names.append(tag.name)
            except Exception as batch_error:
                logger.warning(f"Batch tagging failed: {batch_error}")
        
        total_tags = sum(len(tags) for tags in all_tagged.values())
        
        logger.info(f"Refreshed tags for {len(facts)} facts in document {document_id}: {total_tags} tags assigned")
        
        return {
            "status": "refreshed",
            "document_id": document_id,
            "facts_tagged": len(all_tagged),
            "total_tags": total_tags,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh facts tags: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh facts tags: {str(e)}")


# Serve static files (Web UI)
WEB_DIR = Path(__file__).parent.parent.parent.parent / "web"
logger.info(f"WEB_DIR resolved to: {WEB_DIR}")
logger.info(f"WEB_DIR exists: {WEB_DIR.exists()}")
if WEB_DIR.exists():
    index_file = WEB_DIR / "index.html"
    logger.info(f"Index file: {index_file}, exists: {index_file.exists()}")
    
    # Check if Documents section exists in the file
    if index_file.exists():
        content = index_file.read_text()
        has_documents = "upload-dropzone-main" in content
        logger.info(f"Index file has Documents section: {has_documents}")
    
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/")
    async def serve_index():
        """Serve the main HTML page."""
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


def run():
    """Run the API server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "el_api.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development",
    )


if __name__ == "__main__":
    run()
