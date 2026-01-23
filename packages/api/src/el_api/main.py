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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from el_core import ELAgent
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
    kind: str  # "contradiction" or "change"
    title: str
    fact_id: str | None = None  # ID of the related fact for resolution
    previous_text: str
    previous_source: str
    current_text: str
    current_source: str
    suggested_question: str
    confidence: float
    
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


class ResumeSessionResponse(BaseModel):
    """Response when resuming a session."""
    session_id: str
    topic: str
    domain: str
    resume_message: str
    prior_insights: list[InsightDetail] = []
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

    # Build consistency issues list
    consistency_issues_detail = [
        ConsistencyIssueDetail(
            kind=issue.kind.value,
            title=issue.title,
            fact_id=issue.fact_id,
            previous_text=issue.previous_text,
            previous_source=issue.previous_source,
            current_text=issue.current_text,
            current_source=issue.current_source,
            suggested_question=issue.suggested_question,
            confidence=issue.confidence,
        )
        for issue in response.consistency_issues
    ]

    return MessageResponse(
        response=response.message,
        domain=response.detected_domain.value,
        insights_saved=len(response.insights_saved),
        knowledge_used=len(response.knowledge_used),
        insights_detail=insights_detail,
        knowledge_detail=knowledge_detail,
        consistency_issues=consistency_issues_detail,
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

        return ResumeSessionResponse(
            session_id=resumed_id,
            topic=session.topic if session else "",
            domain=session.domain.value if session else "general",
            resume_message=resume_message,
            prior_insights=insights_detail,
            turn_count=len(session.turns) if session else 0,
            insights_count=len(prior_insights),
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resume session: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume session")


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


# Serve static files (Web UI)
WEB_DIR = Path(__file__).parent.parent.parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/")
    async def serve_index():
        """Serve the main HTML page."""
        return FileResponse(WEB_DIR / "index.html")


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
