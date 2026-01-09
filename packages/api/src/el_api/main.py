"""EL API - FastAPI application for the EL interview agent."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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


class StartSessionResponse(BaseModel):
    session_id: str
    opening_message: str


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str
    domain: str
    insights_saved: int
    knowledge_used: int


class SessionSummary(BaseModel):
    session_id: str
    topic: str
    language: str
    domain: str
    turn_count: int
    insights_saved: int
    knowledge_used: int


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

    return StartSessionResponse(
        session_id=session_id,
        opening_message=opening_message,
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

    return MessageResponse(
        response=response.message,
        domain=response.detected_domain.value,
        insights_saved=len(response.insights_saved),
        knowledge_used=len(response.knowledge_used),
    )


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    """Get session information."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    summary = await agent.get_session_summary(session_id)

    return SessionSummary(**summary)


@app.delete("/api/sessions/{session_id}")
async def end_session(session_id: str) -> dict[str, str]:
    """End a session."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    session = agent.end_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "ended", "session_id": session_id}


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

            # Send response back
            await websocket.send_json({
                "type": "response",
                "message": response.message,
                "domain": response.detected_domain.value,
                "insights_saved": len(response.insights_saved),
                "knowledge_used": len(response.knowledge_used),
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")


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
